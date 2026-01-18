# ------------------------------------------------------------------------
# AIMET FP16 Mixed Precision Training for NAFNet
# Supports a16w16 (16-bit activation, 16-bit weight) quantization
# ------------------------------------------------------------------------
import argparse
import datetime
import logging
import math
import random
import time
import torch
import numpy as np
from os import path as osp

from basicsr.data import create_dataloader, create_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import create_model
from basicsr.utils import (MessageLogger, check_resume, get_env_info,
                           get_root_logger, get_time_str, init_tb_logger,
                           init_wandb_logger, make_exp_dirs, mkdir_and_rename,
                           set_random_seed)
from basicsr.utils.dist_util import get_dist_info, init_dist
from basicsr.utils.options import dict2str, parse

# PyTorch AMP for FP16
from torch.cuda.amp import autocast, GradScaler

# AIMET imports
try:
    from aimet_torch.quantsim import QuantizationSimModel
    from aimet_torch.qc_quantize_op import QcQuantizeWrapper
    from aimet_common.defs import QuantScheme
    AIMET_AVAILABLE = True
except ImportError:
    print("Warning: AIMET not installed. Will use PyTorch native FP16 training.")
    AIMET_AVAILABLE = False


class DarkRegionAugmentation:
    """暗区数据增强，用于混合精度训练"""
    
    def __init__(self, dark_threshold=0.2, aug_prob=0.5, brightness_range=(0.3, 0.7)):
        self.dark_threshold = dark_threshold
        self.aug_prob = aug_prob
        self.brightness_range = brightness_range
    
    def __call__(self, lq, gt):
        """
        Args:
            lq: torch.Tensor, shape (C, H, W)
            gt: torch.Tensor, shape (C, H, W)
        """
        if random.random() < self.aug_prob:
            # 降低亮度，生成更多暗区样本
            brightness_factor = random.uniform(*self.brightness_range)
            lq = lq * brightness_factor
            gt = gt * brightness_factor
            
            # 添加暗区噪声
            dark_mask = (lq < self.dark_threshold).float()
            noise = torch.randn_like(lq) * 0.01
            lq = lq + noise * dark_mask
            
            # Clip
            lq = torch.clamp(lq, 0, 1)
            gt = torch.clamp(gt, 0, 1)
        
        return lq, gt


class BLCNoiseAugmentation:
    """模拟 BLC 误差和 FPN 噪声"""
    
    def __init__(self, blc_std=2.0, fpn_row_std=1.0, fpn_col_std=1.0, aug_prob=0.5):
        self.blc_std = blc_std
        self.fpn_row_std = fpn_row_std
        self.fpn_col_std = fpn_col_std
        self.aug_prob = aug_prob
    
    def __call__(self, img):
        """
        Args:
            img: torch.Tensor, shape (C, H, W), range [0, 1]
        """
        if random.random() > self.aug_prob:
            return img
        
        c, h, w = img.shape
        
        # 全局 BLC 误差
        blc_error = torch.randn(1) * self.blc_std / 255.0
        img = img + blc_error
        
        # 行噪声
        row_noise = torch.randn(1, h, 1) * self.fpn_row_std / 255.0
        row_noise = row_noise.expand(c, h, w)
        img = img + row_noise
        
        # 列噪声
        col_noise = torch.randn(1, 1, w) * self.fpn_col_std / 255.0
        col_noise = col_noise.expand(c, h, w)
        img = img + col_noise
        
        # Bayer 模式噪声 (2×2)
        if h >= 2 and w >= 2:
            bayer_noise = torch.randn(c, h // 2, w // 2) * (self.blc_std / 2) / 255.0
            bayer_noise = torch.nn.functional.interpolate(
                bayer_noise.unsqueeze(0), 
                size=(h, w), 
                mode='nearest'
            ).squeeze(0)
            img = img + bayer_noise
        
        # Clip
        img = torch.clamp(img, 0, 1)
        
        return img


def create_quantsim_model_fp16(model, dummy_input, config_file=None, 
                                default_output_bw=16, default_param_bw=16):
    """创建 AIMET FP16 QuantSim 模型"""
    
    if not AIMET_AVAILABLE:
        raise ImportError("AIMET is not installed. Cannot create QuantSim model.")
    
    # 创建 FP16 量化模拟器
    quantsim = QuantizationSimModel(
        model=model,
        quant_scheme=QuantScheme.post_training_tf_enhanced,
        dummy_input=dummy_input,
        default_output_bw=default_output_bw,  # 激活 16-bit
        default_param_bw=default_param_bw,    # 权重 16-bit
        config_file=config_file,
        in_place=False
    )
    
    return quantsim


def calibrate_model(quantsim, calibration_loader, num_batches=100, use_amp=True):
    """使用校准数据集计算量化参数"""
    
    logger = get_root_logger()
    logger.info(f"Starting FP16 calibration with {num_batches} batches...")
    
    quantsim.model.eval()
    
    def forward_pass(model, args):
        """前向传播回调函数"""
        device = next(model.parameters()).device
        with torch.no_grad():
            for i, data in enumerate(calibration_loader):
                if i >= num_batches:
                    break
                lq = data['lq'].to(device)
                
                if use_amp:
                    with autocast():
                        _ = model(lq)
                else:
                    _ = model(lq)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Calibration: {i+1}/{num_batches} batches processed")
    
    # 计算量化编码
    quantsim.compute_encodings(forward_pass, forward_pass_callback_args=None)
    
    logger.info("FP16 Calibration completed!")
    
    return quantsim


def parse_options(is_train=True):
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none',
                       help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    
    # AIMET FP16 相关参数
    parser.add_argument('--quantization_config', type=str, default=None,
                       help='Path to AIMET quantization config JSON file')
    parser.add_argument('--activation_bw', type=int, default=16,
                       help='Activation bitwidth (default: 16 for FP16)')
    parser.add_argument('--param_bw', type=int, default=16,
                       help='Parameter bitwidth (default: 16 for FP16)')
    parser.add_argument('--calibration_batches', type=int, default=100,
                       help='Number of batches for calibration')
    parser.add_argument('--pretrained_model', type=str, default=None,
                       help='Path to pretrained FP32 model')
    parser.add_argument('--use_native_amp', action='store_true',
                       help='Use PyTorch native AMP instead of AIMET')
    parser.add_argument('--dynamic_loss_scale', action='store_true',
                       help='Use dynamic loss scaling for AMP')
    
    args = parser.parse_args()
    opt = parse(args.opt, is_train=is_train)
    
    # 分布式设置
    if args.launcher == 'none':
        opt['dist'] = False
        print('Disable distributed.', flush=True)
    else:
        opt['dist'] = True
        if args.launcher == 'slurm' and 'dist_params' in opt:
            init_dist(args.launcher, **opt['dist_params'])
        else:
            init_dist(args.launcher)
    
    opt['rank'], opt['world_size'] = get_dist_info()
    
    # 随机种子
    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    set_random_seed(seed + opt['rank'])
    
    # 保存 FP16 参数
    opt['fp16'] = {
        'quantization_config': args.quantization_config,
        'activation_bw': args.activation_bw,
        'param_bw': args.param_bw,
        'calibration_batches': args.calibration_batches,
        'pretrained_model': args.pretrained_model,
        'use_native_amp': args.use_native_amp or not AIMET_AVAILABLE,
        'dynamic_loss_scale': args.dynamic_loss_scale
    }
    
    return opt


def init_loggers(opt):
    """初始化日志"""
    log_file = osp.join(opt['path']['log'],
                        f"train_fp16_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))
    
    # TensorBoard logger
    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        tb_logger = init_tb_logger(log_dir=osp.join('logs', opt['name']))
    
    return logger, tb_logger


def create_train_val_dataloader(opt, logger):
    """创建训练和验证数据加载器"""
    train_loader, val_loader = None, None
    
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = create_dataset(dataset_opt)
            train_sampler = EnlargedSampler(train_set, opt['world_size'],
                                          opt['rank'], dataset_enlarge_ratio)
            train_loader = create_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=train_sampler,
                seed=opt['manual_seed'])
            
            num_iter_per_epoch = math.ceil(
                len(train_set) * dataset_enlarge_ratio /
                (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
            total_iters = int(opt['train']['total_iter'])
            total_epochs = math.ceil(total_iters / num_iter_per_epoch)
            
            logger.info(
                'Training statistics:'
                f'\n\tNumber of train images: {len(train_set)}'
                f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                f'\n\tWorld size (gpu number): {opt["world_size"]}'
                f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')
        
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(
                val_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=None,
                seed=opt['manual_seed'])
            logger.info(
                f'Number of val images/folders in {dataset_opt["name"]}: {len(val_set)}')
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')
    
    return train_loader, train_sampler, val_loader, total_epochs, total_iters


def main():
    """AIMET FP16 混合精度训练主函数"""
    
    # 解析参数
    opt = parse_options(is_train=True)
    
    torch.backends.cudnn.benchmark = True
    
    # 创建实验目录
    make_exp_dirs(opt)
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name'] and opt['rank'] == 0:
        mkdir_and_rename(osp.join('tb_logger', opt['name']))
    
    # 初始化日志
    logger, tb_logger = init_loggers(opt)
    
    # 创建数据加载器
    result = create_train_val_dataloader(opt, logger)
    train_loader, train_sampler, val_loader, total_epochs, total_iters = result
    
    # 创建 FP32 模型
    logger.info("Creating base model...")
    model = create_model(opt)
    
    # 加载预训练模型（如果提供）
    if opt['fp16']['pretrained_model'] is not None:
        logger.info(f"Loading pretrained model from {opt['fp16']['pretrained_model']}")
        checkpoint = torch.load(opt['fp16']['pretrained_model'], map_location='cpu')
        if 'params' in checkpoint:
            model.net_g.load_state_dict(checkpoint['params'], strict=True)
        else:
            model.net_g.load_state_dict(checkpoint, strict=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.net_g.to(device)
    
    # 选择 FP16 训练方式
    use_native_amp = opt['fp16']['use_native_amp']
    
    if use_native_amp:
        # 使用 PyTorch 原生 AMP
        logger.info("Using PyTorch native AMP for FP16 training")
        logger.info(f"Mixed precision: FP16/FP32 (native AMP)")
        scaler = GradScaler(enabled=True, 
                           init_scale=2.**16 if not opt['fp16']['dynamic_loss_scale'] else 2.**10,
                           growth_interval=2000 if opt['fp16']['dynamic_loss_scale'] else 1000)
        quantsim = None
    else:
        # 使用 AIMET QuantSim
        logger.info("Creating AIMET QuantSim model for FP16...")
        logger.info(f"Quantization config: a{opt['fp16']['activation_bw']}w{opt['fp16']['param_bw']}")
        
        # 创建 dummy input
        dummy_input = torch.randn(1, 3, 256, 256).to(device)
        
        quantsim = create_quantsim_model_fp16(
            model=model.net_g,
            dummy_input=dummy_input,
            config_file=opt['fp16']['quantization_config'],
            default_output_bw=opt['fp16']['activation_bw'],
            default_param_bw=opt['fp16']['param_bw']
        )
        
        # 校准
        logger.info("Starting calibration...")
        quantsim = calibrate_model(
            quantsim, 
            train_loader, 
            num_batches=opt['fp16']['calibration_batches'],
            use_amp=True
        )
        
        # 替换模型的网络为量化模型
        model.net_g = quantsim.model
        scaler = GradScaler(enabled=True)
    
    # 初始化数据增强
    dark_aug = DarkRegionAugmentation(aug_prob=0.5)
    blc_aug = BLCNoiseAugmentation(aug_prob=0.3)
    
    # 消息日志
    msg_logger = MessageLogger(opt, 0, tb_logger)
    
    # 数据预取器
    prefetch_mode = opt['datasets']['train'].get('prefetch_mode', 'cpu')
    if prefetch_mode == 'cpu':
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda':
        prefetcher = CUDAPrefetcher(train_loader, opt)
    else:
        raise ValueError(f'Wrong prefetch_mode {prefetch_mode}')
    
    # FP16 训练循环
    logger.info(f'Start FP16 mixed precision training from epoch: 0, iter: 0')
    data_time, iter_time = time.time(), time.time()
    start_time = time.time()
    
    current_iter = 0
    epoch = 0
    
    while current_iter <= total_iters:
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()
        
        while train_data is not None:
            data_time = time.time() - data_time
            current_iter += 1
            
            if current_iter > total_iters:
                break
            
            # 数据增强
            lq, gt = train_data['lq'], train_data['gt']
            
            # 暗区增强
            if random.random() < 0.5:
                batch_size = lq.size(0)
                for i in range(batch_size):
                    lq[i], gt[i] = dark_aug(lq[i], gt[i])
            
            # BLC 噪声增强
            if random.random() < 0.3:
                lq = torch.stack([blc_aug(lq[i]) for i in range(lq.size(0))])
            
            train_data['lq'] = lq
            train_data['gt'] = gt
            
            # 更新学习率
            model.update_learning_rate(
                current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
            
            # FP16 训练
            model.feed_data(train_data, is_val=False)
            
            # 使用 AMP 前向和反向传播
            model.optimizer_g.zero_grad()
            
            with autocast():
                model.net_g.train()
                output = model.net_g(model.lq)
                l_total = 0
                loss_dict = {}
                
                # 计算损失
                if hasattr(model, 'cri_pix') and model.cri_pix:
                    l_pix = model.cri_pix(output, model.gt)
                    l_total += l_pix
                    loss_dict['l_pix'] = l_pix
            
            # Backward with gradient scaling
            scaler.scale(l_total).backward()
            scaler.step(model.optimizer_g)
            scaler.update()
            
            # 更新日志
            model.log_dict = loss_dict
            
            iter_time = time.time() - iter_time
            
            # 日志
            if current_iter % opt['logger']['print_freq'] == 0:
                log_vars = {'epoch': epoch, 'iter': current_iter, 'total_iter': total_iters}
                log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update({'time': iter_time, 'data_time': data_time})
                log_vars.update(model.get_current_log())
                log_vars.update({'loss_scale': scaler.get_scale()})
                msg_logger(log_vars)
            
            # 保存模型
            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving FP16 models and training states.')
                model.save(epoch, current_iter)
                
                # 导出量化编码（如果使用 AIMET）
                if quantsim is not None:
                    encoding_path = osp.join(opt['path']['models'], 
                                            f'quantsim_encodings_fp16_{current_iter}.json')
                    quantsim.export_encodings(encoding_path)
                    logger.info(f'FP16 quantization encodings saved to {encoding_path}')
            
            # 验证
            if opt.get('val') is not None and current_iter % opt['val']['val_freq'] == 0:
                model.net_g.eval()
                with torch.no_grad():
                    with autocast():
                        rgb2bgr = opt['val'].get('rgb2bgr', True)
                        use_image = opt['val'].get('use_image', True)
                        model.validation(val_loader, current_iter, tb_logger,
                                       opt['val']['save_img'], rgb2bgr, use_image)
                model.net_g.train()
                
                log_vars = {'epoch': epoch, 'iter': current_iter, 'total_iter': total_iters}
                log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)
            
            data_time = time.time()
            iter_time = time.time()
            train_data = prefetcher.next()
        
        epoch += 1
    
    # 训练结束
    consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of FP16 training. Time consumed: {consumed_time}')
    logger.info('Save the latest FP16 model.')
    model.save(epoch=-1, current_iter=-1)
    
    # 导出最终量化编码
    if quantsim is not None:
        final_encoding_path = osp.join(opt['path']['models'], 'quantsim_encodings_fp16_final.json')
        quantsim.export_encodings(final_encoding_path)
        logger.info(f'Final FP16 quantization encodings saved to {final_encoding_path}')
    
    # 最终验证
    if opt.get('val') is not None:
        model.net_g.eval()
        with torch.no_grad():
            with autocast():
                rgb2bgr = opt['val'].get('rgb2bgr', True)
                use_image = opt['val'].get('use_image', True)
                model.validation(val_loader, current_iter, tb_logger,
                                opt['val']['save_img'], rgb2bgr, use_image)
    
    if tb_logger:
        tb_logger.close()


if __name__ == '__main__':
    import os
    os.environ['GRPC_POLL_STRATEGY'] = 'epoll1'
    main()

