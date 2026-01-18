# ------------------------------------------------------------------------
# AIMET Quantization Training for NAFNet (A8W8)
# 8-bit activation, 8-bit weight quantization (Standard INT8)
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

# AIMET imports
try:
    from aimet_torch.quantsim import QuantizationSimModel
    from aimet_torch.qc_quantize_op import QcQuantizeWrapper
    from aimet_common.defs import QuantScheme
    AIMET_AVAILABLE = True
except ImportError:
    print("Warning: AIMET not installed. Please install aimet_torch.")
    AIMET_AVAILABLE = False


class AggressiveDarkRegionAugmentation:
    """针对 A8W8 的激进暗区数据增强"""
    
    def __init__(self, dark_threshold=0.25, aug_prob=0.6, brightness_range=(0.2, 0.8)):
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
            noise = torch.randn_like(lq) * 0.015  # 更强的噪声
            lq = lq + noise * dark_mask
            
            # 添加随机 dropout（模拟量化丢失）
            if random.random() < 0.3:
                dropout_mask = torch.rand_like(lq) > 0.05
                lq = lq * dropout_mask.float()
            
            # Clip
            lq = torch.clamp(lq, 0, 1)
            gt = torch.clamp(gt, 0, 1)
        
        return lq, gt


class EnhancedBLCNoiseAugmentation:
    """增强版 BLC 噪声增强，针对 A8W8 优化"""
    
    def __init__(self, blc_std=3.0, fpn_row_std=2.0, fpn_col_std=2.0, aug_prob=0.6):
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
        
        # 添加量化噪声模拟
        if random.random() < 0.4:
            # 模拟 8-bit 量化效果
            img = torch.round(img * 255) / 255
        
        # Clip
        img = torch.clamp(img, 0, 1)
        
        return img


class QuantizationAwareAugmentation:
    """量化感知数据增强，专门针对 A8W8"""
    
    def __init__(self, aug_prob=0.5):
        self.aug_prob = aug_prob
    
    def __call__(self, lq, gt):
        """
        Args:
            lq: torch.Tensor, shape (B, C, H, W) or (C, H, W)
            gt: torch.Tensor, shape (B, C, H, W) or (C, H, W)
        """
        if random.random() > self.aug_prob:
            return lq, gt
        
        # 随机选择增强类型
        aug_type = random.choice(['brightness', 'contrast', 'gamma', 'quantize'])
        
        if aug_type == 'brightness':
            # 亮度调整（更激进）
            factor = random.uniform(0.4, 1.6)
            lq = lq * factor
            gt = gt * factor
        
        elif aug_type == 'contrast':
            # 对比度调整
            factor = random.uniform(0.6, 1.4)
            mean = lq.mean()
            lq = (lq - mean) * factor + mean
            mean = gt.mean()
            gt = (gt - mean) * factor + mean
        
        elif aug_type == 'gamma':
            # Gamma 校正
            gamma = random.uniform(0.7, 1.3)
            lq = torch.pow(torch.clamp(lq, 1e-7, 1.0), gamma)
            gt = torch.pow(torch.clamp(gt, 1e-7, 1.0), gamma)
        
        elif aug_type == 'quantize':
            # 模拟量化效果
            bits = random.choice([6, 7, 8])
            levels = 2 ** bits
            lq = torch.round(lq * (levels - 1)) / (levels - 1)
            # GT 保持高精度
        
        # Clip
        lq = torch.clamp(lq, 0, 1)
        gt = torch.clamp(gt, 0, 1)
        
        return lq, gt


class WeightQuantizationAugmentation:
    """权重量化增强，针对 8-bit 权重"""
    
    def __init__(self, noise_std=0.002, aug_prob=0.4):
        self.noise_std = noise_std
        self.aug_prob = aug_prob
    
    def apply_to_model(self, model):
        """对模型权重添加微小扰动，增强量化鲁棒性"""
        if random.random() < self.aug_prob:
            with torch.no_grad():
                for param in model.parameters():
                    if param.requires_grad:
                        noise = torch.randn_like(param) * self.noise_std
                        param.add_(noise)


def create_quantsim_model(model, dummy_input, config_file=None, 
                          default_output_bw=8, default_param_bw=8):
    """创建 AIMET QuantSim 模型 (A8W8)"""
    
    if not AIMET_AVAILABLE:
        raise ImportError("AIMET is not installed. Cannot create QuantSim model.")
    
    # 创建量化模拟器
    quantsim = QuantizationSimModel(
        model=model,
        quant_scheme=QuantScheme.post_training_tf_enhanced,
        dummy_input=dummy_input,
        default_output_bw=default_output_bw,  # 激活 8-bit
        default_param_bw=default_param_bw,    # 权重 8-bit
        config_file=config_file,
        in_place=False
    )
    
    return quantsim


def calibrate_model(quantsim, calibration_loader, num_batches=200):
    """使用校准数据集计算量化参数（A8W8 需要更多校准数据）"""
    
    logger = get_root_logger()
    logger.info(f"Starting A8W8 calibration with {num_batches} batches...")
    
    quantsim.model.eval()
    
    def forward_pass(model, args):
        """前向传播回调函数"""
        device = next(model.parameters()).device
        with torch.no_grad():
            for i, data in enumerate(calibration_loader):
                if i >= num_batches:
                    break
                lq = data['lq'].to(device)
                _ = model(lq)
                if (i + 1) % 20 == 0:
                    logger.info(f"Calibration: {i+1}/{num_batches} batches processed")
    
    # 计算量化编码
    quantsim.compute_encodings(forward_pass, forward_pass_callback_args=None)
    
    logger.info("A8W8 Calibration completed!")
    
    return quantsim


def parse_options(is_train=True):
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none',
                       help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    
    # AIMET 相关参数 (A8W8 默认值)
    parser.add_argument('--quantization_config', type=str, default=None,
                       help='Path to AIMET quantization config JSON file')
    parser.add_argument('--activation_bw', type=int, default=8,
                       help='Activation bitwidth (default: 8)')
    parser.add_argument('--param_bw', type=int, default=8,
                       help='Parameter bitwidth (default: 8)')
    parser.add_argument('--calibration_batches', type=int, default=200,
                       help='Number of batches for calibration (A8W8 needs more)')
    parser.add_argument('--pretrained_model', type=str, default=None,
                       help='Path to pretrained FP32 model for QAT')
    
    # A8W8 特定参数
    parser.add_argument('--aggressive_aug', action='store_true', default=True,
                       help='Enable aggressive augmentation for A8W8')
    parser.add_argument('--weight_aug', action='store_true', default=True,
                       help='Enable weight augmentation for 8-bit weight quantization')
    parser.add_argument('--quant_aware_aug', action='store_true', default=True,
                       help='Enable quantization-aware augmentation')
    
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
    
    # 保存 AIMET 参数
    opt['aimet'] = {
        'quantization_config': args.quantization_config,
        'activation_bw': args.activation_bw,
        'param_bw': args.param_bw,
        'calibration_batches': args.calibration_batches,
        'pretrained_model': args.pretrained_model,
        'aggressive_aug': args.aggressive_aug,
        'weight_aug': args.weight_aug,
        'quant_aware_aug': args.quant_aware_aug
    }
    
    return opt


def init_loggers(opt):
    """初始化日志"""
    log_file = osp.join(opt['path']['log'],
                        f"train_aimet_a8w8_{opt['name']}_{get_time_str()}.log")
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
    """AIMET A8W8 量化训练主函数"""
    
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
    logger.info("Creating FP32 base model...")
    model = create_model(opt)
    
    # 加载预训练模型（如果提供）
    if opt['aimet']['pretrained_model'] is not None:
        logger.info(f"Loading pretrained model from {opt['aimet']['pretrained_model']}")
        checkpoint = torch.load(opt['aimet']['pretrained_model'], map_location='cpu')
        if 'params' in checkpoint:
            model.net_g.load_state_dict(checkpoint['params'], strict=True)
        else:
            model.net_g.load_state_dict(checkpoint, strict=True)
    
    # 创建 AIMET QuantSim 模型 (A8W8)
    logger.info("Creating AIMET QuantSim model for A8W8 (Standard INT8)...")
    logger.info(f"Quantization config: a{opt['aimet']['activation_bw']}w{opt['aimet']['param_bw']}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.net_g.to(device)
    
    # 创建 dummy input
    dummy_input = torch.randn(1, 3, 256, 256).to(device)
    
    quantsim = create_quantsim_model(
        model=model.net_g,
        dummy_input=dummy_input,
        config_file=opt['aimet']['quantization_config'],
        default_output_bw=opt['aimet']['activation_bw'],
        default_param_bw=opt['aimet']['param_bw']
    )
    
    # 校准（A8W8 需要更多校准数据）
    logger.info("Starting A8W8 calibration...")
    quantsim = calibrate_model(
        quantsim, 
        train_loader, 
        num_batches=opt['aimet']['calibration_batches']
    )
    
    # 替换模型的网络为量化模型
    model.net_g = quantsim.model
    
    # 初始化数据增强
    dark_aug = AggressiveDarkRegionAugmentation(aug_prob=0.6) if opt['aimet']['aggressive_aug'] else None
    blc_aug = EnhancedBLCNoiseAugmentation(aug_prob=0.6)
    quant_aug = QuantizationAwareAugmentation(aug_prob=0.5) if opt['aimet']['quant_aware_aug'] else None
    weight_aug = WeightQuantizationAugmentation(aug_prob=0.4) if opt['aimet']['weight_aug'] else None
    
    logger.info(f"Data augmentation enabled:")
    logger.info(f"  - Dark Region (Aggressive): {opt['aimet']['aggressive_aug']}")
    logger.info(f"  - BLC Noise (Enhanced): Yes")
    logger.info(f"  - Quantization-Aware: {opt['aimet']['quant_aware_aug']}")
    logger.info(f"  - Weight Augmentation: {opt['aimet']['weight_aug']}")
    
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
    
    # QAT 训练循环
    logger.info(f'Start A8W8 QAT training from epoch: 0, iter: 0')
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
            
            # 暗区增强（激进）
            if dark_aug is not None and random.random() < 0.6:
                lq, gt = dark_aug(lq, gt)
            
            # BLC 噪声增强
            if random.random() < 0.6:
                lq = torch.stack([blc_aug(lq[i]) for i in range(lq.size(0))])
            
            # 量化感知增强
            if quant_aug is not None and random.random() < 0.5:
                lq, gt = quant_aug(lq, gt)
            
            train_data['lq'] = lq
            train_data['gt'] = gt
            
            # 权重增强（每 3 个 iter）
            if weight_aug is not None and current_iter % 3 == 0:
                weight_aug.apply_to_model(model.net_g)
            
            # 更新学习率
            model.update_learning_rate(
                current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
            
            # 训练
            model.feed_data(train_data, is_val=False)
            model.optimize_parameters(current_iter, tb_logger)
            
            iter_time = time.time() - iter_time
            
            # 日志
            if current_iter % opt['logger']['print_freq'] == 0:
                log_vars = {'epoch': epoch, 'iter': current_iter, 'total_iter': total_iters}
                log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update({'time': iter_time, 'data_time': data_time})
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)
            
            # 保存模型
            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving A8W8 quantized models and training states.')
                model.save(epoch, current_iter)
                
                # 导出量化编码
                encoding_path = osp.join(opt['path']['models'], 
                                        f'quantsim_a8w8_encodings_{current_iter}.json')
                quantsim.export_encodings(encoding_path)
                logger.info(f'A8W8 quantization encodings saved to {encoding_path}')
            
            # 验证
            if opt.get('val') is not None and current_iter % opt['val']['val_freq'] == 0:
                rgb2bgr = opt['val'].get('rgb2bgr', True)
                use_image = opt['val'].get('use_image', True)
                model.validation(val_loader, current_iter, tb_logger,
                               opt['val']['save_img'], rgb2bgr, use_image)
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
    logger.info(f'End of A8W8 QAT training. Time consumed: {consumed_time}')
    logger.info('Save the latest A8W8 quantized model.')
    model.save(epoch=-1, current_iter=-1)
    
    # 导出最终量化编码
    final_encoding_path = osp.join(opt['path']['models'], 'quantsim_a8w8_encodings_final.json')
    quantsim.export_encodings(final_encoding_path)
    logger.info(f'Final A8W8 quantization encodings saved to {final_encoding_path}')
    
    # 最终验证
    if opt.get('val') is not None:
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

