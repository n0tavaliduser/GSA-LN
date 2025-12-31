import os
import sys
import math
import cv2
import torch
import numpy as np
import yaml
import argparse
from collections import OrderedDict
from PIL import Image, ImageDraw, ImageFont

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from basicsr.archs.gsaln_arch import GSALN
from basicsr.archs.swinir_arch import SwinIR
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.archs.edsr_arch import EDSR
from basicsr.archs.rcan_arch import RCAN
from basicsr.archs.vdsr_arch import VDSR
from basicsr.archs.srcnn_arch import SRCNN
# Some basicsr versions differ in tensor2img import, try utils module
try:
    from basicsr.utils import tensor2img
except ImportError:
    from basicsr.utils.img_util import tensor2img

from basicsr.metrics import calculate_psnr, calculate_ssim
from basicsr.utils import img2tensor

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def load_image(path):
    # Read image using cv2
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not load image from {path}")
    img = img.astype(np.float32) / 255.
    return img

def inference_model(model, img_lr, device, window_size=None, scale=2):
    """
    Run inference on a model with optional padding for window-based models.
    
    Args:
        model: The SR model
        img_lr: LR image as numpy array (H, W, C) in float32 [0, 1]
        device: torch device
        window_size: If provided, pad input to be divisible by this value
        scale: Upscale factor for cropping output after padding
    """
    h, w, _ = img_lr.shape
    
    # Calculate padding if window_size is specified
    if window_size is not None:
        pad_h = (window_size - h % window_size) % window_size
        pad_w = (window_size - w % window_size) % window_size
        
        if pad_h > 0 or pad_w > 0:
            # Pad with reflection
            img_lr_padded = cv2.copyMakeBorder(img_lr, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        else:
            img_lr_padded = img_lr
    else:
        img_lr_padded = img_lr
        pad_h, pad_w = 0, 0
    
    img_lr_tensor = img2tensor(img_lr_padded, bgr2rgb=True, float32=True).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_lr_tensor)
    
    result = tensor2img(output, rgb2bgr=True, min_max=(0, 1))
    
    # Crop padding from output (scaled by upscale factor)
    if pad_h > 0 or pad_w > 0:
        out_h = h * scale
        out_w = w * scale
        result = result[:out_h, :out_w, :]
        
    return result


def get_metrics(img_sr, img_hr, crop_border, test_y_channel=True):
    # img_sr and img_hr should be uint8 images (0-255) in BGR order
    if img_sr.shape != img_hr.shape:
        # Resize SR to match HR for metric calculation if valid
        # But usually they should match. If not, crop or resize.
        h, w = img_hr.shape[:2]
        img_sr = cv2.resize(img_sr, (w, h))

    psnr = calculate_psnr(img_sr, img_hr, crop_border=crop_border, input_order='HWC', test_y_channel=test_y_channel)
    ssim = calculate_ssim(img_sr, img_hr, crop_border=crop_border, input_order='HWC', test_y_channel=test_y_channel)
    return psnr, ssim

def draw_text_pil(img_cv, text, position, font_path='arial.ttf', font_size=20, color=(0, 0, 0)):
    # Convert CV2 BGR to PIL RGB
    img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        # Fallback to default if arial not found
        # On windows usually works, on linux might need path
        try:
             font = ImageFont.truetype("Arial.ttf", font_size)
        except:
             font = ImageFont.load_default()
             print("Warning: Arial font not found, using default.")

    draw.text(position, text, font=font, fill=color)
    
    # Convert PIL RGB back to CV2 BGR
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# Helper to calculate text size using PIL
def get_text_size_pil(text, font_path='arial.ttf', font_size=20):
    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        try:
             font = ImageFont.truetype("Arial.ttf", font_size)
        except:
             font = ImageFont.load_default()
    
    # getbbox returns (left, top, right, bottom)
    bbox = font.getbbox(text)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return width, height

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='options/test/test_gsaln_x2_tta_x8.yml', help='Path to our model config')
    parser.add_argument('--output_dir', type=str, default='results/comparison', help='Output directory for results')
    args = parser.parse_args()

    # ==================== CONFIGURATION ====================
    # Benchmark datasets to process
    BENCHMARK_DATASETS = [
        'Set5', 
        'Set14', 
        'Urban100', 
        'B100', 
        'Manga109'
    ]
    BENCHMARK_BASE_PATH = 'datasets/benchmark'
    # ========================================================

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Config and Our Model
    print(f"Loading configuration from {args.config}...")
    if not os.path.exists(args.config):
        print(f"Error: Config file not found at {args.config}")
        return

    opt = load_yaml(args.config)
    scale = opt.get('scale', 2) # Default to 2 if not found
    
    # Load Main Model (Our Model - GSALN)
    print("Loading Our Model (GSALN)...")
    
    # Construct model 
    if 'network_g' in opt:
        # Remove 'type' key if present as it's not an arg for GSALN class
        net_opt = opt['network_g'].copy()
        if 'type' in net_opt:
            net_opt.pop('type')
        model_g = GSALN(**net_opt)
    else:
        model_g = GSALN(num_in_ch=3, num_out_ch=3, num_feat=64, num_blocks=32, upscale=scale)
        
    model_path = opt.get('path', {}).get('pretrain_network_g')
    
    # Resolve relative path for model (relative to project root)
    if model_path and not os.path.isabs(model_path):
         model_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), model_path)
    
    if model_path and os.path.exists(model_path):
        print(f"Loading weights from {model_path}")
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
        if 'params_ema' in state_dict:
            state_dict = state_dict['params_ema']
        elif 'params' in state_dict:
            state_dict = state_dict['params']
            
        model_g.load_state_dict(state_dict, strict=True)
    else:
        print(f"Warning: Model path {model_path} not found. Running with initialized weights (random results).")
    
    model_g.eval()
    model_g.to(device)
    # Other models configuration - uses 'scale' variable from config
    other_models_config = [
        # {'name': 'ESRGAN', 'path': f'models/ESRGAN_x{scale}.pth', 'arch': RRDBNet, 'args': {'num_in_ch': 3, 'num_out_ch': 3, 'scale': scale, 'num_feat': 64, 'num_block': 23}},
        # {'name': 'SwinIR', 'path': f'models/SwinIR_x{scale}.pth', 'arch': SwinIR, 'window_size': 8, 'args': {'upscale': scale, 'in_chans': 3, 'img_size': 64, 'window_size': 8, 'img_range': 1., 'depths': [6, 6, 6, 6, 6, 6], 'embed_dim': 180, 'num_heads': [6, 6, 6, 6, 6, 6], 'mlp_ratio': 2, 'upsampler': 'pixelshuffle', 'resi_connection': '1conv'}},
        {'name': 'EDSR', 'path': f'models/EDSR_x{scale}.pth', 'arch': EDSR, 'args': {'num_in_ch': 3, 'num_out_ch': 3, 'num_feat': 64, 'num_block': 16, 'upscale': scale, 'res_scale': 1.0, 'img_range': 255., 'rgb_mean': (0.4488, 0.4371, 0.4040)}},
        # {'name': 'RCAN', 'path': f'models/RCAN_x{scale}.pth', 'arch': RCAN, 'args': {'num_in_ch': 3, 'num_out_ch': 3, 'num_feat': 64, 'num_group': 10, 'num_block': 20, 'squeeze_factor': 16, 'upscale': scale, 'res_scale': 1.0, 'img_range': 255., 'rgb_mean': (0.4488, 0.4371, 0.4040)}},
        {'name': 'SRCNN', 'path': f'models/SRCNN_x{scale}.pth', 'arch': SRCNN, 'uses_bicubic_input': True, 'y_channel_only': True, 'args': {'num_in_ch': 1, 'num_out_ch': 1, 'num_feat': 64, 'num_feat2': 32}},
        {'name': 'VDSR', 'path': f'models/VDSR_x{scale}.pth', 'arch': VDSR, 'uses_bicubic_input': True, 'args': {'num_in_ch': 3, 'num_out_ch': 3, 'num_feat': 64, 'num_block': 18}},
    ]
    
    # Resolve benchmark base path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    benchmark_base = os.path.join(project_root, BENCHMARK_BASE_PATH)
    
    # Collect image pairs (HR, LR) from benchmark datasets
    img_pairs = []  # List of (hr_path, lr_path, dataset_name)
    for dataset_name in BENCHMARK_DATASETS:
        hr_dir = os.path.join(benchmark_base, dataset_name, 'HR')
        lr_dir = os.path.join(benchmark_base, dataset_name, 'LR_bicubic', f'X{scale}')
        
        if not os.path.exists(hr_dir):
            print(f"Warning: HR directory not found: {hr_dir}")
            continue
        if not os.path.exists(lr_dir):
            print(f"Warning: LR directory not found: {lr_dir}")
            continue
        
        print(f"Scanning {dataset_name}...")
        for file in os.listdir(hr_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                hr_path = os.path.join(hr_dir, file)
                base_name = os.path.splitext(file)[0]
                ext = os.path.splitext(file)[1]
                
                # Different datasets use different LR filename patterns
                if dataset_name == 'Manga109':
                    # Manga109 uses: {base_name}_LRBI_x{scale}.png
                    lr_filename = f"{base_name}_LRBI_x{scale}.png"
                else:
                    # Default pattern: {base_name}x{scale}{ext}
                    lr_filename = f"{base_name}x{scale}{ext}"
                
                lr_path = os.path.join(lr_dir, lr_filename)
                
                if os.path.exists(lr_path):
                    img_pairs.append((hr_path, lr_path, dataset_name))
                else:
                    print(f"Warning: LR image not found for {file}: {lr_path}")

    if not img_pairs:
        print("No image pairs found to process.")
        return

    print(f"Found {len(img_pairs)} image pairs from {len(BENCHMARK_DATASETS)} datasets.")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # 3. Process Images Loop
    for idx, (hr_path, lr_path, dataset_name) in enumerate(img_pairs):
        print(f"[{idx+1}/{len(img_pairs)}] Processing {dataset_name}/{os.path.basename(hr_path)}...")
        
        try:
            # Load HR and LR images directly from benchmark
            img_hr = load_image(hr_path)
            img_lr = load_image(lr_path)
            
            h_hr, w_hr, _ = img_hr.shape
            h_lr, w_lr, _ = img_lr.shape
            
            # HR dimensions should match LR * scale
            h_new, w_new = h_lr * scale, w_lr * scale
            
            # Crop HR if needed to match expected dimensions
            if h_hr != h_new or w_hr != w_new:
                img_hr = img_hr[:h_new, :w_new, :]

            # Generate Bicubic Upscale (Baseline) from LR
            img_bicubic = cv2.resize(img_lr, (w_new, h_new), interpolation=cv2.INTER_CUBIC)
            img_bicubic = np.clip(img_bicubic, 0, 1)  # Clip to prevent color artifacts from overshoot

            # 4. Inference Our Model
            res_our = inference_model(model_g, img_lr, device, scale=scale)

            # 5. Inference Other Models
            results_others = []
            for m_conf in other_models_config:
                m_name = m_conf['name']
                m_path = m_conf['path']
                
                # Resolve path relative to project root if not absolute
                if not os.path.isabs(m_path):
                     m_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), m_path)
                
                if not os.path.exists(m_path):
                    print(f"Skipping {m_name}: Model file not found at {m_path}")
                    continue
                try:
                    model_other = m_conf['arch'](**m_conf.get('args', {}))
                    state_dict = torch.load(m_path, map_location=lambda storage, loc: storage)
                    if 'params_ema' in state_dict:
                        state_dict = state_dict['params_ema']
                    elif 'params' in state_dict:
                        state_dict = state_dict['params']
                    
                    # Clean "module." prefix
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k.replace('module.', '')
                        new_state_dict[name] = v
                    state_dict = new_state_dict

                    # Specific handling for RCAN mismatch (original repo vs basicsr)
                    if m_name == 'RCAN':
                         if 'head.0.weight' in state_dict:
                            mapped_dict = OrderedDict()
                            n_groups = m_conf['args'].get('num_group', 10)
                            n_blocks = m_conf['args'].get('num_block', 20)
                            
                            for k, v in state_dict.items():
                                if k.startswith('head.0.'):
                                    # head.0 -> conv_first
                                    mapped_dict[k.replace('head.0.', 'conv_first.')] = v
                                elif k.startswith('body.') and k.split('.')[1].isdigit():
                                    parts = k.split('.')
                                    group_idx = int(parts[1])
                                    
                                    if group_idx < n_groups:
                                        # body.G.body.B.body.X -> body.G.residual_group.B.rcab.X
                                        # body.G.body.B.body.3.conv_du.X -> body.G.residual_group.B.rcab.3.attention.X
                                        # body.G.body.20 -> body.G.conv
                                        if len(parts) >= 4 and parts[2] == 'body':
                                            block_idx = int(parts[3]) if parts[3].isdigit() else -1
                                            if block_idx == n_blocks:
                                                # body.G.body.20 -> body.G.conv
                                                new_k = k.replace(f'body.{group_idx}.body.{block_idx}.', f'body.{group_idx}.conv.')
                                                mapped_dict[new_k] = v
                                            elif block_idx >= 0 and block_idx < n_blocks:
                                                # body.G.body.B.body.X -> body.G.residual_group.B.rcab.X
                                                new_k = k.replace(f'.body.{block_idx}.body.', f'.residual_group.{block_idx}.rcab.')
                                                # conv_du -> attention with index shift (0->1, 2->3)
                                                # Original: conv_du has [Conv, ReLU, Conv, Sigmoid] at indices 0,1,2,3
                                                # BasicSR: attention has [AdaptiveAvgPool, Conv, ReLU, Conv, Sigmoid] at indices 0,1,2,3,4
                                                new_k = new_k.replace('.conv_du.0.', '.attention.1.')
                                                new_k = new_k.replace('.conv_du.2.', '.attention.3.')
                                                mapped_dict[new_k] = v
                                            else:
                                                mapped_dict[k] = v
                                        else:
                                            mapped_dict[k] = v
                                    elif group_idx == n_groups:
                                        # body.10 -> conv_after_body
                                        new_k = k.replace(f'body.{group_idx}.', 'conv_after_body.')
                                        mapped_dict[new_k] = v
                                    else:
                                        mapped_dict[k] = v
                                elif k.startswith('tail.'):
                                    # tail.0.0 -> upsample.0, tail.1 -> conv_last
                                    if 'tail.0.0.' in k:
                                        mapped_dict[k.replace('tail.0.0.', 'upsample.0.')] = v
                                    elif 'tail.1.' in k:
                                        mapped_dict[k.replace('tail.1.', 'conv_last.')] = v
                                    else:
                                        mapped_dict[k] = v
                                elif k.startswith(('sub_mean', 'add_mean')):
                                    pass  # Ignore mean shift layers
                                else:
                                    mapped_dict[k] = v
                            
                            state_dict = mapped_dict

                    # Specific handling for EDSR mismatch (Head/Body/Tail vs conv_first/body/etc)
                    if m_name == 'EDSR':
                         n_blocks = m_conf['args'].get('num_block', 16)
                         
                         # Attempt to map keys if structure mismatches
                         if 'head.0.weight' in state_dict:
                            mapped_dict = OrderedDict()
                            for k, v in state_dict.items():
                                if k.startswith('head.0.'):
                                    mapped_dict[k.replace('head.0.', 'conv_first.')] = v
                                elif k.startswith('body.'):
                                    # Check if this is the last body element (conv_after_body)
                                    # Expected format: body.16.weight (if 16 blocks)
                                    # or body.32.weight etc.
                                    
                                    # Extract block index
                                    parts = k.split('.')
                                    try:
                                        idx = int(parts[1])
                                    except ValueError:
                                        mapped_dict[k] = v
                                        continue
                                        
                                    if idx == n_blocks:
                                        # This is conv_after_body
                                        new_k = k.replace(f'body.{idx}.', 'conv_after_body.')
                                        mapped_dict[new_k] = v
                                    elif idx < n_blocks:
                                        # Normal body block mapping
                                        # body.X.body.0 -> body.X.conv1
                                        # body.X.body.2 -> body.X.conv2
                                        new_k = k
                                        new_k = new_k.replace('.body.0.', '.conv1.')
                                        new_k = new_k.replace('.body.2.', '.conv2.')
                                        mapped_dict[new_k] = v
                                    else:
                                        # Index > n_blocks? Should not happen if config matches model
                                        mapped_dict[k] = v
                                        
                                elif k.startswith('tail.'):
                                    # tail.0.0 -> upsample.0
                                    # tail.0.2 -> upsample.2 (for x4 scale)
                                    # tail.1 -> conv_last
                                    if 'tail.0.0.' in k:
                                        mapped_dict[k.replace('tail.0.0.', 'upsample.0.')] = v
                                    elif 'tail.0.2.' in k:
                                        mapped_dict[k.replace('tail.0.2.', 'upsample.2.')] = v
                                    elif 'tail.1.' in k:
                                        mapped_dict[k.replace('tail.1.', 'conv_last.')] = v
                                    else:
                                         mapped_dict[k] = v
                                elif k.startswith(('sub_mean', 'add_mean')):
                                    pass # Ignore mean shift layers as they are functional in basicsr impl
                                else:
                                    mapped_dict[k] = v
                            
                            state_dict = mapped_dict

                    # Specific handling for VDSR mismatch (original naming vs our architecture)
                    # Original checkpoint: conv_1, conv_2_to_19.conv_X (X=2-19), conv_20
                    # Our architecture: conv_first, body.0/2/4/.../34, conv_last
                    if m_name == 'VDSR':
                        if 'conv_1.weight' in state_dict:
                            mapped_dict = OrderedDict()
                            for k, v in state_dict.items():
                                if k.startswith('conv_1.'):
                                    # conv_1 -> conv_first
                                    mapped_dict[k.replace('conv_1.', 'conv_first.')] = v
                                elif k.startswith('conv_2_to_19.'):
                                    # conv_2_to_19.conv_X -> body.((X-2)*2)
                                    # e.g., conv_2_to_19.conv_2 -> body.0
                                    #       conv_2_to_19.conv_3 -> body.2
                                    #       conv_2_to_19.conv_19 -> body.34
                                    import re
                                    match = re.match(r'conv_2_to_19\.conv_(\d+)\.(.*)', k)
                                    if match:
                                        conv_idx = int(match.group(1))
                                        suffix = match.group(2)
                                        body_idx = (conv_idx - 2) * 2
                                        new_k = f'body.{body_idx}.{suffix}'
                                        mapped_dict[new_k] = v
                                    else:
                                        mapped_dict[k] = v
                                elif k.startswith('conv_20.'):
                                    # conv_20 -> conv_last
                                    mapped_dict[k.replace('conv_20.', 'conv_last.')] = v
                                else:
                                    mapped_dict[k] = v
                            
                            state_dict = mapped_dict

                    # Specific handling for SRCNN mismatch
                    # Common checkpoint naming patterns for SRCNN
                    if m_name == 'SRCNN':
                        # Check for various naming conventions
                        # Pattern 1: layer1, layer2, layer3 (common in some repos)
                        if 'layer1.weight' in state_dict or 'layer1.0.weight' in state_dict:
                            mapped_dict = OrderedDict()
                            for k, v in state_dict.items():
                                if k.startswith('layer1.0.') or k.startswith('layer1.'):
                                    new_k = k.replace('layer1.0.', 'conv1.').replace('layer1.', 'conv1.')
                                    mapped_dict[new_k] = v
                                elif k.startswith('layer2.0.') or k.startswith('layer2.'):
                                    new_k = k.replace('layer2.0.', 'conv2.').replace('layer2.', 'conv2.')
                                    mapped_dict[new_k] = v
                                elif k.startswith('layer3.0.') or k.startswith('layer3.'):
                                    new_k = k.replace('layer3.0.', 'conv3.').replace('layer3.', 'conv3.')
                                    mapped_dict[new_k] = v
                                else:
                                    mapped_dict[k] = v
                            state_dict = mapped_dict

                    model_other.load_state_dict(state_dict, strict=True)
                    model_other.eval().to(device)
                    # Get window_size from config if available (for SwinIR, etc.)
                    ws = m_conf.get('window_size', None)
                    
                    # VDSR and similar models require bicubic-upscaled input
                    if m_conf.get('uses_bicubic_input', False):
                        # Upscale LR to HR size using bicubic first
                        img_bicubic_input = cv2.resize(img_lr, (w_new, h_new), interpolation=cv2.INTER_CUBIC)
                        img_bicubic_input = np.clip(img_bicubic_input, 0, 1)  # Clip to prevent artifacts
                        
                        # Handle Y-channel only models (like original SRCNN)
                        if m_conf.get('y_channel_only', False):
                            # Convert BGR to YCbCr
                            img_bicubic_uint8 = (img_bicubic_input * 255.0).round().astype(np.uint8)
                            img_ycbcr = cv2.cvtColor(img_bicubic_uint8, cv2.COLOR_BGR2YCrCb)
                            
                            # Extract Y channel and normalize to [0, 1]
                            y_channel = img_ycbcr[:, :, 0:1].astype(np.float32) / 255.0
                            
                            # Run inference on Y channel only
                            y_tensor = torch.from_numpy(y_channel.transpose(2, 0, 1)).unsqueeze(0).to(device)
                            with torch.no_grad():
                                y_sr = model_other(y_tensor)
                            y_sr = y_sr.squeeze(0).cpu().numpy().transpose(1, 2, 0)
                            y_sr = np.clip(y_sr * 255.0, 0, 255).astype(np.uint8)
                            
                            # Combine SR Y with bicubic CbCr
                            img_ycbcr[:, :, 0] = y_sr[:, :, 0]
                            
                            # Convert back to BGR
                            res = cv2.cvtColor(img_ycbcr, cv2.COLOR_YCrCb2BGR)
                        else:
                            res = inference_model(model_other, img_bicubic_input, device, window_size=ws, scale=1)
                    else:
                        res = inference_model(model_other, img_lr, device, window_size=ws, scale=scale)
                    results_others.append((m_name, res))
                except Exception as e:
                    print(f"Error running {m_name}: {e}")
                    import traceback
                    traceback.print_exc()

            # 6. Prepare visuals
            img_hr_disp = (img_hr * 255.0).round().astype(np.uint8)
            img_lr_disp = (img_lr * 255.0).round().astype(np.uint8)
            img_bicubic_disp = (img_bicubic * 255.0).round().astype(np.uint8)
            img_lr_vis = cv2.resize(img_lr_disp, (w_new, h_new), interpolation=cv2.INTER_NEAREST)

            # Calculate Metrics
            psnr_bic, ssim_bic = get_metrics(img_bicubic_disp, img_hr_disp, crop_border=scale)
            psnr_our, ssim_our = get_metrics(res_our, img_hr_disp, crop_border=scale)

            # Determine Crop Coordinates (High Variance Area)
            crop_size = 120
            # Ensure crop_size isn't larger than image
            crop_size = min(crop_size, h_new, w_new)
            
            def get_best_crop_coords(img, c_size):
                # Simple sliding window variance
                best_var = -1
                best_y, best_x = 0, 0
                step = c_size // 2
                
                # Convert to gray for variance check
                if len(img.shape) == 3:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    gray = img
                
                h, w = gray.shape
                for y in range(0, h - c_size + 1, step):
                    for x in range(0, w - c_size + 1, step):
                        patch = gray[y:y+c_size, x:x+c_size]
                        var = np.var(patch)
                        if var > best_var:
                            best_var = var
                            best_y, best_x = y, x
                return best_y, best_x

            cy, cx = get_best_crop_coords(img_hr_disp, crop_size)
            
            # Prepare Full HR with Box
            img_hr_full_vis = img_hr_disp.copy()
            # Red Box (BGR: 0, 0, 255)
            cv2.rectangle(img_hr_full_vis, (cx, cy), (cx+crop_size, cy+crop_size), (0, 0, 255), 4)

            # Colors for ranking
            RED = (255, 0, 0)
            BLUE = (0, 0, 255)
            BLACK = (0, 0, 0)

            # Collection of Items with Metadata for Ranking
            comp_items = []
            
            # HR Crop (Reference, not ranked)
            hr_crop = img_hr_disp[cy:cy+crop_size, cx:cx+crop_size]
            comp_items.append({
                'name': 'HR (Crop)',
                'img': hr_crop,
                'psnr': float('inf'),
                'ssim': 1.0,
                'is_model': False
            })
            
            # Bicubic
            bic_crop = img_bicubic_disp[cy:cy+crop_size, cx:cx+crop_size]
            comp_items.append({
                'name': 'Bicubic',
                'img': bic_crop,
                'psnr': psnr_bic,
                'ssim': ssim_bic,
                'is_model': True
            })

            # Other Models
            for name, img in results_others:
                psnr, ssim = get_metrics(img, img_hr_disp, crop_border=scale)
                c_img = img[cy:cy+crop_size, cx:cx+crop_size]
                comp_items.append({
                    'name': name,
                    'img': c_img,
                    'psnr': psnr,
                    'ssim': ssim,
                    'is_model': True
                })

            # Our Model
            our_crop = res_our[cy:cy+crop_size, cx:cx+crop_size]
            comp_items.append({
                'name': 'GSA-LN (ours)',
                'img': our_crop,
                'psnr': psnr_our,
                'ssim': ssim_our,
                'is_model': True
            })

            # Calculate Rankings (only for models)
            pps = sorted([item['psnr'] for item in comp_items if item.get('is_model')], reverse=True)
            sss = sorted([item['ssim'] for item in comp_items if item.get('is_model')], reverse=True)
            
            best_psnr = pps[0] if pps else -1
            second_psnr = pps[1] if len(pps) > 1 else -1
            best_ssim = sss[0] if sss else -1
            second_ssim = sss[1] if len(sss) > 1 else -1

           # 7. Create Composite Image with aesthetic layout
            # Requirement: HR large on left. Others (LR, Bicubic, Models) on right in 2 rows.
            # Labels at bottom.
            # Use Arial Font using PIL.
            
            # Font Config
            font_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'fonts', 'Arial.ttf')
            
            # Robust font search
            possible_paths = [
                "arial.ttf",
                "Arial.ttf",
            ]
            
            if os.name == 'nt':
                possible_paths.extend([
                    os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), 'Fonts', 'arial.ttf'),
                    os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), 'Fonts', 'Arial.ttf'),
                ])
            else:
                 possible_paths.extend([
                    "/mnt/c/Windows/Fonts/arial.ttf",
                    "/mnt/c/Windows/Fonts/Arial.ttf",
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                     "/usr/share/fonts/truetype/freefont/FreeSans.ttf"
                ])
            
            # Try loading default first if path not specific
            for p in possible_paths:
                if os.path.exists(p):
                    font_path = p
                    break

            font_size = 20 # Smaller font for small crops? Or keep readable? 
            # If crops are 120px wide, font 48 is too big. 
            # Let's upscale crops for display? "Zooms" usually implies pixel replication or just showing raw pixels.
            # Visualizing 120x120 is small. Let's resize crops x2 for better visibility?
            visual_scale = 2
            
            text_pad = 60 # Pad for text
            padding = 10 
            
            # Use separate variable so logic below holds
            # Show LR input image (the actual input to models) with crop box
            label_hr = "LR input"
            
            # Draw red box on LR image to show crop area
            # Crop coordinates are for HR, so divide by scale for LR
            lr_cx = cx // scale
            lr_cy = cy // scale
            lr_crop_size = crop_size // scale
            img_lr_with_box = img_lr_disp.copy()
            cv2.rectangle(img_lr_with_box, 
                         (lr_cx, lr_cy), 
                         (lr_cx + lr_crop_size, lr_cy + lr_crop_size), 
                         (0, 0, 255), 2)  # Red color, thickness 2
            img_hr_vis = img_lr_with_box  # Use LR image with box
            
            # Helper to create labeled image with colored parts
            def create_labeled_chip(img, name_txt, metric_parts, target_h=None, target_w=None):
                # Resize image if target dimensions provided
                if target_h is not None and target_w is not None:
                    if img.shape[:2] != (target_h, target_w):
                        img = cv2.resize(img, (target_w, target_h))
                
                h_i, w_i = img.shape[:2]
                
                # Create canvas with bottom padding for text
                canvas = np.full((h_i + text_pad, w_i, 3), 255, dtype=np.uint8)
                canvas[:h_i, :, :] = img
                
                pil_canvas = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_canvas)
                
                try:
                    font = ImageFont.truetype(font_path, 28)
                except:
                    font = ImageFont.load_default()
                
                line_spacing = 5

                # Measure Name
                bbox_name = font.getbbox(name_txt)
                w_name = bbox_name[2] - bbox_name[0]
                h_name = bbox_name[3] - bbox_name[1]
                
                # Measure Metrics Line
                # metric_parts is list of (text, color)
                w_metrics = 0
                h_metrics = 0
                for txt, col in metric_parts:
                    bb = font.getbbox(txt)
                    w_metrics += (bb[2] - bb[0])
                    curr_h = bb[3] - bb[1]
                    h_metrics = max(h_metrics, curr_h)
                
                total_text_h = h_name + line_spacing + h_metrics
                y_start = h_i + (text_pad - total_text_h) // 2
                
                # Draw Name (Centered, Black)
                x_name = (w_i - w_name) // 2
                draw.text((x_name, y_start), name_txt, font=font, fill=(0,0,0))
                
                # Draw Metrics Parts (Centered as a block)
                y_metrics = y_start + h_name + line_spacing
                x_curr = (w_i - w_metrics) // 2
                
                for txt, col in metric_parts:
                    draw.text((x_curr, y_metrics), txt, font=font, fill=col)
                    bb = font.getbbox(txt)
                    x_curr += (bb[2] - bb[0])
                
                canvas = cv2.cvtColor(np.array(pil_canvas), cv2.COLOR_RGB2BGR)
                return canvas
            
            # Process items for grid (include HR crop)
            others_data = comp_items
            
            others_chips = []
            for item in others_data:
                img = item['img']
                # Upscale crop
                h_c, w_c = img.shape[:2]
                img_big = cv2.resize(img, (w_c * visual_scale, h_c * visual_scale), interpolation=cv2.INTER_NEAREST)
                
                # Determine Metric Parts with colors
                if item.get('is_model'):
                    p = item['psnr']
                    s = item['ssim']
                    
                    p_col = BLACK
                    if p == best_psnr: p_col = RED
                    elif p == second_psnr: p_col = BLUE
                    
                    s_col = BLACK
                    if s == best_ssim: s_col = RED
                    elif s == second_ssim: s_col = BLUE
                    
                    parts = [
                        (f"{p:.2f}", p_col),
                        ("/", BLACK),
                        (f"{s:.4f}", s_col)
                    ]
                else:
                    parts = [("PSNR/SSIM: Inf", BLACK)]
                
                chip = create_labeled_chip(img_big, item['name'], parts)
                others_chips.append(chip)
                 
            # Grid calculations for 'others'
            n_others = len(others_chips)
            n_rows = 2
            n_cols = math.ceil(n_others / n_rows)
            
            # Standard size from first chip
            h_std, w_std = others_chips[0].shape[:2]
            
            # Create Right Grid first to determine total height
            # Each cell size
            cell_h = h_std
            cell_w = w_std
            
            # Total height of right grid = n_rows * cell_h + (n_rows+1) * padding
            right_h_total = n_rows * cell_h + (n_rows + 1) * padding
            # Total width of right grid = n_cols * cell_w + (n_cols+1) * padding
            right_w_total = n_cols * cell_w + (n_cols + 1) * padding
            
            right_grid_img = np.full((right_h_total, right_w_total, 3), 255, dtype=np.uint8)
            
            for idx, chip in enumerate(others_chips):
                row = idx // n_cols
                col = idx % n_cols
                
                y = padding + row * (cell_h + padding)
                x = padding + col * (cell_w + padding)
                
                if y + cell_h <= right_h_total and x + cell_w <= right_w_total:
                     right_grid_img[y:y+cell_h, x:x+cell_w] = chip

            # Create Large HR Image
            # Height should match right_grid_img height
            # HR Image Height + Text Pad = right_h_total
            hr_target_thumb_h = right_h_total - text_pad
            # Calculate width to maintain aspect ratio
            h_hr, w_hr = img_hr_vis.shape[:2]
            aspect = w_hr / h_hr
            hr_target_thumb_w = int(hr_target_thumb_h * aspect)
            
            hr_parts = [("PSNR/SSIM: Inf", BLACK)]
            hr_chip = create_labeled_chip(img_hr_vis, label_hr, hr_parts, hr_target_thumb_h, hr_target_thumb_w)
            
            # Final Composition
            # [HR] [Padding] [Right Grid]
            total_composite_w = hr_chip.shape[1] + padding + right_grid_img.shape[1] + 2 * padding
            total_composite_h = right_h_total + 2 * padding
            
            combined_image = np.full((total_composite_h, total_composite_w, 3), 255, dtype=np.uint8)
            
            # Place HR
            y_hr = padding
            x_hr = padding
            combined_image[y_hr:y_hr+hr_chip.shape[0], x_hr:x_hr+hr_chip.shape[1]] = hr_chip
            
            # Place Right Grid
            y_rg = padding
            x_rg = x_hr + hr_chip.shape[1] + padding
            combined_image[y_rg:y_rg+right_grid_img.shape[0], x_rg:x_rg+right_grid_img.shape[1]] = right_grid_img

            # 8. Save
            img_base_name = os.path.splitext(os.path.basename(hr_path))[0]
            output_name = f"comparison_{dataset_name}_{img_base_name}.png"
            output_path = os.path.join(args.output_dir, output_name)
            
            cv2.imwrite(output_path, combined_image)
            print(f"Saved: {output_path}")

        except Exception as e:
            print(f"Failed to process {dataset_name}/{os.path.basename(hr_path)}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    main()
