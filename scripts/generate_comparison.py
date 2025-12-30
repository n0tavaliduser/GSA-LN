import os
import sys
import math
import cv2
import torch
import numpy as np
import yaml
import argparse
from collections import OrderedDict

# Ensure basicsr is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from basicsr.archs.gsaln_arch import GSALN
from basicsr.archs.swinir_arch import SwinIR
from basicsr.archs.rrdbnet_arch import RRDBNet
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

def inference_model(model, img_lr, device):
    img_lr_tensor = img2tensor(img_lr, bgr2rgb=True, float32=True).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_lr_tensor)
        
    return tensor2img(output, rgb2bgr=True, min_max=(0, 1))

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

def main():
    parser = argparse.ArgumentParser()
    # Default path handling: use user specified path if provided, else default
    parser.add_argument('--input_image', type=str, default='datasets/single/test_1.png', help='Path to testing image (HR)')
    parser.add_argument('--config', type=str, default='options/test/test_gsaln_x2_tta_x8.yml', help='Path to our model config')
    parser.add_argument('--output', type=str, default='results/comparison/comparison_result.png', help='Output image path')
    args = parser.parse_args()

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
    
    # Resolve relative path for model
    if model_path and not os.path.isabs(model_path):
        # Assuming path is relative to project root (where this script is)
         model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_path)
    
    model_loaded = False
    if model_path and os.path.exists(model_path):
        print(f"Loading weights from {model_path}")
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
        if 'params_ema' in state_dict:
            state_dict = state_dict['params_ema']
        elif 'params' in state_dict:
            state_dict = state_dict['params']
            
        model_g.load_state_dict(state_dict, strict=True)
        model_loaded = True
    else:
        print(f"Warning: Model path {model_path} not found. Running with initialized weights (random results).")
    
    model_g.eval()
    model_g.to(device)

    # 2. Define Other Models to Compare
    # You can add more models here manually.
    # Paths should be verified.
    other_models_config = [
        # {'name': 'ESRGAN', 'path': 'models/ESRGAN_x4.pth', 'arch': RRDBNet, 'args': {'num_in_ch': 3, 'num_out_ch': 3, 'scale': 4}},
        # {'name': 'SwinIR', 'path': 'models/001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth', 'arch': SwinIR, 'args': {'upscale': 2, 'in_chans': 3, 'img_size': 64, 'window_size': 8, 'img_range': 1., 'depths': [6, 6, 6, 6, 6, 6], 'embed_dim': 180, 'num_heads': [6, 6, 6, 6, 6, 6], 'mlp_ratio': 2, 'upsampler': 'pixelshuffle', 'resi_connection': '1conv'}},
    ]

    # 3. Process Image
    print(f"Processing image: {args.input_image}")
    if not os.path.exists(args.input_image):
        print(f"Error: Input image not found at {args.input_image}")
        # Try to find any png in datasets/single
        search_dir = r'datasets\single'
        if os.path.exists(search_dir):
            files = [f for f in os.listdir(search_dir) if f.endswith('.png') or f.endswith('.jpg')]
            if files:
                args.input_image = os.path.join(search_dir, files[0])
                print(f"Falling back to found image: {args.input_image}")
            else:
                return
        else:
            return

    img_hr = load_image(args.input_image)
    h, w, _ = img_hr.shape

    # Ensure divisible by scale
    h_new, w_new = h - h % scale, w - w % scale
    img_hr = img_hr[:h_new, :w_new, :]
    
    # Generate LR (Downsample)
    # Using cv2.resize with INTER_CUBIC 
    img_lr = cv2.resize(img_hr, (w_new // scale, h_new // scale), interpolation=cv2.INTER_CUBIC)

    # Generate Bicubic Upscale (Baseline)
    img_bicubic = cv2.resize(img_lr, (w_new, h_new), interpolation=cv2.INTER_CUBIC)

    # 4. Inference Our Model
    print("Running inference: Our Model")
    res_our = inference_model(model_g, img_lr, device)

    # 5. Inference Other Models
    results_others = []
    for m_conf in other_models_config:
        m_name = m_conf['name']
        m_path = m_conf['path']
        if not os.path.exists(m_path):
            print(f"Skipping {m_name}: Model file not found at {m_path}")
            continue
            
        print(f"Running inference: {m_name}")
        try:
            model_other = m_conf['arch'](**m_conf.get('args', {}))
            state_dict = torch.load(m_path, map_location=lambda storage, loc: storage)
            if 'params_ema' in state_dict:
                state_dict = state_dict['params_ema']
            elif 'params' in state_dict:
                state_dict = state_dict['params']
            model_other.load_state_dict(state_dict, strict=True)
            model_other.eval().to(device)
            res = inference_model(model_other, img_lr, device)
            results_others.append((m_name, res))
        except Exception as e:
            print(f"Error running {m_name}: {e}")

    # 6. Prepare visuals
    # Convert numpy inputs to uint8 (0-255) for display if not already
    img_hr_disp = (img_hr * 255.0).round().astype(np.uint8)
    img_lr_disp = (img_lr * 255.0).round().astype(np.uint8)
    img_bicubic_disp = (img_bicubic * 255.0).round().astype(np.uint8)

    # Resize LR to HR size for visualization (Nearest Neighbor to show pixelation)
    img_lr_vis = cv2.resize(img_lr_disp, (w_new, h_new), interpolation=cv2.INTER_NEAREST)

    # Calculate Metrics for baselines
    psnr_bic, ssim_bic = get_metrics(img_bicubic_disp, img_hr_disp, crop_border=scale)
    
    # Calculate Metrics for Our Model
    psnr_our, ssim_our = get_metrics(res_our, img_hr_disp, crop_border=scale)

    # Collection of images: (Label, Image)
    images_to_show = [
        ('HR (Ground Truth)\nPSNR/SSIM: Inf', img_hr_disp),
        (f'LR (Input x{scale})\n', img_lr_vis),
        (f'Bicubic\n{psnr_bic:.2f}/{ssim_bic:.4f}', img_bicubic_disp),
    ]

    # Add other models
    for name, img in results_others:
        psnr, ssim = get_metrics(img, img_hr_disp, crop_border=scale)
        images_to_show.append((f'{name}\n{psnr:.2f}/{ssim:.4f}', img))

    # Add Our Model
    # Determine label for our model (use filename of model if possible or config name)
    our_label = f"Our Model (GSA-LN)\n{psnr_our:.2f}/{ssim_our:.4f}"
    images_to_show.append((our_label, res_our))

    # 7. Create Composite Image
    # Concatenate horizontally
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    text_pad = 60 # padding for text
    
    final_list = []
    print("Generating Composite Image...")
    for label, img in images_to_show:
        # Make sure shapes match
        if img.shape != img_hr_disp.shape:
            img = cv2.resize(img, (w_new, h_new))
            
        # Add label on top
        # Create white padding
        labeled_img = np.full((h_new + text_pad, w_new, 3), 255, dtype=np.uint8)
        labeled_img[text_pad:, :, :] = img
        
        # Handle multi-line labels
        y0, dy = 25, 25
        for i, line in enumerate(label.split('\n')):
            y = y0 + i * dy
            text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
            text_x = (w_new - text_size[0]) // 2
            text_x = max(0, text_x)
            cv2.putText(labeled_img, line, (text_x, y), font, font_scale, (0, 0, 0), thickness)
        
        final_list.append(labeled_img)
        
    combined_image = np.hstack(final_list)

    # 8. Save
    print(f"Saving comparison to {args.output}")
    cv2.imwrite(args.output, combined_image)

if __name__ == '__main__':
    main()
