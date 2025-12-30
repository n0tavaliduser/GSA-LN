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
    parser.add_argument('--input_path', type=str, default='datasets/single', help='Path to image or directory containing images')
    parser.add_argument('--config', type=str, default='options/test/test_gsaln_x2_tta_x8.yml', help='Path to our model config')
    parser.add_argument('--output_dir', type=str, default='results/comparison', help='Output directory for results')
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

    # 2. Define Other Models to Compare
    # You can add more models here manually.
    other_models_config = [
        # {'name': 'ESRGAN', 'path': 'models/ESRGAN_x4.pth', 'arch': RRDBNet, 'args': {'num_in_ch': 3, 'num_out_ch': 3, 'scale': 4}},
    ]
    
    # Determine input images
    img_paths = []
    if os.path.isdir(args.input_path):
        print(f"Scanning directory {args.input_path} for images...")
        for root, _, files in os.walk(args.input_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    img_paths.append(os.path.join(root, file))
    elif os.path.isfile(args.input_path):
        img_paths.append(args.input_path)
    else:
        print(f"Error: Input path {args.input_path} not found.")
        return

    if not img_paths:
        print("No images found to process.")
        return

    print(f"Found {len(img_paths)} images.")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # 3. Process Images Loop
    for idx, img_path in enumerate(img_paths):
        print(f"[{idx+1}/{len(img_paths)}] Processing {img_path}...")
        
        try:
            img_hr = load_image(img_path)
            h, w, _ = img_hr.shape

            # Ensure divisible by scale
            h_new, w_new = h - h % scale, w - w % scale
            img_hr = img_hr[:h_new, :w_new, :]
            
            # Generate LR (Downsample)
            img_lr = cv2.resize(img_hr, (w_new // scale, h_new // scale), interpolation=cv2.INTER_CUBIC)

            # Generate Bicubic Upscale (Baseline)
            img_bicubic = cv2.resize(img_lr, (w_new, h_new), interpolation=cv2.INTER_CUBIC)

            # 4. Inference Our Model
            res_our = inference_model(model_g, img_lr, device)

            # 5. Inference Other Models
            results_others = []
            for m_conf in other_models_config:
                m_name = m_conf['name']
                m_path = m_conf['path']
                if not os.path.exists(m_path):
                    continue
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
            img_hr_disp = (img_hr * 255.0).round().astype(np.uint8)
            img_lr_disp = (img_lr * 255.0).round().astype(np.uint8)
            img_bicubic_disp = (img_bicubic * 255.0).round().astype(np.uint8)
            img_lr_vis = cv2.resize(img_lr_disp, (w_new, h_new), interpolation=cv2.INTER_NEAREST)

            # Calculate Metrics
            psnr_bic, ssim_bic = get_metrics(img_bicubic_disp, img_hr_disp, crop_border=scale)
            psnr_our, ssim_our = get_metrics(res_our, img_hr_disp, crop_border=scale)

            # Collection
            images_to_show = [
                ('HR (Ground Truth)\nPSNR/SSIM: Inf', img_hr_disp),
                (f'LR (Input x{scale})\n', img_lr_vis),
                (f'Bicubic\n{psnr_bic:.2f}/{ssim_bic:.4f}', img_bicubic_disp),
            ]

            for name, img in results_others:
                psnr, ssim = get_metrics(img, img_hr_disp, crop_border=scale)
                images_to_show.append((f'{name}\n{psnr:.2f}/{ssim:.4f}', img))

            our_label = f"Our Model (GSA-LN)\n{psnr_our:.2f}/{ssim_our:.4f}"
            images_to_show.append((our_label, res_our))

           # 7. Create Composite Image with aesthetic layout
            # Requirement: HR large on left. Others (LR, Bicubic, Models) on right in 2 rows.
            # Labels at bottom.
            # Use Arial Font using PIL.
            
            # Font Config
            font_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'fonts', 'Arial.ttf')
            font_size = 32
            text_pad = 80
            padding = 20 # Space between images
            
            # Separate HR from others
            label_hr, img_hr_vis = images_to_show[0]
            others = images_to_show[1:]

            # Grid calculations for 'others'
            n_others = len(others)
            n_rows = 2
            n_cols = math.ceil(n_others / n_rows)
            
            # Use standardized size from the HR/LR basic shapes
            h_std, w_std = h_new, w_new
            
            # Helper to create labeled image with label at DOWN
            def create_labeled_chip(img, txt, target_h=None, target_w=None):
                # Resize image if target dimensions provided
                if target_h is not None and target_w is not None:
                    if img.shape[:2] != (target_h, target_w):
                        img = cv2.resize(img, (target_w, target_h))
                
                h_i, w_i = img.shape[:2]
                
                # Create canvas with bottom padding for text
                canvas = np.full((h_i + text_pad, w_i, 3), 255, dtype=np.uint8)
                canvas[:h_i, :, :] = img
                
                # Draw text centered in the bottom area using PIL
                # Convert canvas to PIL
                # We need to process multiline text centering manually 
                
                pil_canvas = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_canvas)
                
                try:
                    font = ImageFont.truetype(font_path, font_size)
                except:
                    font = ImageFont.load_default()
                
                lines = txt.split('\n')
                line_spacing = 5
                
                # Calculating total block height
                total_text_h = 0
                widths = []
                for line in lines:
                    bbox = font.getbbox(line)
                    w_l = bbox[2] - bbox[0]
                    h_l = bbox[3] - bbox[1]
                    widths.append(w_l)
                    total_text_h += h_l + line_spacing
                total_text_h -= line_spacing # remove last spacing
                
                # Start Y position
                y_start = h_i + (text_pad - total_text_h) // 2
                
                current_y = y_start
                for line in lines:
                    bbox = font.getbbox(line)
                    w_l = bbox[2] - bbox[0]
                    h_l = bbox[3] - bbox[1]
                    
                    x = (w_i - w_l) // 2
                    
                    draw.text((x, current_y), line, font=font, fill=(0, 0, 0))
                    current_y += h_l + line_spacing
                
                canvas = cv2.cvtColor(np.array(pil_canvas), cv2.COLOR_RGB2BGR)
                return canvas

            # Create Right Grid first to determine total height
            # Each cell size
            cell_h = h_std
            cell_w = w_std
            
            # Total height of right grid = 2 * (cell_h + text_pad) + padding
            right_h_total = 2 * (cell_h + text_pad) + padding
            # Total width of right grid = n_cols * cell_w + (n_cols-1) * padding
            right_w_total = n_cols * cell_w + (max(0, n_cols - 1)) * padding
            
            right_grid_img = np.full((right_h_total, right_w_total, 3), 255, dtype=np.uint8)
            
            for idx, (lbl, img) in enumerate(others):
                chip = create_labeled_chip(img, lbl, cell_h, cell_w)
                
                row = idx // n_cols
                col = idx % n_cols
                
                y = row * (cell_h + text_pad + padding)
                x = col * (cell_w + padding)
                
                if y + chip.shape[0] <= right_h_total and x + chip.shape[1] <= right_w_total:
                     right_grid_img[y:y+chip.shape[0], x:x+chip.shape[1]] = chip

            # Create Large HR Image
            # Height should match right_grid_img height
            # HR Image Height + Text Pad = right_h_total
            hr_target_thumb_h = right_h_total - text_pad
            # Calculate width to maintain aspect ratio
            aspect = w_std / h_std
            hr_target_thumb_w = int(hr_target_thumb_h * aspect)
            
            hr_chip = create_labeled_chip(img_hr_vis, label_hr, hr_target_thumb_h, hr_target_thumb_w)
            
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
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            output_name = f"comparison_{base_name}.png"
            output_path = os.path.join(args.output_dir, output_name)
            
            cv2.imwrite(output_path, combined_image)
            print(f"Saved: {output_path}")

        except Exception as e:
            print(f"Failed to process {img_path}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    main()

if __name__ == '__main__':
    main()
