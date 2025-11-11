import torch
import yaml
import numpy as np
import os
import argparse
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from src.backbones.edsr import EDSR
from src.utils.datasets import SuperResolutionDataset
from src.utils.device import get_device

def main():
    # --- 1. Setup --- 
    parser = argparse.ArgumentParser(description='Evaluate a Super-Resolution model on a specific dataset.')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the evaluation dataset (e.g., Set5, Set14).')
    args = parser.parse_args()

    # Load configuration
    with open('configs/config.yml', 'r') as f:
        config = yaml.safe_load(f)

    # Setup device
    device = get_device(config)

    # --- 2. Initialize Model --- 
    model_config = config['model']
    model = EDSR(
        n_resblocks=model_config['n_resblocks'],
        n_feats=model_config['n_feats'],
        scale=model_config['scale'],
        rgb_range=model_config['rgb_range'],
        n_colors=model_config['n_colors'],
        res_scale=model_config['res_scale']
    ).to(device)

    # Load pre-trained weights
    test_config = config['test']
    model_path = test_config['pretrained_model']
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded model from {model_path}")

    # --- 3. Prepare Dataset --- 
    scale = model_config['scale']
    eval_hr_dir = os.path.join('data', 'evaluation', args.dataset, 'HR')
    eval_lr_dir = os.path.join('data', 'evaluation', args.dataset, f'LR_bicubic/X{scale}')

    if not os.path.exists(eval_hr_dir) or not os.path.exists(eval_lr_dir):
        print(f"Error: Dataset directories not found for '{args.dataset}'.")
        print(f"Looked for HR in: {eval_hr_dir}")
        print(f"Looked for LR in: {eval_lr_dir}")
        return

    val_dataset = SuperResolutionDataset(
        hr_dir=eval_hr_dir,
        lr_dir=eval_lr_dir,
        scale=scale,
        mode='val'
    )
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    print(f"Evaluating on: {args.dataset}")

    # --- 4. Evaluation Loop --- 
    total_psnr = 0
    total_ssim = 0

    with torch.no_grad():
        for i, (lr_tensor, hr_tensor) in enumerate(val_dataloader):
            lr_tensor, hr_tensor = lr_tensor.to(device), hr_tensor.to(device)
            sr_tensor = model(lr_tensor)

            # Crop images to the same size
            h, w = sr_tensor.shape[-2], sr_tensor.shape[-1]
            hr_tensor_cropped = hr_tensor[..., :h, :w]

            # Convert tensors to numpy arrays in range [0, 255]
            sr_img = sr_tensor.squeeze(0).mul(255).clamp(0, 255).byte().cpu().numpy().transpose(1, 2, 0)
            hr_img = hr_tensor_cropped.squeeze(0).mul(255).clamp(0, 255).byte().cpu().numpy().transpose(1, 2, 0)

            current_psnr = psnr(hr_img, sr_img, data_range=255)
            current_ssim = ssim(hr_img, sr_img, data_range=255, channel_axis=2, win_size=7)

            total_psnr += current_psnr
            total_ssim += current_ssim
            print(f"Evaluated image {i + 1}/{len(val_dataloader)}: PSNR: {current_psnr:.4f}, SSIM: {current_ssim:.4f}")

    # --- 5. Display Results --- 
    avg_psnr = total_psnr / len(val_dataloader)
    avg_ssim = total_ssim / len(val_dataloader)

    print("\n---------------------")
    print(f"Dataset: {args.dataset}")
    print(f"Average PSNR: {avg_psnr:.4f}")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print("---------------------")

if __name__ == '__main__':
    main()