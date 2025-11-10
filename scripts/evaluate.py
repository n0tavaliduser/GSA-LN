import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from src.backbones.edsr import EDSR
from src.utils.datasets import SuperResolutionDataset

def main():
    # Load configuration
    with open('configs/config.yml', 'r') as f:
        config = yaml.safe_load(f)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize model from config
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
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded model from {model_path}")

    # Prepare validation dataset and dataloader
    dataset_config = config['data']
    training_config = config['training']
    val_dataset = SuperResolutionDataset(
        hr_dir=dataset_config['valid_hr'],
        lr_dir=dataset_config['valid_lr'],
        scale=config['model']['scale'],
        mode='val'
    )
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Initialize metrics
    total_psnr = 0
    total_ssim = 0

    with torch.no_grad():
        for i, (lr_image, hr_image) in enumerate(val_dataloader):
            lr_image, hr_image = lr_image.to(device), hr_image.to(device)

            sr_image = model(lr_image)

            # Clamp the output to [0, 1] range before converting to numpy
            sr_image = torch.clamp(sr_image, 0, 1)

            # Convert tensors to numpy arrays for evaluation
            # Tensors are in (B, C, H, W) format, and normalized to [0, 1]
            sr_image_np = sr_image.squeeze(0).cpu().numpy()
            hr_image_np = hr_image.squeeze(0).cpu().numpy()

            # Transpose from (C, H, W) to (H, W, C) for skimage metrics
            sr_image_np = sr_image_np.transpose(1, 2, 0)
            hr_image_np = hr_image_np.transpose(1, 2, 0)

            # Calculate PSNR and SSIM
            # data_range is 1.0 because images are normalized to [0, 1]
            current_psnr = psnr(hr_image_np, sr_image_np, data_range=1.0)
            # For multichannel (color) images, ssim requires the channel_axis argument in recent skimage versions
            try:
                current_ssim = ssim(hr_image_np, sr_image_np, data_range=1.0, channel_axis=-1, multichannel=True)
            except TypeError:
                # Fallback for older skimage versions
                current_ssim = ssim(hr_image_np, sr_image_np, data_range=1.0, multichannel=True)

            total_psnr += current_psnr
            total_ssim += current_ssim
            print(f"Evaluated image {i + 1}/{len(val_dataloader)}: PSNR: {current_psnr:.4f}, SSIM: {current_ssim:.4f}")

    avg_psnr = total_psnr / len(val_dataloader)
    avg_ssim = total_ssim / len(val_dataloader)

    print("\n---------------------")
    print(f"Average PSNR: {avg_psnr:.4f}")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print("---------------------")

if __name__ == '__main__':
    main()