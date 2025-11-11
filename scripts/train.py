import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import sys
import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from src.backbones.edsr import EDSR
from src.utils.datasets import SuperResolutionDataset
from src.utils.device import get_device

class Trainer:
    def __init__(self, config, eval_dataset_name):
        self.config = config
        self.device = get_device(config)
        self.model = self._init_model()
        self.criterion = torch.nn.L1Loss()
        self.optimizer = self._init_optimizer()
        self.train_dataloader = self._init_dataloader(config['data']['train_hr'], config['data']['train_lr'], 'train')

        scale = self.config['model']['scale']
        hr_dir = f"data/evaluation/{eval_dataset_name}/HR"
        lr_dir = f"data/evaluation/{eval_dataset_name}/LR_bicubic/X{scale}"
        self.val_dataloader = self._init_dataloader(hr_dir, lr_dir, 'val')

        self.output_dir = 'pretrained'
        os.makedirs(self.output_dir, exist_ok=True)

    def _init_model(self):
        model_config = self.config['model']
        model = EDSR(
            n_resblocks=model_config['n_resblocks'],
            n_feats=model_config['n_feats'],
            scale=model_config['scale'],
            rgb_range=model_config['rgb_range'],
            n_colors=model_config['n_colors'],
            res_scale=model_config['res_scale']
        )
        model.to(self.device)
        return model

    def _init_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.config['training']['lr'])

    def _init_dataloader(self, hr_dir, lr_dir, mode):
        training_config = self.config['training']
        dataset = SuperResolutionDataset(
            hr_dir=hr_dir,
            lr_dir=lr_dir,
            scale=self.config['model']['scale'],
            mode=mode,
            lr_patch_size=training_config['lr_patch_size'] if mode == 'train' else None
        )
        return DataLoader(dataset, batch_size=training_config['batch_size'] if mode == 'train' else 1, shuffle=mode == 'train')

    def _train_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0.0
        num_batches = len(self.train_dataloader)
        for i, (inputs, targets) in enumerate(self.train_dataloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

            progress = (i + 1) / num_batches
            progress_bar = '#' * int(20 * progress) + '-' * (20 - int(20 * progress))
            sys.stdout.write(f'\rEpoch {epoch + 1}/{self.config["training"]["epochs"]} [{progress_bar}] {progress * 100:.2f}% - Batch Loss: {loss.item():.4f}')
            sys.stdout.flush()
        
        avg_epoch_loss = epoch_loss / num_batches
        sys.stdout.write('\r' + ' ' * 80 + '\r')
        sys.stdout.flush()
        return avg_epoch_loss

    def _run_evaluation(self):
        self.model.eval()
        total_psnr = 0
        total_ssim = 0
        with torch.no_grad():
            for inputs, targets in self.val_dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                sr = self.model(inputs)

                # Crop images to the same size
                h, w = sr.shape[-2], sr.shape[-1]
                hr_cropped = targets[..., :h, :w]

                # Convert tensors to numpy arrays in range [0, 255]
                sr_img = sr.squeeze(0).mul(255).clamp(0, 255).byte().cpu().numpy().transpose(1, 2, 0)
                hr_img = hr_cropped.squeeze(0).mul(255).clamp(0, 255).byte().cpu().numpy().transpose(1, 2, 0)

                total_psnr += psnr(hr_img, sr_img, data_range=255)
                total_ssim += ssim(hr_img, sr_img, data_range=255, channel_axis=2, win_size=7)

        avg_psnr = total_psnr / len(self.val_dataloader)
        avg_ssim = total_ssim / len(self.val_dataloader)
        return avg_psnr, avg_ssim

    def _save_checkpoint(self, epoch):
        scale = self.config['model']['scale']
        save_path = os.path.join(self.output_dir, f'trifa_x{scale}_epoch_{epoch+1}.pth')
        torch.save(self.model.state_dict(), save_path)

    def run(self):
        for epoch in range(self.config['training']['epochs']):
            loss = self._train_epoch(epoch)
            avg_psnr, avg_ssim = self._run_evaluation()
            print(f"Epoch [{epoch+1}/{self.config['training']['epochs']}], Loss: {loss:.4f}, PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f}")
            self._save_checkpoint(epoch)
        print(f"\nTraining complete. Models saved in {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Train the model.")
    parser.add_argument('--config', type=str, default='configs/config.yml', help='Path to the config file.')
    parser.add_argument('--eval', type=str, default='Set5', help='Evaluation dataset name (e.g., Set5, Set14).')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    trainer = Trainer(config, args.eval)
    trainer.run()

if __name__ == '__main__':
    main()