import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from src.backbones.edsr import EDSR
from src.utils.datasets import SuperResolutionDataset
from src.utils.device import get_device
import sys
import os

def main():
    # Load configuration
    with open('configs/config.yml', 'r') as f:
        config = yaml.safe_load(f)

    # Initialize model
    model_config = config['model']
    model = EDSR(
        n_resblocks=model_config['n_resblocks'],
        n_feats=model_config['n_feats'],
        scale=model_config['scale'],
        rgb_range=model_config['rgb_range'],
        n_colors=model_config['n_colors'],
        res_scale=model_config['res_scale']
    )

    # Setup device
    device = get_device(config)
    model.to(device)

    # Loss and optimizer
    criterion = torch.nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'])

    # Dataloader
    data_config = config['data']
    training_config = config['training']
    
    train_dataset = SuperResolutionDataset(
        hr_dir=data_config['train_hr'],
        lr_dir=data_config['train_lr'],
        scale=model_config['scale'],
        mode='train',
        lr_patch_size=training_config['lr_patch_size']
    )
    train_dataloader = DataLoader(train_dataset, batch_size=training_config['batch_size'], shuffle=True)

    # Training loop
    output_dir = 'pretrained'
    os.makedirs(output_dir, exist_ok=True)
    scale = model_config['scale']

    for epoch in range(training_config['epochs']):
        # --- Training ---
        model.train()
        epoch_loss = 0.0
        num_batches = len(train_dataloader)
        for i, batch in enumerate(train_dataloader):
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            progress = (i + 1) / num_batches
            progress_bar = '#' * int(20 * progress) + '-' * (20 - int(20 * progress))
            sys.stdout.write(f'\rEpoch {epoch + 1}/{training_config["epochs"]} [{progress_bar}] {progress * 100:.2f}% - Batch Loss: {loss.item():.4f}')
            sys.stdout.flush()
        
        avg_epoch_loss = epoch_loss / num_batches
        sys.stdout.write('\r' + ' ' * 80 + '\r')
        sys.stdout.flush()

        print(f"Epoch [{epoch+1}/{training_config['epochs']}], Loss: {avg_epoch_loss:.4f}")

        # Save the model
        save_path = os.path.join(output_dir, f'trifa_x{scale}_epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), save_path)

    print(f"\nTraining complete. Models saved in {output_dir}")

if __name__ == '__main__':
    main()