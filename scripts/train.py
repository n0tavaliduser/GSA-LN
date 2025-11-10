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
    model = EDSR(
        n_resblocks=config['model']['n_resblocks'],
        n_feats=config['model']['n_feats'],
        scale=config['model']['scale'],
        rgb_range=config['model']['rgb_range'],
        n_colors=config['model']['n_colors'],
        res_scale=config['model']['res_scale']
    )

    # Setup device
    device = get_device(config)
    model.to(device)

    # Loss and optimizer
    criterion = torch.nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'])

    # Dataloader
    train_dataset = SuperResolutionDataset(
        hr_dir=config['data']['train_hr'],
        lr_dir=config['data']['train_lr'],
        scale=config['model']['scale'],
        mode='train',
        lr_patch_size=config['training']['lr_patch_size']
    )
    dataloader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)

    # Training loop
    for epoch in range(config['training']['epochs']):
        model.train()
        epoch_loss = 0.0
        num_batches = len(dataloader)
        for i, batch in enumerate(dataloader):
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
            sys.stdout.write(f'\rEpoch {epoch + 1}/{config["training"]["epochs"]} [{progress_bar}] {progress * 100:.2f}% - Batch Loss: {loss.item():.4f}')
            sys.stdout.flush()
        
        avg_epoch_loss = epoch_loss / num_batches
        sys.stdout.write('\r' + ' ' * 80 + '\r') # Clear the line
        sys.stdout.flush()
        print(f"Epoch [{epoch+1}/{config['training']['epochs']}], Average Loss: {avg_epoch_loss:.4f}")

    # Save the model
    output_dir = 'pretrained'
    os.makedirs(output_dir, exist_ok=True)
    scale = config['model']['scale']
    save_path = os.path.join(output_dir, f'trifa_x{scale}.pth')
    torch.save(model.state_dict(), save_path)
    print(f"\nTraining complete. Model saved to {save_path}")
    print() # Newline after epoch

if __name__ == '__main__':
    main()