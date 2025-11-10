import torch
import yaml
import os
from PIL import Image
from torchvision import transforms
from src.backbones.edsr import EDSR

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

    # Prepare input and output directories
    input_dir = test_config['input_dir']
    output_dir = test_config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    # Image transformations
    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()

    # Process images
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for filename in image_files:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        print(f"Processing {filename}...")

        with Image.open(input_path).convert('RGB') as img:
            # Convert to tensor and add batch dimension
            input_tensor = to_tensor(img).unsqueeze(0).to(device)

            # Run inference
            with torch.no_grad():
                output_tensor = model(input_tensor)

            # Post-process and save
            output_image = output_tensor.squeeze(0).cpu()
            # Clamp values to the valid range [0, 1] before converting to PIL Image
            output_image = torch.clamp(output_image, 0, 1)
            output_image = to_pil(output_image)
            output_image.save(output_path)

    print(f"\nInference complete. Upscaled images are saved in {output_dir}")

if __name__ == '__main__':
    main()