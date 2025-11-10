import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import random
from torchvision.transforms.functional import crop, hflip, vflip, rotate

class SuperResolutionDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, scale, mode='train', lr_patch_size=None):
        super().__init__()
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.scale = scale
        self.mode = mode
        self.lr_patch_size = lr_patch_size
        self.hr_images = sorted(os.listdir(hr_dir))
        self.lr_images = sorted(os.listdir(lr_dir))
        self.to_tensor = ToTensor()

        if self.mode == 'train' and self.lr_patch_size is None:
            raise ValueError("lr_patch_size must be specified for training mode.")

    def __getitem__(self, index):
        hr_image_path = os.path.join(self.hr_dir, self.hr_images[index])
        
        hr_filename = self.hr_images[index]
        lr_filename = hr_filename.replace('.png', f'x{self.scale}.png')
        lr_image_path = os.path.join(self.lr_dir, lr_filename)

        hr_image = Image.open(hr_image_path).convert('RGB')
        lr_image = Image.open(lr_image_path).convert('RGB')

        if self.mode == 'train':
            # Random crop
            lr_w, lr_h = lr_image.size
            hr_patch_size = self.lr_patch_size * self.scale
            
            if lr_w < self.lr_patch_size or lr_h < self.lr_patch_size:
                raise ValueError(f"Image {self.hr_images[index]} is too small for the patch size.")

            lr_x = random.randint(0, lr_w - self.lr_patch_size)
            lr_y = random.randint(0, lr_h - self.lr_patch_size)
            hr_x = lr_x * self.scale
            hr_y = lr_y * self.scale

            lr_patch = crop(lr_image, lr_y, lr_x, self.lr_patch_size, self.lr_patch_size)
            hr_patch = crop(hr_image, hr_y, hr_x, hr_patch_size, hr_patch_size)

            # Augmentation
            if random.random() > 0.5:
                lr_patch = hflip(lr_patch)
                hr_patch = hflip(hr_patch)
            if random.random() > 0.5:
                lr_patch = vflip(lr_patch)
                hr_patch = vflip(hr_patch)
            angle = random.choice([0, 90, 180, 270])
            lr_patch = rotate(lr_patch, angle)
            hr_patch = rotate(hr_patch, angle)

            hr_tensor = self.to_tensor(hr_patch)
            lr_tensor = self.to_tensor(lr_patch)
        else: # Validation/Test mode
            hr_tensor = self.to_tensor(hr_image)
            lr_tensor = self.to_tensor(lr_image)

        return lr_tensor, hr_tensor

    def __len__(self):
        return len(self.hr_images)