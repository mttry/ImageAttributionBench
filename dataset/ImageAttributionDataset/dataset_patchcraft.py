import torchvision.transforms as T  
from .dataset import ImageAttributionDataset
from ImageAttributionDataset import DATASET
from PIL import Image, ImageFilter  
import random  
import numpy as np  
import io 

@DATASET.register_module(module_name='patchcraft')
class PatchCraftDataset(ImageAttributionDataset):  
    def __init__(self, root_dir, num_images_per_semantic_per_class=2000, transform=None,degraded=0, **kwargs):  
        super().__init__(root_dir, num_images_per_semantic_per_class, transform, degraded=degraded)  
        if self.transform is None:  
            self.transform = T.Compose([  
                T.Resize(256),  
                T.CenterCrop(256),  
                T.ToTensor(),  
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  
            ])  

    def __getitem__(self, idx):  
        item = super().__getitem__(idx)  
        image = item['image']  

        enhanced_image = random_augmentation(image)  

        image_np = np.array(enhanced_image)  

        p_reconstructed, r_reconstructed = split_and_reconstruct_image(image_np)  

        if self.transform:  
            r_reconstructed = self.transform(Image.fromarray(r_reconstructed))  
            p_reconstructed = self.transform(Image.fromarray(p_reconstructed))  
            image = self.transform(image) 
        item['img1'] = r_reconstructed  
        item['img2'] = p_reconstructed  
        item['image'] = image  
        return item  

def random_augmentation(image):  
    if random.random() < 0.1: 
        quality = random.randint(70, 100)  
        image = jpeg_compression(image, quality)  

    if random.random() < 0.1:  
        sigma = random.uniform(0, 1)   
        image = gaussian_blur(image, sigma)  

    if random.random() < 0.1: 
        scale = random.uniform(0.25, 0.5)  
        image = downsample(image, scale)  

    return image  


def compute(patch):  
    weight, height = patch.size  
    res = 0  
    patch = np.array(patch).astype(np.int64)  
    diff_horizontal = np.sum(np.abs(patch[:, :-1, :] - patch[:, 1:, :]))  
    diff_vertical = np.sum(np.abs(patch[:-1, :, :] - patch[1:, :, :]))  
    diff_diagonal = np.sum(np.abs(patch[:-1, :-1, :] - patch[1:, 1:, :]))  
    diff_diagonal += np.sum(np.abs(patch[1:, :-1, :] - patch[:-1, 1:, :]))  
    res = diff_horizontal + diff_vertical + diff_diagonal  
    return res.sum()  

def split_and_reconstruct_image(image_np, patch_size=32):  
    height, width, _ = image_np.shape  
    patches = []  

    for i in range(0, height, patch_size):  
        for j in range(0, width, patch_size):  
            patch = image_np[i:i + patch_size, j:j + patch_size, :]  
            if patch.shape[0] == patch_size and patch.shape[1] == patch_size:  
                patches.append(patch)  

    if not patches:  
        print("no patch generated!")  
        return np.zeros((256, 256, 3), dtype=np.uint8), np.zeros((256, 256, 3), dtype=np.uint8)  

    diversity_scores = [compute(Image.fromarray(patch)) for patch in patches]  
    sorted_indices = np.argsort(diversity_scores)  

    num_patches = len(patches)  

    poor_texture_count = max(1, num_patches // 3) 
    poor_indices = sorted_indices[:poor_texture_count]  
    rich_texture_count = max(1, num_patches // 3)  
    rich_indices = sorted_indices[-rich_texture_count:] 

    poor_texture_indices = poor_indices[:64] if len(poor_indices) > 64 else poor_indices  
    rich_texture_indices = rich_indices[-64:] if len(rich_indices) > 64 else rich_indices  
    rich_texture_indices = rich_texture_indices[::-1] 
    poor_texture_patches = [patches[i] for i in poor_texture_indices]  
    rich_texture_patches = [patches[i] for i in rich_texture_indices]  

    p_reconstructed = reconstruct_image(poor_texture_patches, output_size=256) 
    r_reconstructed = reconstruct_image(rich_texture_patches, output_size=256)  

    return p_reconstructed, r_reconstructed  

def reconstruct_image(patches, output_size):  
    if not patches:  
        raise ValueError("no patch for reconstrction!")  

    patch_size = 32  
    grid_size = output_size // patch_size  # 256 // 32 = 8  
    image_reconstructed = np.zeros((grid_size * patch_size, grid_size * patch_size, 3), dtype=np.uint8)  

    total_patches = len(patches)  
    for i in range(grid_size):  
        for j in range(grid_size):  
            patch_index = (i * grid_size + j) % total_patches  
            image_reconstructed[i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size] = patches[patch_index]  

    return image_reconstructed 

def jpeg_compression(image, quality):  
    img_byte_arr = io.BytesIO()  
    image.save(img_byte_arr, format='JPEG', quality=quality)  
    img_byte_arr.seek(0)  
    return Image.open(img_byte_arr)  

def gaussian_blur(image, sigma):  
    return image.filter(ImageFilter.GaussianBlur(sigma))  

def downsample(image, scale):  
    new_size = (int(image.size[0] * scale), int(image.size[1] * scale))  
    return image.resize(new_size, Image.BILINEAR)  