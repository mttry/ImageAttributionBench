import numpy as np  
import torch  
from torch.utils.data import DataLoader  
import random  

def set_seed(seed):  
    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed)  
    if torch.cuda.is_available():  
        torch.cuda.manual_seed(seed)  
        torch.cuda.manual_seed_all(seed)  
        torch.backends.cudnn.deterministic = True  
        torch.backends.cudnn.benchmark = False  

def get_all_indices_from_loader(dataloader):  
    indices = []  
    subset = dataloader.dataset  
    for batch in dataloader:  
        pass  

    return list(dataloader.dataset.indices)  

def compare_dataloaders(loader1, loader2):  
    idx1 = list(loader1.dataset.indices)  
    idx2 = list(loader2.dataset.indices)  
    print("Subset indices equal:", idx1 == idx2)  

    batches1 = []  
    batches2 = []  
    for b1, b2 in zip(loader1, loader2):  
        print(b1['label'])
        print(b2['label'])
        label1 = b1['label'] 
        label2 = b2['label']
        batches1.append(label1.cpu().numpy())  
        batches2.append(label2.cpu().numpy())  
    
    all_equal = all(np.array_equal(b1, b2) for b1, b2 in zip(batches1, batches2))  
    print("All batches equal:", all_equal)  

if __name__ == "__main__":  

    root_dir = "/home/final_dataset"  
    from dataloader import get_dataloader  

    train_loader_1, val_loader_1, test_loader_1 = get_dataloader(  
        root_dir=root_dir,  
        model_name="resnet50",  
        use_semantic_split=False,  
        batch_size=32,  
        num_workers=0,  
        seed=42,  
        num_images_per_semantic_per_class=20
    )  
    train_loader_2, val_loader_2, test_loader_2 = get_dataloader(  
        root_dir=root_dir,  
        model_name="resnet50",  
        use_semantic_split=False,  
        batch_size=32,  
        num_workers=0,  
        seed=42,  
        num_images_per_semantic_per_class=20
    )  

    print("Compare train loaders:")  
    compare_dataloaders(train_loader_1, train_loader_2)  
    print("Compare val loaders:")  
    compare_dataloaders(val_loader_1, val_loader_2)  
    print("Compare test loaders:")  
    compare_dataloaders(test_loader_1, test_loader_2)  