import numpy as np  
import torch  
from torch.utils.data import DataLoader  
import random  

# 复用你给出的 set_seed 函数  
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
    # Subset封装的dataset中，batch是索引映射到原dataset的，所以先拿到原索引：  
    # 假设 dataloader.dataset 是 Subset  
    subset = dataloader.dataset  
    # dataloader.batch_sampler 或 sampler 的索引即是子集索引  
    for batch in dataloader:  
        # batch是数据本身，这里print idx需自行修改为能追踪到原idx的代码  
        # 这里用 Subset 的 dataset.indices 保存所有样本索引列表  
        # 但dataloader shuffle 改变了顺序，只能通过其它方式确认顺序是否一致  
        # 更简单的做法是直接拿索引列表  
        pass  

    # 返回所有索引序列（元素顺序即取样顺序）  
    # 对于不shuffle的loader，顺序就是dataset.indices，shuffle的需要额外测试  
    return list(dataloader.dataset.indices)  

def compare_dataloaders(loader1, loader2):  
    idx1 = list(loader1.dataset.indices)  
    idx2 = list(loader2.dataset.indices)  
    print("Subset indices equal:", idx1 == idx2)  

    # 进一步比较前N个batch数据是否一致（你需要根据数据结构改写）  
    batches1 = []  
    batches2 = []  
    for b1, b2 in zip(loader1, loader2):  
        # 假设数据结构是一tuple (image, label, ...)  
        # 取label对比  
        print(b1['label'])
        print(b2['label'])
        label1 = b1['label'] 
        label2 = b2['label']
        batches1.append(label1.cpu().numpy())  
        batches2.append(label2.cpu().numpy())  
    
    all_equal = all(np.array_equal(b1, b2) for b1, b2 in zip(batches1, batches2))  
    print("All batches equal:", all_equal)  

if __name__ == "__main__":  
    # 固定随机种子确保划分可复现  
    # set_seed(42)  

    root_dir = "/home/final_dataset"  
    from dataloader import get_dataloader # 请替换为你的脚本模块名  

    # 第一次加载  
    train_loader_1, val_loader_1, test_loader_1 = get_dataloader(  
        root_dir=root_dir,  
        model_name="resnet50",  
        use_semantic_split=False,  
        batch_size=32,  
        num_workers=0,  
        seed=42,  
        num_images_per_semantic_per_class=20
    )  
    # 第二次加载  
    # set_seed(42)  
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