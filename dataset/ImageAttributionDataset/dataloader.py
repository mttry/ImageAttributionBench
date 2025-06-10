import numpy as np  
import torch  
from torch.utils.data import DataLoader, Subset  
from collections import defaultdict  
import sys 
import os 
current_file_path = os.path.abspath(__file__)  
parent_dir = os.path.dirname(current_file_path)          # ImageAttributionDataset 
dataset_root_dir = os.path.dirname(parent_dir)           # dataset
project_root_dir = os.path.dirname(dataset_root_dir)     # project-root
sys.path.append(parent_dir)
sys.path.append(dataset_root_dir)
sys.path.append(project_root_dir)

from ImageAttributionDataset import DATASET  
from torchvision import transforms  
import random
import copy  


def set_seed(seed):  
    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed)  
    if torch.cuda.is_available():  
        torch.cuda.manual_seed(seed)  
        torch.cuda.manual_seed_all(seed)  
        torch.backends.cudnn.deterministic = True  
        torch.backends.cudnn.benchmark = False  



def stratified_split_by_label(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):  
    # np.random.seed(seed)  
    set_seed(seed)
    label_to_indices = defaultdict(list)  
    for idx, (_, label, _, _) in enumerate(dataset.samples):  
        label_to_indices[label].append(idx)  
    
    train_indices = []  
    val_indices = []  
    test_indices = []  
    
    for label, indices in label_to_indices.items():  
        indices = np.array(indices)  
        np.random.shuffle(indices)  
        
        n_total = len(indices)  
        n_train = int(n_total * train_ratio)  
        n_val = int(n_total * val_ratio)  
        n_test = n_total - n_train - n_val  
        
        train_indices.extend(indices[:n_train])  
        val_indices.extend(indices[n_train:n_train+n_val])  
        test_indices.extend(indices[n_train+n_val:])  
    
    np.random.shuffle(train_indices)  
    np.random.shuffle(val_indices)  
    np.random.shuffle(test_indices)  

    train_dataset = copy.deepcopy(dataset) 
    val_dataset = copy.deepcopy(dataset)  
    test_dataset = copy.deepcopy(dataset)

    train_indices = train_indices[:len(train_indices) - len(train_indices) % 2]
    train_dataset.set_train()  
    train_dataset = Subset(train_dataset, train_indices)  

    val_dataset.set_val()  
    val_dataset = Subset(val_dataset, val_indices)  

    test_dataset.set_test()  
    test_dataset = Subset(test_dataset, test_indices)   
    
    return train_dataset, val_dataset, test_dataset  

import numpy as np  
from collections import defaultdict  
from torch.utils.data import Subset  

def split_dataset_trainval_by_label_semantics(dataset,   
                                              train_semantics: set,   
                                              test_semantics: set,  
                                              train_ratio=0.9, val_ratio=0.1,  
                                              seed=42):  

    # np.random.seed(seed)  
    set_seed(seed)
    label_to_indices = defaultdict(list)  
    for idx, (_, label, _, semantic_subclass) in enumerate(dataset.samples):  
        label_to_indices[label].append((idx, semantic_subclass))  

    train_indices = []  
    val_indices = []  
    test_indices = []  

    for label, idx_semantics in label_to_indices.items():  
        semantic_to_indices = defaultdict(list)  
        for idx, sem in idx_semantics:  
            semantic_to_indices[sem].append(idx)  

        train_label_indices = []  
        val_label_indices = []  
        test_label_indices = []  

        for sem, sem_indices in semantic_to_indices.items():  
            sem_indices = np.array(sem_indices)  
            np.random.shuffle(sem_indices)  

            if sem in test_semantics:  
                test_label_indices.extend(sem_indices.tolist())  
            elif sem in train_semantics:  
                n_total = len(sem_indices)  
                n_train = int(n_total * train_ratio)  
                n_val = n_total - n_train  

                train_label_indices.extend(sem_indices[:n_train].tolist())  
                val_label_indices.extend(sem_indices[n_train:].tolist())  
            else:  
                pass  

        train_indices.extend(train_label_indices)  
        val_indices.extend(val_label_indices)  
        test_indices.extend(test_label_indices)  

    np.random.shuffle(train_indices)  
    np.random.shuffle(val_indices)  
    np.random.shuffle(test_indices)  

    from torch.utils.data import Subset  
    
    train_indices = train_indices[:len(train_indices) - len(train_indices) % 2]
    train_dataset = copy.deepcopy(dataset) 
    val_dataset = copy.deepcopy(dataset)  
    test_dataset = copy.deepcopy(dataset)

    train_dataset.set_train()  
    train_dataset = Subset(train_dataset, train_indices)  
    val_dataset.set_val()  
    val_dataset = Subset(val_dataset, val_indices)  

    test_dataset.set_test()  
    test_dataset = Subset(test_dataset, test_indices)  

    return train_dataset, val_dataset, test_dataset  



from torch.utils.data import DataLoader  

from torch.utils.data import DataLoader  



def get_dataloader(root_dir,  
                   model_name,
                   num_images_per_semantic_per_class=2000,
                   degraded = 0,
                   config = None,  
                   batch_size=64,  
                   num_workers=4,  
                   train_ratio=0.8,  
                   val_ratio=0.1,  
                   test_ratio=0.1,  
                   seed=42,  
                   transform=None,  
                   use_semantic_split=False,  
                   train_semantics=None,  
                   test_semantics=None):  


    dataset_class = DATASET[model_name]
    dataset = dataset_class(root_dir=root_dir,  
                                      num_images_per_semantic_per_class=num_images_per_semantic_per_class,  
                                      transform=transform, degraded=degraded, config=config)  

    def worker_init_fn(worker_id):  
        worker_seed = seed + worker_id  
        np.random.seed(worker_seed)  
        random.seed(worker_seed)  
    if use_semantic_split:  
        if train_semantics is None:  
            train_semantics = {"cat", "dog", "wild", "COCO", "FFHQ", "celebahq", "ImageNet-1k"}  
        if test_semantics is None:  
            test_semantics = {"bedroom", "church", "classroom"}  

        train_ds, val_ds, test_ds = split_dataset_trainval_by_label_semantics(  
            dataset,  
            train_semantics=train_semantics,  
            test_semantics=test_semantics,  
            train_ratio=train_ratio,  
            val_ratio=val_ratio,  
            seed=seed  
        )  
    else:  
        train_ds, val_ds, test_ds = stratified_split_by_label(  
            dataset,  
            train_ratio=train_ratio,  
            val_ratio=val_ratio,  
            test_ratio=test_ratio,  
            seed=seed  
        )  
    g = torch.Generator()  
    g.manual_seed(seed) 

    def create_dataloader(dataset, batch_size, shuffle, num_workers, **extra_kwargs):  
        collate_fn = getattr(dataset.dataset, 'collate_fn', None)  
        kwargs = {  
            'batch_size': batch_size,  
            'shuffle': shuffle,  
            'num_workers': num_workers,  
        }  
        if collate_fn is not None:  
            kwargs['collate_fn'] = collate_fn  
        kwargs.update(extra_kwargs)  
        return DataLoader(dataset, **kwargs)  

    train_loader = create_dataloader(train_ds, batch_size, True, num_workers, generator=g, worker_init_fn=worker_init_fn)  
    val_loader = create_dataloader(val_ds, batch_size, False, num_workers)  
    test_loader = create_dataloader(test_ds, batch_size, False, num_workers) 

    return train_loader, val_loader, test_loader   
if __name__ == "__main__":
    root_dir = "/home/final_dataset"
    # normal split
    # print("Testing split normally...")
    # train_loader, val_loader, test_loader = get_dataloader(  
    #     root_dir=root_dir,  
    #     model_name="resnet50",
    #     use_semantic_split=False,  
    #     batch_size=32 ,
    # )  
    # print("train:", len(train_loader))
    # print("val:", len(val_loader))
    # print("test:", len(test_loader))

    # # get one batch to test
    # print("------------train----------")
    # batch = next(iter(train_loader))  
    # print("------------val----------")
    # batch = next(iter(val_loader)) 
    # print("------------test----------")
    # batch = next(iter(test_loader)) 

    # split by semantic
    print("Testing split by semantic...")
    train_semantics = {"cat", "dog", "wild"}  
    test_semantics = {"bedroom", "church"}  

    train_loader, val_loader, test_loader = get_dataloader(  
        root_dir=root_dir,  
        model_name="resnet50",
        use_semantic_split=True,  
        train_semantics={"bedroom"},  
        test_semantics={"church"},  
        batch_size=1  
    )  
    print("train:", len(train_loader))
    print("val:", len(val_loader))
    print("test:", len(test_loader))

    # # get one batch to test
    # print("------------train----------")
    # batch = next(iter(train_loader))  
    # print("------------val----------")
    # batch = next(iter(val_loader)) 
    # print("------------test----------")
    # batch = next(iter(test_loader)) 