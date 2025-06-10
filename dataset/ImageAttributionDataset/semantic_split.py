subclass_to_superclass = {  
    "cat": "AnimalFace",  
    "dog": "AnimalFace",  
    "wild": "AnimalFace",  
    "COCO": "COCO",  
    "FFHQ": "HumanFace",  
    "celebahq": "HumanFace",  
    "ImageNet-1k": "ImageNet-1k",  
    "bedroom": "Scene",  
    "church": "Scene",  
    "classroom": "Scene",   
}  

def get_semantic(task_id):  
    # 所有类别  
    all_subclasses = set(subclass_to_superclass.keys())  

    if task_id == 1:  
        train_semantics = {"cat"}  
    elif task_id == 2:  
        train_semantics = {"bedroom"}  
    elif task_id == 3:  
        train_semantics = {"FFHQ"}  
    else:  
        raise ValueError(f"Task ID {task_id} not supported.")  

    test_semantics = all_subclasses - train_semantics  
    return train_semantics, test_semantics  

# 示例用法  
if __name__ == "__main__":
    train_s, test_s = get_semantic(3)  
    print("Train semantics:", train_s)  
    print("Test semantics:", test_s)  