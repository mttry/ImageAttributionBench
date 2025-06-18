import torch

def inspect_ckpt(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location='cpu')['model_state_dict']

    def print_structure(obj, indent=0):
        prefix = '  ' * indent
        if isinstance(obj, dict):
            print(f"{prefix}dict with keys:")
            for k, v in obj.items():
                print(f"{prefix}  - {k}:")
                # print_structure(v, indent+2)
        elif isinstance(obj, list):
            print(f"{prefix}list of length {len(obj)}")
            for i, v in enumerate(obj):
                print(f"{prefix}  [{i}]:")
                # print_structure(v, indent+2)
        elif torch.is_tensor(obj):
            print(f"{prefix}Tensor shape {tuple(obj.shape)} dtype {obj.dtype}")
        else:
            print(f"{prefix}{type(obj)}: {str(obj)[:100]}")  # 简短打印

    print(f"Checkpoint loaded from: {ckpt_path}")
    print_structure(checkpoint)

if __name__ == "__main__":
    ckpt_path = "/home/ImageAttributionBench/training/logs_try/default_split/dna/_2025-05-02-11-36-03/ckpt_best.pth"
    inspect_ckpt(ckpt_path)