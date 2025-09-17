import torch
from typing import List
import os


def compare_models(pt_paths: List[str], atol=0.0, max_print=5):
    #Compares the weights of multiple PyTorch models stored in .pt files.

    models = []
    for path in pt_paths:
        model = torch.load(path, map_location='cpu')
        if isinstance(model, dict) and 'model' in model:
            model = model['model']
        if hasattr(model, 'eval'):
            model.eval()
        models.append(model)

    base_model = models[0]
    base_state = base_model.state_dict()
    base_trainable_keys = {k for k, v in base_model.named_parameters()}
    all_base_keys = set(base_state.keys())
    non_trainable_base_keys = all_base_keys - base_trainable_keys


    print(f"\n=== Comparing {len(models)} models ===")

    for idx, model in enumerate(models[1:], 1):
        print(f"\n--- model 0 vs. model {idx} ---")
        other_state = model.state_dict()
        other_trainable_keys = {k for k, v in model.named_parameters()}
        all_other_keys = set(other_state.keys())
        non_trainable_other_keys = all_other_keys - other_trainable_keys

        print("\ntrainable parameters:")
        shared_keys = base_trainable_keys & other_trainable_keys
        only_in_0 = base_trainable_keys - other_trainable_keys
        only_in_1 = other_trainable_keys - base_trainable_keys

        if only_in_0:
            print(f"  only in model 0:\n    {sorted(list(only_in_0))}")
        if only_in_1:
            print(f"  only in model {idx}:\n    {sorted(list(only_in_1))}")
        if not only_in_0 and not only_in_1:
            print("Same trainable architecture")

        diff_keys = []
        same_keys = []

        for key in shared_keys:
            a = base_state[key]
            b = other_state[key]
            if torch.allclose(a.float(), b.float(), atol=atol):
                same_keys.append(key)
            else:
                diff_keys.append(key)

        print(f"{len(diff_keys)} parameters different (trainable)")
        for k in diff_keys[:max_print]:
            print(f"    ✗ {k}")
        if len(diff_keys) > max_print:
            print(f"    ... {len(diff_keys) - max_print} more differences")

        print(f"{len(same_keys)} parameters equal (trainable)")
        for k in same_keys[:max_print]:
            print(f"{k}")
        if len(same_keys) > max_print:
            print(f"... {len(same_keys) - max_print} more equalities")

        # === compare not trainable parameters ===
        print("\nnot trainable parameters:")
        shared_nontrain_keys = non_trainable_base_keys & non_trainable_other_keys
        only_in_0_nt = non_trainable_base_keys - non_trainable_other_keys
        only_in_1_nt = non_trainable_other_keys - non_trainable_base_keys

        if only_in_0_nt:
            print(f"only in model 0:\n    {sorted(list(only_in_0_nt))}")
        if only_in_1_nt:
            print(f"only in model {idx}:\n    {sorted(list(only_in_1_nt))}")
        if not only_in_0_nt and not only_in_1_nt:
            print("same non-trainable architecture")

        diff_keys_nt = []
        same_keys_nt = []

        for key in shared_nontrain_keys:
            a = base_state[key]
            b = other_state[key]
            if torch.allclose(a.float(), b.float(), atol=atol):
                same_keys_nt.append(key)
            else:
                diff_keys_nt.append(key)

        print(f"{len(diff_keys_nt)} parameters different (not-trainable)")
        for k in diff_keys_nt[:max_print]:
            print(f"{k}")
        if len(diff_keys_nt) > max_print:
            print(f"... {len(diff_keys_nt) - max_print} more differences")

        print(f"{len(same_keys_nt)} parameters equal (not-trainable)")
        for k in same_keys_nt[:max_print]:
            print(f"    ✓ {k}")
        if len(same_keys_nt) > max_print:
            print(f" ... {len(same_keys_nt) - max_print} more equalities")

import torch
from pathlib import Path

# from YOLOv7 repo
from models.experimental import attempt_load

def info_yolov7_models(weights_list):
    """
    Prints total, learnable, and non-learnable parameters for YOLOv7 weights.

    Args:
        weights_list (list[str]): List of paths to YOLOv7 .pt files
    """
    for weights_path in weights_list:
        # Load YOLOv7 model
        model = attempt_load(weights_path, map_location="cpu")
        model.eval()

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_learnable_params = total_params - learnable_params
        file_size_mb = Path(weights_path).stat().st_size / (1024**2)

        # Print info
        print(f"\nModel: {weights_path}")
        print(f"  File size: {file_size_mb:.2f} MB")
        print(f"  Total params: {total_params:,}")
        print(f"  Learnable params: {learnable_params:,}")
        print(f"  Non-learnable params: {non_learnable_params:,}")



if __name__ == "__main__":

    pt_files = [
        "../runs/train/finalDataset/B3_freeze50/weights/best.pt"
    ]
    info_yolov7_models(pt_files)
    #for res in results:
    #    print(res)

