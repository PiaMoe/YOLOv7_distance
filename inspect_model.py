import torch
from typing import List


def compare_models(pt_paths: List[str], atol=0.0, max_print=5):
    """
    Vergleicht mehrere PyTorch-Modelle hinsichtlich Architektur und Gewichten.

    Args:
        pt_paths (List[str]): Liste von Pfaden zu .pt-Dateien
        atol (float): Absoluter Toleranzwert beim Vergleich der Gewichte (für float-Vergleich mit allclose)
        max_print (int): Wie viele Gewichtsunterschiede/-gleichheiten sollen maximal angezeigt werden
    """
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

    for idx, model in enumerate(models):
        print(f"\nModell {idx}:")
        for name, module in model.named_modules():
            if "105" in name:
                #print(f"Gefundenes Modul ({name}): {module}")
                if hasattr(module, 'anchors'):
                    print("anchors:\n", module.anchors)
                if hasattr(module, 'anchor_grid'):
                    print("anchor_grid:\n", module.anchor_grid)

    print(f"\n=== Vergleich von {len(models)} Modellen (trainierbare & nicht-trainierbare Parameter) ===")

    for idx, model in enumerate(models[1:], 1):
        print(f"\n--- Vergleich: Modell 0 <-> Modell {idx} ---")
        other_state = model.state_dict()
        other_trainable_keys = {k for k, v in model.named_parameters()}
        all_other_keys = set(other_state.keys())
        non_trainable_other_keys = all_other_keys - other_trainable_keys

        # === Trainierbare Parameter vergleichen ===
        print("\n📦 Trainierbare Parameter:")
        shared_keys = base_trainable_keys & other_trainable_keys
        only_in_0 = base_trainable_keys - other_trainable_keys
        only_in_1 = other_trainable_keys - base_trainable_keys

        if only_in_0:
            print(f"  Nur in Modell 0:\n    {sorted(list(only_in_0))}")
        if only_in_1:
            print(f"  Nur in Modell {idx}:\n    {sorted(list(only_in_1))}")
        if not only_in_0 and not only_in_1:
            print("  ✔ Gleiche trainierbare Architektur")

        diff_keys = []
        same_keys = []

        for key in shared_keys:
            a = base_state[key]
            b = other_state[key]
            if torch.allclose(a.float(), b.float(), atol=atol):
                same_keys.append(key)
            else:
                diff_keys.append(key)

        print(f"  🔁 {len(diff_keys)} Parameter unterschiedlich (trainierbar)")
        for k in diff_keys[:max_print]:
            print(f"    ✗ {k}")
        if len(diff_keys) > max_print:
            print(f"    ... {len(diff_keys) - max_print} weitere Unterschiede")

        print(f"  ✅ {len(same_keys)} Parameter gleich (trainierbar)")
        for k in same_keys[:max_print]:
            print(f"    ✓ {k}")
        if len(same_keys) > max_print:
            print(f"    ... {len(same_keys) - max_print} weitere Gleichheiten")

        # === Nicht-trainierbare Parameter vergleichen ===
        print("\n🧊 Nicht-trainierbare Parameter:")
        shared_nontrain_keys = non_trainable_base_keys & non_trainable_other_keys
        only_in_0_nt = non_trainable_base_keys - non_trainable_other_keys
        only_in_1_nt = non_trainable_other_keys - non_trainable_base_keys

        if only_in_0_nt:
            print(f"  Nur in Modell 0:\n    {sorted(list(only_in_0_nt))}")
        if only_in_1_nt:
            print(f"  Nur in Modell {idx}:\n    {sorted(list(only_in_1_nt))}")
        if not only_in_0_nt and not only_in_1_nt:
            print("  ✔ Gleiche nicht-trainierbare Architektur")

        diff_keys_nt = []
        same_keys_nt = []

        for key in shared_nontrain_keys:
            a = base_state[key]
            b = other_state[key]
            if torch.allclose(a.float(), b.float(), atol=atol):
                same_keys_nt.append(key)
            else:
                diff_keys_nt.append(key)

        print(f"  🔁 {len(diff_keys_nt)} Parameter unterschiedlich (nicht-trainierbar)")
        for k in diff_keys_nt[:max_print]:
            print(f"    ✗ {k}")
        if len(diff_keys_nt) > max_print:
            print(f"    ... {len(diff_keys_nt) - max_print} weitere Unterschiede")

        print(f"  ✅ {len(same_keys_nt)} Parameter gleich (nicht-trainierbar)")
        for k in same_keys_nt[:max_print]:
            print(f"    ✓ {k}")
        if len(same_keys_nt) > max_print:
            print(f"    ... {len(same_keys_nt) - max_print} weitere Gleichheiten")


if __name__ == "__main__":
    pt_files = [
        "weights/bestDet.pt",
        "weights/freezeMultHeadsallInit.pt"
    ]
    compare_models(pt_files, max_print=15)
