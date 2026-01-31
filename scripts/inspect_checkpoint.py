import sys
import torch

path = sys.argv[1]
print(f"Inspecting checkpoint: {path}")
try:
    data = torch.load(path, map_location='cpu')
except Exception as e:
    print(f"Failed to load checkpoint: {e}")
    sys.exit(2)

# If it's a state_dict (mapping of tensors)
if isinstance(data, dict):
    # Some checkpoints save {'state_dict': ..., 'epoch':...}
    if 'state_dict' in data and isinstance(data['state_dict'], dict):
        sd = data['state_dict']
        print("Found wrapper dict with 'state_dict' key; using that.")
    else:
        sd = data

    print(f"Top-level number of keys in state_dict: {len(sd)}")
    # show first 40 keys
    keys = list(sd.keys())
    for i,k in enumerate(keys[:40]):
        v = sd[k]
        try:
            shape = tuple(v.shape)
        except Exception:
            shape = type(v)
        print(f"{i+1:03d}: {k} -> {shape}")

    # Simple heuristic: does it look like a resnet (conv1, bn1, layer1...) or vit (class_token, encoder.)
    keystr = ' '.join(keys[:200])
    if 'conv1.weight' in keystr or 'layer1.0.conv1.weight' in keystr:
        print('\nHeuristic: Looks like a ResNet checkpoint (contains conv1/layer1 keys).')
    if 'class_token' in keystr or 'encoder.pos_embedding' in keystr or 'conv_proj.weight' in keystr:
        print('\nHeuristic: Looks like a ViT checkpoint (contains class_token/encoder keys).')
else:
    print('Checkpoint is not a dict; type =', type(data))
    sys.exit(3)
