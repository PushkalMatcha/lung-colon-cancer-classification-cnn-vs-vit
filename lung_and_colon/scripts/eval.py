import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import argparse
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm

from model import get_model, get_vit_model
from dataset import get_dataloaders

# --- Configuration ---
BATCH_SIZE = 32
PLOTS_DIR = 'results/plots'

def evaluate_model(args):
    """
    Evaluates the trained model on the test set and generates visualizations.
    """
    # Create plots directory if it doesn't exist
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # --- 1. Setup ---
    print("--- 1. Setting up ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create descriptive names to find the correct files
    model_name = f"{args.model_type}{'_stain_norm' if args.use_stain_norm else ''}"
    MODEL_PATH = f'results/saved_models/best_model_{model_name}.pth'
    # If a checkpoint path was supplied explicitly, use that instead
    if getattr(args, 'checkpoint', None):
        MODEL_PATH = args.checkpoint
    HISTORY_PATH = f'results/training_history_{model_name}.json'

    # Check if the required model file exists
    if not os.path.exists(MODEL_PATH):
        print(f"FATAL ERROR: Model file not found at '{MODEL_PATH}'")
        print("Please train the model first by running the train.py script with the correct arguments.")
        return

    # Guard against zero-length / corrupted checkpoint files which cause EOFError on load
    try:
        file_size = os.path.getsize(MODEL_PATH)
    except OSError:
        file_size = 0

    if file_size == 0:
        print(f"FATAL ERROR: Model file at '{MODEL_PATH}' is empty (0 bytes). It may be corrupted or was not saved correctly.")
        # Show other candidate checkpoints in the same directory (if any)
        saved_dir = os.path.dirname(MODEL_PATH)
        try:
            candidates = [f for f in os.listdir(saved_dir) if os.path.isfile(os.path.join(saved_dir, f))]
        except Exception:
            candidates = []

        if candidates:
            print("Available files in the checkpoint directory:")
            for c in candidates:
                try:
                    s = os.path.getsize(os.path.join(saved_dir, c))
                except Exception:
                    s = 0
                print(f" - {c} ({s} bytes)")

        print("Suggested fixes:")
        print("  * If you have a valid checkpoint (e.g. 'best_model.pth'), copy it to the expected path or run eval against that file.")
        print("    PowerShell example: Copy-Item .\\results\\saved_models\\best_model.pth .\\results\\saved_models\\best_model_resnet.pth -Force")
        print("  * Or retrain the model: run scripts/train.py with the same arguments to produce a valid checkpoint.")
        return

    # Get the test data loader and class names
    # Pass use_stain_norm to ensure test data is processed the same way as training
    _, _, test_loader, class_names = get_dataloaders(
        batch_size=BATCH_SIZE, use_stain_norm=args.use_stain_norm
    )
    num_classes = len(class_names)

    # --- Determine which model architecture to build by inspecting the checkpoint ---
    # Load the checkpoint first so we can examine its keys without yet binding
    # to a particular architecture. This allows using --checkpoint even if
    # --model_type doesn't match the checkpoint (we'll warn then proceed).
    raw_ckpt = torch.load(MODEL_PATH, map_location='cpu')
    if isinstance(raw_ckpt, dict) and 'state_dict' in raw_ckpt and isinstance(raw_ckpt['state_dict'], dict):
        state_dict = raw_ckpt['state_dict']
    elif isinstance(raw_ckpt, dict):
        state_dict = raw_ckpt
    else:
        raise RuntimeError(f"Unsupported checkpoint format: {type(raw_ckpt)}")

    keys = list(state_dict.keys())

    # Heuristic detection
    looks_like_resnet = any(k.startswith('conv1') or k.startswith('layer1.') or k.startswith('bn1') for k in keys)
    looks_like_vit = any('class_token' in k or k.startswith('encoder.') or 'conv_proj' in k or k.startswith('heads.') for k in keys)

    chosen_arch = None
    if looks_like_vit and not looks_like_resnet:
        chosen_arch = 'vit'
    elif looks_like_resnet and not looks_like_vit:
        chosen_arch = 'resnet'
    elif looks_like_resnet and looks_like_vit:
        # ambiguous — prefer explicit arg if it matches
        chosen_arch = args.model_type
    else:
        # fallback to the user-provided model_type
        chosen_arch = args.model_type

    if chosen_arch != args.model_type:
        print(f"Warning: checkpoint appears to be '{chosen_arch}' but --model_type='{args.model_type}' was requested. Overriding to '{chosen_arch}'.")

    # Instantiate appropriate model and load weights
    if chosen_arch == 'resnet':
        model = get_model(num_classes=num_classes, pretrained=False)
    elif chosen_arch == 'vit':
        model = get_vit_model(num_classes=num_classes, pretrained=False)
    else:
        raise ValueError("Could not determine model architecture from checkpoint or args.")

    # Move model to device then load weights
    model.to(device)
    # Load state dict onto device
    # If we loaded the full checkpoint wrapper earlier, reuse it; otherwise
    # move tensors to device during load_state_dict.
    try:
        # move checkpoint tensors to device to avoid dtype/device mismatches
        state_dict_on_device = {k: v.to(device) if hasattr(v, 'to') else v for k, v in state_dict.items()}
        model.load_state_dict(state_dict_on_device)
    except Exception:
        # fallback: attempt without moving (torch will handle map_location)
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval() # Set model to evaluation mode

    # --- 2. Run Inference on Test Set ---
    print(f"\n--- 2. Evaluating model '{model_name}' on Test Set ---")
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # --- 3. Calculate and Print Metrics ---
    print("\n--- 3. Performance Metrics ---")
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Overall Test Accuracy: {accuracy:.4f}")

    # Classification report as a structured dict
    report_dict = classification_report(all_labels, all_preds, target_names=class_names, digits=4, output_dict=True)
    # Print the human-readable report as before
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    # --- 4. Generate and Save Plots ---
    print("\n--- 4. Generating Plots ---")
    
    # Plot 1: Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix for {model_name}')
    cm_path = os.path.join(PLOTS_DIR, f'confusion_matrix_{model_name}.png')
    plt.savefig(cm_path)
    print(f"Confusion matrix saved to {cm_path}")
    
    # --- 5. Save metrics to single combined JSON file ---
    ALL_HISTORY_PATH = os.path.join('results', 'all_models_history.json')
    os.makedirs(os.path.dirname(ALL_HISTORY_PATH), exist_ok=True)

    # Prepare payload for this model
    model_entry = {
        model_name: {
            "overall_accuracy": float(round(accuracy, 4)),
            "classification_report": report_dict,
            "confusion_matrix": cm.tolist()
        }
    }

    # Load existing all-models history if present, update and write back
    if os.path.exists(ALL_HISTORY_PATH):
        try:
            with open(ALL_HISTORY_PATH, 'r') as f:
                all_history = json.load(f)
        except Exception:
            all_history = {}
    else:
        all_history = {}

    # Update (overwrite) this model's entry
    all_history.update(model_entry)

    with open(ALL_HISTORY_PATH, 'w') as f:
        json.dump(all_history, f, indent=4)

    print(f"Model metrics saved/updated in {ALL_HISTORY_PATH}")
    plt.close()

    # Plot 2: Training & Validation Curves
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, 'r') as f:
            history = json.load(f)
            
        plt.figure(figsize=(12, 5))
        
        # Loss curves
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Loss Curves for {model_name}')
        plt.legend()
        
        # Accuracy curves
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Train Accuracy')
        plt.plot(history['val_acc'], label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title(f'Accuracy Curves for {model_name}')
        plt.legend()
        
        plt.tight_layout()
        curves_path = os.path.join(PLOTS_DIR, f'training_curves_{model_name}.png')
        plt.savefig(curves_path)
        print(f"Training curves saved to {curves_path}")
        plt.close()
    else:
        print(f"Warning: Training history not found at '{HISTORY_PATH}'. Skipping curve plots.")

    print("\n--- Evaluation Finished ---")

if __name__ == '__main__':
    # Set up argument parser to match train.py
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument('--model_type', type=str, required=True, choices=['resnet', 'vit'], 
                        help="Type of model to evaluate ('resnet' or 'vit').")
    parser.add_argument('--use_stain_norm', action='store_true',
                        help="Specify if the model was trained with stain normalization.")
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Optional: path to a specific checkpoint file to evaluate (overrides default MODEL_PATH).')
    
    args = parser.parse_args()
    evaluate_model(args)