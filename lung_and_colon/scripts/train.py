import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
import os
import json
import argparse # For command-line arguments

from model import get_model, get_vit_model 
from dataset import get_dataloaders

# --- Configuration ---
BATCH_SIZE = 64
NUM_EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 1e-4

def train_model(args):
    """Main function to train the model, accepts command-line arguments."""

    # --- 1. Setup ---
    print("--- 1. Setting up ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create descriptive names for saving files based on arguments
    model_name = f"{args.model_type}{'_stain_norm' if args.use_stain_norm else ''}"
    MODEL_SAVE_PATH = f'results/saved_models/best_model_{model_name}.pth'
    HISTORY_SAVE_PATH = f'results/training_history_{model_name}.json'
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    print(f"Model will be saved to: {MODEL_SAVE_PATH}")

    # Get DataLoaders, passing the stain norm flag
    train_loader, val_loader, _, class_names = get_dataloaders(
        batch_size=BATCH_SIZE, use_stain_norm=args.use_stain_norm
    )
    num_classes = len(class_names)

    # Select the model based on the command-line argument
    if args.model_type == 'resnet':
        model = get_model(num_classes=num_classes)
        print("Using ResNet50 model.")
    elif args.model_type == 'vit':
        model = get_vit_model(num_classes=num_classes)
        print("Using Vision Transformer model.")
    else:
        raise ValueError("Invalid model type specified. Choose 'resnet' or 'vit'.")
        
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    # Fine-tune only the final layer
    if args.model_type == 'resnet':
        optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
    else: # For ViT
        optimizer = optim.Adam(model.heads.head.parameters(), lr=LEARNING_RATE)

    # Determine number of epochs (if args.epochs is None, fall back to default)
    num_epochs = args.epochs if getattr(args, 'epochs', None) is not None else NUM_EPOCHS

    # --- 2. Training Loop ---
    print("\n--- 2. Starting Training ---")
    best_val_accuracy = 0.0
    training_history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training Phase
        model.train()
        running_train_loss = 0.0
        correct_train_predictions = 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train_predictions += torch.sum(preds == labels.data)
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        epoch_train_acc = correct_train_predictions.double() / len(train_loader.dataset)

        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        correct_val_predictions = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val_predictions += torch.sum(preds == labels.data)
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        epoch_val_acc = correct_val_predictions.double() / len(val_loader.dataset)

        # Log Epoch Results & Save Best Model
        end_time = time.time()
        epoch_duration = end_time - start_time
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Time: {epoch_duration:.2f}s | "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}")
        
        training_history['train_loss'].append(epoch_train_loss)
        training_history['train_acc'].append(epoch_train_acc.item())
        training_history['val_loss'].append(epoch_val_loss)
        training_history['val_acc'].append(epoch_val_acc.item())

        if epoch_val_acc > best_val_accuracy:
            best_val_accuracy = epoch_val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"✅ New best model saved to {MODEL_SAVE_PATH} (Val Acc: {best_val_accuracy:.4f})")

    with open(HISTORY_SAVE_PATH, 'w') as f:
        json.dump(training_history, f)
    print(f"\nTraining history saved to {HISTORY_SAVE_PATH}")
    print("--- Training Finished ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a model for histopathology classification.")
    parser.add_argument('--model_type', type=str, required=True, choices=['resnet', 'vit'], 
                        help="Type of model to train ('resnet' or 'vit').")
    parser.add_argument('--use_stain_norm', action='store_true',
                        help="Enable stain normalization.")
    parser.add_argument('--epochs', type=int, default=None,
                        help="Optional: number of epochs to run (overrides default).")
    args = parser.parse_args()
    train_model(args)