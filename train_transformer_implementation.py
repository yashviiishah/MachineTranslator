import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformer_implementation import Transformer
from tokenizer import create_tokenizers
from data_loader import TranslationDataset
from tqdm import tqdm
import os
import pandas as pd

def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_tokens = 0
    
    for batch in tqdm(train_loader, desc="Training"):
        src = batch['source'].to(device)
        trg = batch['target'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(src, trg[:, :-1])
        
        # Calculate loss
        output = output.reshape(-1, output.shape[-1])
        trg = trg[:, 1:].reshape(-1)
        
        # Calculate loss only for non-padding tokens
        non_pad_mask = trg != 0
        if non_pad_mask.sum().item() == 0:
            continue
            
        loss = criterion(output[non_pad_mask], trg[non_pad_mask])
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Accumulate loss and tokens
        total_loss += loss.item() * non_pad_mask.sum().item()
        total_tokens += non_pad_mask.sum().item()
        
        # Print batch statistics
        print(f"Batch Loss: {loss.item():.4f}, Tokens: {non_pad_mask.sum().item()}")
    
    return total_loss / total_tokens if total_tokens > 0 else float('inf')

def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            src = batch['source'].to(device)
            trg = batch['target'].to(device)
            
            output = model(src, trg[:, :-1])
            
            output = output.reshape(-1, output.shape[-1])
            trg = trg[:, 1:].reshape(-1)
            
            non_pad_mask = trg != 0
            loss = criterion(output[non_pad_mask], trg[non_pad_mask])
            
            total_loss += loss.item() * non_pad_mask.sum().item()
            total_tokens += non_pad_mask.sum().item()
    
    return total_loss / total_tokens if total_tokens > 0 else float('inf')

def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs, device):
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        
        # Evaluate
        val_loss = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, 'checkpoints/best_model.pth')
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        
        # Early stopping
        if epoch > 5 and min(val_losses[-5:]) > best_val_loss:
            print("Early stopping triggered")
            break
    
    return train_losses, val_losses

if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Hyperparameters
    EMBED_SIZE = 512
    NUM_LAYERS = 6
    HEADS = 8
    FORWARD_EXPANSION = 4
    DROPOUT = 0.1
    MAX_LENGTH = 100
    BATCH_SIZE = 8
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.0001
    
    try:
        # Create tokenizers
        print("Creating tokenizers...")
        source_tokenizer, target_tokenizer = create_tokenizers('english_assamese.csv')
        
        # Create datasets
        print("Creating datasets...")
        train_dataset = TranslationDataset(
            'english_assamese.csv',
            source_tokenizer,
            target_tokenizer,
            MAX_LENGTH,
            split='train'
        )
        
        val_dataset = TranslationDataset(
            'english_assamese.csv',
            source_tokenizer,
            target_tokenizer,
            MAX_LENGTH,
            split='val'
        )
        
        # Create data loaders
        print("Creating data loaders...")
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
        
        # Initialize model
        print("Initializing model...")
        model = Transformer(
            src_vocab_size=source_tokenizer.get_vocab_size(),
            trg_vocab_size=target_tokenizer.get_vocab_size(),
            src_pad_idx=source_tokenizer.get_pad_idx(),
            trg_pad_idx=target_tokenizer.get_pad_idx(),
            embed_size=EMBED_SIZE,
            num_layers=NUM_LAYERS,
            forward_expansion=FORWARD_EXPANSION,
            heads=HEADS,
            dropout=DROPOUT,
            device=device,
            max_length=MAX_LENGTH
        ).to(device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss(ignore_index=source_tokenizer.get_pad_idx())
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        # Create checkpoints directory
        os.makedirs('checkpoints', exist_ok=True)
        
        # Train model
        print("Starting training...")
        train_losses, val_losses = train_model(
            model, train_loader, val_loader, optimizer, criterion, NUM_EPOCHS, device
        )
        
        print("Training completed!")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise