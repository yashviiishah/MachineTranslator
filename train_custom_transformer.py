import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
from tqdm import tqdm
from transformer_implementation import Transformer
from data_loader import TranslationDataset, load_data, create_tokenizers

def collate_fn(batch, source_tokenizer, target_tokenizer):
    # Convert text to tensors
    source_texts = [item['source'] for item in batch]
    target_texts = [item['target'] for item in batch]
    
    # Tokenize and convert to tensors
    source_tokens = [source_tokenizer.encode_as_ids(text) for text in source_texts]
    target_tokens = [target_tokenizer.encode_as_ids(text) for text in target_texts]
    
    # Pad sequences
    source_padded = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(tokens) for tokens in source_tokens],
        batch_first=True,
        padding_value=0
    )
    
    target_padded = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(tokens) for tokens in target_tokens],
        batch_first=True,
        padding_value=0
    )
    
    return source_padded, target_padded

def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_tokens = 0
    
    for batch_idx, (src, tgt) in enumerate(tqdm(train_loader, desc="Training")):
        # Move tensors to device
        src = src.to(device)
        tgt = tgt.to(device)
        
        # Create proper attention masks
        src_mask = torch.zeros((src.size(1), src.size(1)), device=device)
        tgt_mask = model.generate_square_subsequent_mask(tgt.size(1)).to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output, _, _, _ = model(src, tgt[:, :-1], src_mask, tgt_mask[:-1, :-1])
        
        # Calculate loss
        loss = criterion(output.reshape(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update statistics
        total_loss += loss.item()
        total_tokens += (tgt[:, 1:] != 0).sum().item()
        
        # Print progress
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    return total_loss / total_tokens if total_tokens > 0 else float('inf')

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch_idx, (src, tgt) in enumerate(tqdm(val_loader, desc="Validating")):
            # Move tensors to device
            src = src.to(device)
            tgt = tgt.to(device)
            
            # Create proper attention masks
            src_mask = torch.zeros((src.size(1), src.size(1)), device=device)
            tgt_mask = model.generate_square_subsequent_mask(tgt.size(1)).to(device)
            
            # Forward pass
            output, _, _, _ = model(src, tgt[:, :-1], src_mask, tgt_mask[:-1, :-1])
            
            # Calculate loss
            loss = criterion(output.reshape(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
            
            # Update statistics
            total_loss += loss.item()
            total_tokens += (tgt[:, 1:] != 0).sum().item()
    
    return total_loss / total_tokens if total_tokens > 0 else float('inf')

def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, is_best, save_dir):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss
    }
    
    # Save regular checkpoint
    torch.save(state, os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth'))
    
    # Save best model
    if is_best:
        torch.save(state, os.path.join(save_dir, 'best_model.pth'))
        print(f"Saved best model with validation loss: {val_loss:.4f}")

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Hyperparameters
    D_MODEL = 512
    NHEAD = 8
    NUM_ENCODER_LAYERS = 6
    NUM_DECODER_LAYERS = 6
    DIM_FEEDFORWARD = 2048
    DROPOUT = 0.1
    BATCH_SIZE = 8
    LEARNING_RATE = 0.0001
    NUM_EPOCHS = 10
    MAX_SEQ_LENGTH = 500
    PATIENCE = 3
    MIN_DELTA = 0.001
    
    # Load data
    print("Loading data...")
    source_texts, target_texts = load_data('english_assamese.csv')
    
    if not source_texts or not target_texts:
        print("Failed to load data. Exiting...")
        return
    
    # Create tokenizers
    print("Creating tokenizers...")
    source_tokenizer, target_tokenizer = create_tokenizers(source_texts, target_texts)
    
    if not source_tokenizer or not target_tokenizer:
        print("Failed to create tokenizers. Exiting...")
        return
    
    # Get vocabulary sizes
    INPUT_DIM = source_tokenizer.get_piece_size()
    OUTPUT_DIM = target_tokenizer.get_piece_size()
    print(f"Source vocabulary size: {INPUT_DIM}")
    print(f"Target vocabulary size: {OUTPUT_DIM}")
    
    # Split data into train and validation
    train_size = int(0.8 * len(source_texts))
    train_source = source_texts[:train_size]
    train_target = target_texts[:train_size]
    val_source = source_texts[train_size:]
    val_target = target_texts[train_size:]
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = TranslationDataset(
        train_source,
        train_target,
        source_tokenizer,
        target_tokenizer,
        max_length=MAX_SEQ_LENGTH
    )
    
    val_dataset = TranslationDataset(
        val_source,
        val_target,
        source_tokenizer,
        target_tokenizer,
        max_length=MAX_SEQ_LENGTH
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, source_tokenizer, target_tokenizer)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, source_tokenizer, target_tokenizer)
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create model
    model = Transformer(
        src_vocab_size=INPUT_DIM,
        tgt_vocab_size=OUTPUT_DIM,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        max_seq_length=MAX_SEQ_LENGTH
    ).to(device)
    
    # Create optimizer and criterion
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # Training loop with early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    save_dir = 'checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    
    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        
        # Print epoch results
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} - {epoch_time:.2f}s")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        
        # Early stopping check
        is_best = val_loss < best_val_loss - MIN_DELTA
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement in validation loss for {patience_counter} epochs")
            
            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        # Save checkpoint
        save_checkpoint(model, optimizer, epoch, train_loss, val_loss, is_best, save_dir)
    
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main() 