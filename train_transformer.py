import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import os
import time
import math
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Add parent directory to path to import data_loader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import load_data, create_tokenizers, TranslationDataset
from transformers.transformer_model import Transformer

def train_epoch(model, train_loader, optimizer, criterion, device, clip=1.0):
    """
    Train the model for one epoch.
    
    Args:
        model: Transformer model
        train_loader: DataLoader for training data
        optimizer: Optimizer
        criterion: Loss function
        device: Device to use for computation
        clip: Gradient clipping value
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0
    
    for batch in tqdm(train_loader, desc="Training"):
        src = batch['source'].to(device)  # (batch_size, seq_len)
        tgt = batch['target'].to(device)  # (batch_size, seq_len)
        
        # Create masks
        src_padding_mask = (src == 0).to(device)  # (batch_size, src_len)
        tgt_padding_mask = (tgt == 0).to(device)  # (batch_size, tgt_len)
        
        # Create target mask to prevent attention to future tokens
        tgt_len = tgt.size(1)
        tgt_mask = model.generate_square_subsequent_mask(tgt_len).to(device)
        
        # Forward pass
        optimizer.zero_grad()
        
        # Use teacher forcing: target input is shifted right (remove last token)
        # and target output is shifted left (remove first token)
        output, _, _, _ = model(
            src, 
            tgt[:, :-1], 
            src_padding_mask=src_padding_mask,
            tgt_padding_mask=tgt_padding_mask[:, :-1],
            tgt_mask=tgt_mask[:-1, :-1]
        )
        
        # Reshape output and target for loss calculation
        output = output.contiguous().view(-1, output.size(-1))
        target = tgt[:, 1:].contiguous().view(-1)
        
        # Calculate loss
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        # Update parameters
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def evaluate(model, val_loader, criterion, device):
    """
    Evaluate the model on validation data.
    
    Args:
        model: Transformer model
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to use for computation
        
    Returns:
        Average loss for the validation set
    """
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            src = batch['source'].to(device)
            tgt = batch['target'].to(device)
            
            # Create masks
            src_padding_mask = (src == 0).to(device)
            tgt_padding_mask = (tgt == 0).to(device)
            
            # Create target mask to prevent attention to future tokens
            tgt_len = tgt.size(1)
            tgt_mask = model.generate_square_subsequent_mask(tgt_len).to(device)
            
            # Forward pass
            output, _, _, _ = model(
                src, 
                tgt[:, :-1], 
                src_padding_mask=src_padding_mask,
                tgt_padding_mask=tgt_padding_mask[:, :-1],
                tgt_mask=tgt_mask[:-1, :-1]
            )
            
            # Reshape output and target for loss calculation
            output = output.contiguous().view(-1, output.size(-1))
            target = tgt[:, 1:].contiguous().view(-1)
            
            # Calculate loss
            loss = criterion(output, target)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

def translate_sentence(model, sentence, source_tokenizer, target_tokenizer, device, max_len=50):
    """
    Translate a sentence using the trained model.
    
    Args:
        model: Transformer model
        sentence: Input sentence
        source_tokenizer: Tokenizer for source language
        target_tokenizer: Tokenizer for target language
        device: Device to use for computation
        max_len: Maximum length of generated translation
        
    Returns:
        Translated sentence
    """
    model.eval()
    
    # Tokenize the input sentence
    tokens = source_tokenizer.encode_as_ids(sentence)
    
    # Convert to tensor and add batch dimension
    src = torch.LongTensor(tokens).unsqueeze(0).to(device)
    
    # Create source padding mask
    src_padding_mask = (src == 0).to(device)
    
    # Encode the source sentence
    with torch.no_grad():
        encoder_output, _ = model.encoder(src, src_padding_mask)
    
    # Start with <sos> token
    tgt = torch.ones(1, 1).fill_(2).long().to(device)  # <sos> token
    
    for i in range(max_len):
        # Create target mask
        tgt_mask = model.generate_square_subsequent_mask(tgt.size(1)).to(device)
        
        # Decode
        with torch.no_grad():
            output, _, _, _ = model.decoder(
                tgt, encoder_output, tgt_mask, src_padding_mask
            )
            output = model.final_layer(output)
        
        # Get next token
        next_token = output[:, -1, :].argmax(dim=1).unsqueeze(1)
        
        # Add to target sequence
        tgt = torch.cat([tgt, next_token], dim=1)
        
        # Stop if <eos> token is predicted
        if next_token.item() == 3:  # <eos> token
            break
    
    # Convert tokens to text
    tokens = tgt.squeeze().cpu().numpy()
    
    # Skip <sos> and <eos> tokens
    tokens = tokens[1:]
    if 3 in tokens:  # Remove <eos> token if present
        tokens = tokens[:np.where(tokens == 3)[0][0]]
    
    # Convert tokens to text
    translation = target_tokenizer.decode_ids(tokens.tolist())
    
    return translation

def calculate_bleu(references, translations):
    """
    Calculate BLEU score for translations.
    
    Args:
        references: List of reference sentences
        translations: List of translated sentences
        
    Returns:
        BLEU score
    """
    smoothie = SmoothingFunction().method1
    bleu_scores = []
    
    for ref, trans in zip(references, translations):
        # Split sentences into words
        ref_words = ref.split()
        trans_words = trans.split()
        
        # Calculate BLEU score
        score = sentence_bleu([ref_words], trans_words, smoothing_function=smoothie)
        bleu_scores.append(score)
    
    return np.mean(bleu_scores)

def train_and_evaluate(model, train_loader, val_loader, optimizer, criterion, num_epochs, device,
                      source_tokenizer, target_tokenizer, val_df, checkpoint_dir='checkpoints'):
    """
    Train and evaluate the model.
    
    Args:
        model: Transformer model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        optimizer: Optimizer
        criterion: Loss function
        num_epochs: Number of epochs to train
        device: Device to use for computation
        source_tokenizer: Tokenizer for source language
        target_tokenizer: Tokenizer for target language
        val_df: Validation DataFrame
        checkpoint_dir: Directory to save checkpoints
        
    Returns:
        Tuple of (train_losses, val_losses, bleu_scores)
    """
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize tracking variables
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    bleu_scores = []
    
    # Training loop
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Train for one epoch
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        
        # Evaluate on validation set
        val_loss = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        # Calculate BLEU score on a subset of validation data
        translations = []
        references = []
        
        for i in range(min(10, len(val_df))):
            src_sentence = val_df['eng'].iloc[i]
            tgt_sentence = val_df['asm'].iloc[i]
            
            translation = translate_sentence(
                model, src_sentence, source_tokenizer, target_tokenizer, device
            )
            
            translations.append(translation)
            references.append(tgt_sentence)
        
        bleu_score = calculate_bleu(references, translations)
        bleu_scores.append(bleu_score)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'bleu_score': bleu_score,
            }, f'{checkpoint_dir}/best_model_custom_transformer.pth')
        
        # Print epoch results
        epoch_time = time.time() - start_time
        print(f'Epoch: {epoch+1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}')
        print(f'BLEU Score: {bleu_score:.4f}')
        print(f'Epoch Time: {epoch_time:.2f}s')
        
        # Print some example translations
        print("\nExample Translations:")
        for i in range(min(3, len(translations))):
            print(f"Source: {val_df['eng'].iloc[i]}")
            print(f"Reference: {references[i]}")
            print(f"Translation: {translations[i]}")
            print()
        
        print('-' * 50)
    
    return train_losses, val_losses, bleu_scores

def plot_training_progress(train_losses, val_losses, bleu_scores, output_dir='plots'):
    """
    Plot training progress.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        bleu_scores: List of BLEU scores
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot losses
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_dir}/losses.png')
    plt.close()
    
    # Plot BLEU scores
    plt.figure(figsize=(10, 6))
    plt.plot(bleu_scores, label='BLEU Score')
    plt.xlabel('Epoch')
    plt.ylabel('BLEU Score')
    plt.title('BLEU Score on Validation Set')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_dir}/bleu_scores.png')
    plt.close()

if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Hyperparameters
    D_MODEL = 512
    NUM_HEADS = 8
    D_FF = 2048
    NUM_ENCODER_LAYERS = 6
    NUM_DECODER_LAYERS = 6
    DROPOUT = 0.1
    VOCAB_SIZE = 8000
    BATCH_SIZE = 64
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.0001
    
    # Load data
    train_df, val_df = load_data('english_assamese.csv')
    source_tokenizer, target_tokenizer = create_tokenizers(train_df, vocab_size=VOCAB_SIZE)
    
    # Get vocabulary sizes
    INPUT_DIM = source_tokenizer.get_piece_size()
    OUTPUT_DIM = target_tokenizer.get_piece_size()
    
    print(f"Input vocabulary size: {INPUT_DIM}")
    print(f"Output vocabulary size: {OUTPUT_DIM}")
    
    # Create datasets
    train_dataset = TranslationDataset(
        train_df['eng'].tolist(),
        train_df['asm'].tolist(),
        source_tokenizer,
        target_tokenizer
    )
    
    val_dataset = TranslationDataset(
        val_df['eng'].tolist(),
        val_df['asm'].tolist(),
        source_tokenizer,
        target_tokenizer
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Initialize model
    model = Transformer(
        src_vocab_size=INPUT_DIM,
        tgt_vocab_size=OUTPUT_DIM,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dropout=DROPOUT
    ).to(device)
    
    # Print model architecture
    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding token (0)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)
    
    # Create directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    os.makedirs('attention_maps', exist_ok=True)
    
    # Train and evaluate model
    train_losses, val_losses, bleu_scores = train_and_evaluate(
        model, train_loader, val_loader, optimizer, criterion, NUM_EPOCHS, device,
        source_tokenizer, target_tokenizer, val_df
    )
    
    # Plot training progress
    plot_training_progress(train_losses, val_losses, bleu_scores)
    
    print("Training completed!")
