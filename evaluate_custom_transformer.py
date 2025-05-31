import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
import matplotlib.pyplot as plt
import seaborn as sns
import os
from transformer_implementation import Transformer
from tokenizer import create_tokenizers
from data_loader import TranslationDataset
import math
from tqdm import tqdm

def plot_attention(attention, source, target, save_path):
    plt.figure(figsize=(10, 10))
    sns.heatmap(attention, cmap='viridis', xticklabels=source, yticklabels=target)
    plt.title('Attention Map')
    plt.xlabel('Source')
    plt.ylabel('Target')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def translate_sentence(model, sentence, source_tokenizer, target_tokenizer, device, max_length=50):
    model.eval()
    
    # Tokenize the input sentence
    tokens = source_tokenizer.encode_as_ids(sentence)
    tokens = [source_tokenizer.get_sos_idx()] + tokens + [source_tokenizer.get_eos_idx()]
    
    # Convert to tensor
    src = torch.LongTensor(tokens).unsqueeze(0).to(device)
    
    # Initialize target sequence with <sos> token
    tgt_tokens = [target_tokenizer.get_sos_idx()]
    
    with torch.no_grad():
        # Get encoder output
        src = model.src_embedding(src) * math.sqrt(model.d_model)
        src = model.pos_encoder(src)
        
        # Encoder forward pass
        memory = src
        for layer in model.encoder_layers:
            memory, _ = layer(memory)
        
        # Generate tokens one by one
        for i in range(max_length):
            tgt = torch.LongTensor(tgt_tokens).unsqueeze(0).to(device)
            
            # Create masks
            tgt_mask = model.generate_square_subsequent_mask(tgt.size(1)).to(device)
            
            # Decoder forward pass
            tgt = model.tgt_embedding(tgt) * math.sqrt(model.d_model)
            tgt = model.pos_encoder(tgt)
            
            output = tgt
            for layer in model.decoder_layers:
                output, _, _ = layer(output, memory, tgt_mask=tgt_mask)
            
            # Get predicted token
            output = model.fc_out(output)
            pred_token = output.argmax(2)[-1].item()
            tgt_tokens.append(pred_token)
            
            # Stop if we predict <eos> token
            if pred_token == target_tokenizer.get_eos_idx():
                break
    
    # Convert tokens to text
    translation = []
    for token in tgt_tokens[1:-1]:  # Skip <sos> and <eos> tokens
        translation.append(target_tokenizer.idx2word.get(token, '<unk>'))
    
    # Join tokens into sentence
    translation = ' '.join(translation)
    
    return translation

def evaluate_model(model, iterator, target_tokenizer, device):
    model.eval()
    translations = []
    references = []
    sources = []
    
    with torch.no_grad():
        for batch in iterator:
            src = batch['source'].to(device)
            tgt = batch['target'].to(device)
            
            # Create masks
            src_mask = (src != model.src_embedding.num_embeddings - 1).unsqueeze(-2)
            tgt_mask = model.generate_square_subsequent_mask(tgt.size(1)).to(device)
            
            # Forward pass
            output, _, _, _ = model(src, tgt[:, :-1], src_mask, tgt_mask[:-1, :-1])
            
            # Get predicted tokens
            pred_tokens = output.argmax(2)
            
            # Convert to text
            for i in range(len(pred_tokens)):
                # Get the tokens
                pred_seq = pred_tokens[i].cpu().numpy()
                ref_seq = tgt[i].cpu().numpy()
                src_seq = src[i].cpu().numpy()
                
                # Convert to text
                pred_text = []
                ref_text = []
                src_text = []
                
                # Process prediction sequence
                for token in pred_seq:
                    if token == target_tokenizer.get_pad_idx():
                        continue
                    if token == target_tokenizer.get_eos_idx():
                        break
                    pred_text.append(target_tokenizer.idx2word.get(token, '<unk>'))
                
                # Process reference sequence
                for token in ref_seq:
                    if token == target_tokenizer.get_pad_idx():
                        continue
                    if token == target_tokenizer.get_eos_idx():
                        break
                    ref_text.append(target_tokenizer.idx2word.get(token, '<unk>'))
                
                # Process source sequence
                for token in src_seq:
                    if token == target_tokenizer.get_pad_idx():
                        continue
                    if token == target_tokenizer.get_eos_idx():
                        break
                    src_text.append(target_tokenizer.idx2word.get(token, '<unk>'))
                
                # Join tokens into sentences
                pred_sentence = ' '.join(pred_text)
                ref_sentence = ' '.join(ref_text)
                src_sentence = ' '.join(src_text)
                
                translations.append(pred_sentence)
                references.append(ref_sentence)
                sources.append(src_sentence)
    
    return sources, translations, references

def calculate_bleu(references, translations):
    smoothie = SmoothingFunction().method1
    bleu_scores = []
    
    for ref, trans in zip(references, translations):
        # Split sentences into words for BLEU calculation
        ref_words = ref.split()
        trans_words = trans.split()
        score = sentence_bleu([ref_words], trans_words, smoothing_function=smoothie)
        bleu_scores.append(score)
    
    return np.mean(bleu_scores)

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
    MAX_SEQ_LENGTH = 500
    VOCAB_SIZE = 8000  # Match the vocabulary size from training
    
    try:
        # Load data
        print("Loading data...")
        csv_file = 'english_assamese.csv'
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found: {csv_file}")
        
        # Create tokenizers with the same vocabulary size as training
        source_tokenizer, target_tokenizer = create_tokenizers(
            csv_file,
            source_column='English',
            target_column='Assamese',
            source_vocab_size=VOCAB_SIZE,
            target_vocab_size=VOCAB_SIZE
        )
        
        if source_tokenizer is None or target_tokenizer is None:
            raise ValueError("Failed to create tokenizers. Please check the CSV file format and required columns.")
        
        print(f"Source vocabulary size: {len(source_tokenizer.vocab)}")
        print(f"Target vocabulary size: {len(target_tokenizer.vocab)}")
        
        # Create model with matching vocabulary sizes
        model = Transformer(
            src_vocab_size=VOCAB_SIZE,
            tgt_vocab_size=VOCAB_SIZE,
            d_model=D_MODEL,
            nhead=NHEAD,
            num_encoder_layers=NUM_ENCODER_LAYERS,
            num_decoder_layers=NUM_DECODER_LAYERS,
            dim_feedforward=DIM_FEEDFORWARD,
            dropout=DROPOUT,
            max_seq_length=MAX_SEQ_LENGTH
        ).to(device)
        
        # Load checkpoint
        checkpoint_path = 'checkpoints/best_model.pth'
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded successfully!")
        
        # Create validation dataset
        val_dataset = TranslationDataset(
            source_texts=source_texts[:EVAL_SIZE],
            target_texts=target_texts[:EVAL_SIZE],
            source_tokenizer=source_tokenizer,
            target_tokenizer=target_tokenizer,
            max_length=MAX_SEQ_LENGTH
        )
        
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # Evaluate model
        print("\nEvaluating model...")
        model.eval()
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        with torch.no_grad():
            total_loss = 0
            total_tokens = 0
            all_predictions = []
            all_targets = []
            
            for batch_idx, (src, tgt) in enumerate(tqdm(val_loader, desc="Evaluating")):
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
                
                # Store predictions and targets
                predictions = output.argmax(dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(tgt[:, 1:].cpu().numpy())
            
            # Calculate average loss
            avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
            print(f"\nAverage Loss: {avg_loss:.4f}")
            
            # Calculate BLEU score
            bleu_score = calculate_bleu_score(all_predictions, all_targets, target_tokenizer)
            print(f"BLEU Score: {bleu_score:.4f}")
            
            # Print example translations
            print("\nExample Translations:")
            for i in range(min(5, len(val_dataset))):
                src_text = val_dataset.source_texts[i]
                tgt_text = val_dataset.target_texts[i]
                pred_text = target_tokenizer.decode(all_predictions[i])
                
                print(f"\nSource: {src_text}")
                print(f"Target: {tgt_text}")
                print(f"Predicted: {pred_text}")
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please check the following:")
        print("1. The CSV file exists and has the correct format")
        print("2. The checkpoint file exists in the checkpoints directory")
        print("3. The model architecture matches the saved checkpoint")

if __name__ == "__main__":
    main()