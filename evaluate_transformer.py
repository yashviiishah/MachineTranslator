import torch
from transformer_implementation import Transformer
from tokenizer import create_tokenizers
from data_loader import TranslationDataset
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
import os
from tqdm import tqdm

def load_model(model_path, device):
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    
    # Create tokenizers
    source_tokenizer, target_tokenizer = create_tokenizers(
        'english_assamese.csv',
        source_column='English',
        target_column='Assamese',
        source_vocab_size=8000,
        target_vocab_size=8000
    )
    
    if source_tokenizer is None or target_tokenizer is None:
        raise ValueError("Failed to create tokenizers. Please check the CSV file format and required columns.")
    
    # Create model with correct parameter names
    model = Transformer(
        src_vocab_size=8000,
        tgt_vocab_size=8000,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        max_seq_length=500
    ).to(device)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, source_tokenizer, target_tokenizer

def translate_sentence(model, sentence, source_tokenizer, target_tokenizer, device, max_length=100):
    model.eval()
    
    # Tokenize source sentence
    source_tokens = source_tokenizer.encode(sentence)
    source_tokens = source_tokens[:max_length] + [source_tokenizer.get_pad_idx()] * (max_length - len(source_tokens))
    source_tensor = torch.LongTensor(source_tokens).unsqueeze(0).to(device)
    
    # Initialize target with start token
    target_tokens = [target_tokenizer.get_pad_idx()]
    target_tensor = torch.LongTensor(target_tokens).unsqueeze(0).to(device)
    
    # Generate translation
    with torch.no_grad():
        for _ in range(max_length - 1):
            output = model(source_tensor, target_tensor)
            next_token = output.argmax(2)[:, -1].item()
            target_tokens.append(next_token)
            target_tensor = torch.LongTensor(target_tokens).unsqueeze(0).to(device)
            
            if next_token == target_tokenizer.get_pad_idx():
                break
    
    # Decode translation
    translation = target_tokenizer.decode(target_tokens)
    return translation

def plot_attention(attention, source_sentence, target_sentence, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention, cmap='YlOrRd', xticklabels=source_sentence.split(), yticklabels=target_sentence.split())
    plt.title('Attention Heatmap')
    plt.xlabel('Source')
    plt.ylabel('Target')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def evaluate_model(model, val_loader, source_tokenizer, target_tokenizer, device, num_examples=5):
    model.eval()
    references = []
    hypotheses = []
    attention_maps = []
    
    # Create directory for attention maps
    os.makedirs('attention_maps', exist_ok=True)
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader, desc="Evaluating")):
            if i >= num_examples:
                break
                
            src = batch['source'].to(device)
            trg = batch['target'].to(device)
            
            # Get model output and attention
            output = model(src, trg[:, :-1])
            
            # Get predictions
            pred_tokens = output.argmax(2).cpu().numpy()
            
            # Convert to text
            source_text = source_tokenizer.decode(src[0].cpu().numpy())
            target_text = target_tokenizer.decode(trg[0].cpu().numpy())
            pred_text = target_tokenizer.decode(pred_tokens[0])
            
            # Store for BLEU score
            references.append([target_text.split()])
            hypotheses.append(pred_text.split())
            
            # Plot attention heatmap
            attention = output[0].cpu().numpy()  # Get attention weights
            save_path = f'attention_maps/attention_map_{i}.png'
            plot_attention(attention, source_text, pred_text, save_path)
            attention_maps.append(attention)
            
            print(f"\nExample {i+1}:")
            print(f"Source: {source_text}")
            print(f"Target: {target_text}")
            print(f"Predicted: {pred_text}")
    
    # Calculate BLEU score
    bleu_score = corpus_bleu(references, hypotheses)
    print(f"\nCorpus BLEU Score: {bleu_score:.4f}")
    
    return bleu_score, attention_maps

if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model and tokenizers
    model, source_tokenizer, target_tokenizer = load_model('checkpoints/best_model.pth', device)
    
    # Create validation dataset
    val_dataset = TranslationDataset(
        'english_assamese.csv',
        source_tokenizer,
        target_tokenizer,
        max_length=100,
        split='val'
    )
    
    # Create validation loader
    val_loader = DataLoader(val_dataset, batch_size=1)
    
    # Evaluate model
    bleu_score, attention_maps = evaluate_model(
        model, val_loader, source_tokenizer, target_tokenizer, device
    )
    
    # Interactive translation
    print("\nInteractive Translation Mode")
    print("Enter 'quit' to exit")
    while True:
        sentence = input("\nEnter English sentence: ")
        if sentence.lower() == 'quit':
            break
            
        translation = translate_sentence(
            model, sentence, source_tokenizer, target_tokenizer, device
        )
        print(f"Translation: {translation}") 