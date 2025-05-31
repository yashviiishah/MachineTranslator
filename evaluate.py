import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_loader import load_data, create_tokenizers, TranslationDataset
from transformer import Transformer
import numpy as np
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
import os
from tqdm import tqdm

def load_model(model_path, src_vocab_size, tgt_vocab_size, device):
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=512,
        num_heads=8,
        num_layers=6,
        d_ff=2048
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model

def translate_sentence(model, sentence, source_tokenizer, target_tokenizer, device, max_length=50):
    model.eval()
    
    # Tokenize source sentence
    src_tokens = source_tokenizer.encode_as_ids(sentence)
    src_tensor = torch.LongTensor(src_tokens).unsqueeze(0).to(device)
    
    # Initialize target sequence with <s> token
    tgt_tokens = [target_tokenizer.bos_id()]
    
    # Generate translation
    with torch.no_grad():
        for _ in range(max_length):
            tgt_tensor = torch.LongTensor(tgt_tokens).unsqueeze(0).to(device)
            
            output, enc_attentions, dec_self_attentions, dec_cross_attentions = model(src_tensor, tgt_tensor)
            
            # Get the next token
            next_token = output[:, -1, :].argmax(1).item()
            
            # Break if we predict </s> token
            if next_token == target_tokenizer.eos_id():
                break
                
            tgt_tokens.append(next_token)
    
    # Decode the generated sequence
    translation = target_tokenizer.decode_ids(tgt_tokens[1:])  # Skip <s> token
    
    return translation, enc_attentions, dec_self_attentions, dec_cross_attentions

def calculate_bleu(references, hypotheses):
    smoothie = SmoothingFunction().method1
    bleu_scores = []
    
    for ref, hyp in zip(references, hypotheses):
        # Tokenize reference and hypothesis
        ref_tokens = [nltk.word_tokenize(ref.lower())]
        hyp_tokens = nltk.word_tokenize(hyp.lower())
        
        # Calculate BLEU score
        score = sentence_bleu(ref_tokens, hyp_tokens, smoothing_function=smoothie)
        bleu_scores.append(score)
    
    return np.mean(bleu_scores)

def plot_attention(attention, layer, head, save_path):
    plt.figure(figsize=(10, 10))
    plt.imshow(attention[0, head].cpu().numpy(), cmap='viridis')
    plt.colorbar()
    plt.title(f'Attention Map - Layer {layer}, Head {head}')
    plt.savefig(save_path)
    plt.close()

def main():
    # Create directories
    os.makedirs('evaluation_results', exist_ok=True)
    os.makedirs('attention_maps', exist_ok=True)
    
    # Load data and create tokenizers
    train_df, val_df = load_data('english_assamese.csv')
    source_tokenizer, target_tokenizer = create_tokenizers(train_df)
    
    # Create validation dataset and loader
    val_dataset = TranslationDataset(
        val_df['eng'].tolist(),
        val_df['asm'].tolist(),
        source_tokenizer,
        target_tokenizer
    )
    val_loader = DataLoader(val_dataset, batch_size=1)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(
        'checkpoints/best_model.pth',
        source_tokenizer.get_piece_size(),
        target_tokenizer.get_piece_size(),
        device
    )
    
    # Evaluate
    references = []
    hypotheses = []
    
    for i, batch in enumerate(tqdm(val_loader, desc="Evaluating")):
        src = batch['source'].to(device)
        tgt = batch['target'].to(device)
        
        # Get reference translation
        ref_tokens = tgt[0].tolist()
        ref = target_tokenizer.decode_ids(ref_tokens)
        references.append(ref)
        
        # Get source sentence
        src_tokens = src[0].tolist()
        src_sentence = source_tokenizer.decode_ids(src_tokens)
        
        # Generate translation
        translation, enc_attentions, dec_self_attentions, dec_cross_attentions = translate_sentence(
            model, src_sentence, source_tokenizer, target_tokenizer, device
        )
        hypotheses.append(translation)
        
        # Save attention maps for the first few examples
        if i < 5:
            plot_attention(
                enc_attentions[0],
                0, 0,
                f'attention_maps/enc_attention_example_{i}.png'
            )
            plot_attention(
                dec_self_attentions[0],
                0, 0,
                f'attention_maps/dec_self_attention_example_{i}.png'
            )
            plot_attention(
                dec_cross_attentions[0],
                0, 0,
                f'attention_maps/dec_cross_attention_example_{i}.png'
            )
    
    # Calculate BLEU score
    bleu_score = calculate_bleu(references, hypotheses)
    print(f"\nBLEU Score: {bleu_score:.4f}")
    
    # Save results
    with open('evaluation_results/results.txt', 'w', encoding='utf-8') as f:
        f.write(f"BLEU Score: {bleu_score:.4f}\n\n")
        f.write("Example Translations:\n")
        for i in range(min(10, len(references))):
            f.write(f"Source: {val_df['eng'].iloc[i]}\n")
            f.write(f"Reference: {references[i]}\n")
            f.write(f"Translation: {hypotheses[i]}\n")
            f.write("\n")

if __name__ == "__main__":
    main() 