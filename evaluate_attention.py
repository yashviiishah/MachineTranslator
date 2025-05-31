import torch
from model_attention import Encoder, Decoder, Seq2Seq
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from data_loader import load_data, create_tokenizers, TranslationDataset
from torch.utils.data import DataLoader
import os
import time

# Create attention_maps directory if it doesn't exist
os.makedirs('attention_maps', exist_ok=True)

def plot_attention(attention, source, target, src_vocab, trg_vocab, filename=None):
    # Convert attention weights to numpy array and handle dimensions
    if isinstance(attention, torch.Tensor):
        attention = attention.squeeze(0).cpu().numpy()  # Remove batch dimension if present
    else:
        attention = np.array(attention)
    
    # Ensure attention matrix is 2D
    if attention.ndim > 2:
        attention = attention.squeeze()
    
    # If attention is 1D, reshape it
    if attention.ndim == 1:
        attention = attention.reshape(1, -1)
    
    # Get source and target tokens
    src_tokens = []
    for i in source:
        if i not in [src_vocab['<sos>'], src_vocab['<eos>'], src_vocab['<pad>']]:
            token = list(src_vocab.keys())[list(src_vocab.values()).index(i)]
            src_tokens.append(token)
    
    trg_tokens = []
    for i in target:
        if i not in [trg_vocab['<sos>'], trg_vocab['<eos>'], trg_vocab['<pad>']]:
            token = list(trg_vocab.keys())[list(trg_vocab.values()).index(i)]
            trg_tokens.append(token)
    
    # Create figure and axis
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    
    # Plot attention matrix
    cax = ax.matshow(attention, cmap='viridis')
    fig.colorbar(cax)
    
    # Set up axes
    ax.set_xticklabels([''] + src_tokens, rotation=90)
    ax.set_yticklabels([''] + trg_tokens)
    
    # Show every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    
    plt.title('Attention Map')
    plt.xlabel('Source Tokens')
    plt.ylabel('Target Tokens')
    plt.tight_layout()
    
    # Save with timestamp if no filename provided
    if filename is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f'attention_maps/validation_{timestamp}.png'
    else:
        filename = f'attention_maps/{filename}'
    
    plt.savefig(filename)
    plt.close()
    print(f"Attention map saved as {filename}")

def evaluate_model(model, iterator, target_tokenizer, device):
    model.eval()
    translations = []
    references = []
    attention_maps = []
    
    with torch.no_grad():
        for batch in iterator:
            src = batch['source'].to(device)
            trg = batch['target'].to(device)
            
            output = model(src, trg, 0)  # Turn off teacher forcing
            
            # Get the predicted tokens
            pred_tokens = output.argmax(2)
            
            # Convert to text
            for i in range(len(pred_tokens)):
                # Get the tokens
                pred_seq = pred_tokens[i].cpu().numpy()
                ref_seq = trg[i].cpu().numpy()
                
                # Convert to text - handle each token individually
                pred_text = []
                ref_text = []
                
                # Process prediction sequence
                for token in pred_seq:
                    if token == 0:  # Skip padding
                        continue
                    if token == 2:  # Stop at <eos>
                        break
                    pred_text.append(target_tokenizer.id_to_piece(int(token)))
                
                # Process reference sequence
                for token in ref_seq:
                    if token == 0:  # Skip padding
                        continue
                    if token == 2:  # Stop at <eos>
                        break
                    ref_text.append(target_tokenizer.id_to_piece(int(token)))
                
                # Join tokens into sentences
                pred_sentence = ' '.join(pred_text)
                ref_sentence = ' '.join(ref_text)
                
                translations.append(pred_sentence)
                references.append(ref_sentence)
                
                # Store attention maps if available
                if hasattr(model, 'attention_weights'):
                    attention_maps.append((src[i].cpu().numpy(), pred_seq, model.attention_weights[i].cpu().numpy()))
    
    return translations, references, attention_maps

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

if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Hyperparameters (must match training)
    HIDDEN_DIM = 256
    N_LAYERS = 2
    DROPOUT = 0.3
    VOCAB_SIZE = 8000  # Must match training vocabulary size
    BATCH_SIZE = 64
    
    # Load data
    train_df, val_df = load_data('english_assamese.csv')
    source_tokenizer, target_tokenizer = create_tokenizers(train_df, vocab_size=VOCAB_SIZE)
    
    # Get vocabulary sizes
    INPUT_DIM = source_tokenizer.get_piece_size()
    OUTPUT_DIM = target_tokenizer.get_piece_size()
    
    # Initialize model
    enc = Encoder(INPUT_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT)
    dec = Decoder(OUTPUT_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT)
    model = Seq2Seq(enc, dec)
    model = model.to(device)
    
    # Load model state
    checkpoint_path = os.path.join('checkpoints', 'best_model_attention.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(checkpoint['state_dict'])
        print("Model loaded successfully from checkpoint")
    else:
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    # Create validation dataset and loader
    val_dataset = TranslationDataset(
        val_df['eng'].tolist(),
        val_df['asm'].tolist(),
        source_tokenizer,
        target_tokenizer
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Evaluate model
    translations, references, attention_maps = evaluate_model(model, val_loader, target_tokenizer, device)
    
    # Calculate BLEU score
    bleu_score = calculate_bleu(references, translations)
    print(f"Average BLEU score: {bleu_score:.4f}")
    
    # Print some example translations
    print("\nExample Translations:")
    for i in range(min(5, len(translations))):
        print(f"Source: {val_df['eng'].iloc[i]}")
        print(f"Reference: {references[i]}")
        print(f"Translation: {translations[i]}")
        print()
    
    # Plot attention maps for examples
    if attention_maps:
        for i, (source, target, attention) in enumerate(attention_maps[:5]):
            plot_attention(torch.tensor(attention), source, target, source_tokenizer, target_tokenizer)
            print(f"Saved attention map for example {i+1}") 