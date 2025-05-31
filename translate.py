import torch
import torch.nn as nn
from model import Encoder, Decoder, Seq2Seq
import sentencepiece as spm

def load_model(model_path, source_tokenizer_path='english.model', target_tokenizer_path='assamese.model'):
    """Load the trained model and tokenizers"""
    # Initialize model components
    HIDDEN_SIZE = 512
    NUM_LAYERS = 2
    DROPOUT = 0.2
    
    # Load the tokenizers
    source_tokenizer = spm.SentencePieceProcessor()
    source_tokenizer.load(source_tokenizer_path)
    
    target_tokenizer = spm.SentencePieceProcessor()
    target_tokenizer.load(target_tokenizer_path)
    
    # Initialize model
    encoder = Encoder(
        input_size=source_tokenizer.get_piece_size(),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    )
    
    decoder = Decoder(
        output_size=target_tokenizer.get_piece_size(),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    )
    
    model = Seq2Seq(encoder, decoder)
    
    # Load the saved model state
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, source_tokenizer, target_tokenizer

def translate_text(model, text, source_tokenizer, target_tokenizer, device):
    """Translate a single text input"""
    model.eval()
    
    # Tokenize input text
    source_tensor = torch.tensor(source_tokenizer.encode_as_ids(text)).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Generate translation
        output = model(source_tensor, source_tensor, teacher_forcing_ratio=0)
        predicted_tokens = output.argmax(2)
        
        # Decode the translation
        translation = target_tokenizer.decode_ids(predicted_tokens[0].tolist())
    
    return translation

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model and tokenizers
    model, source_tokenizer, target_tokenizer = load_model(
        model_path='best_model.pth',
        source_tokenizer_path='english.model',
        target_tokenizer_path='assamese.model'
    )
    model = model.to(device)
    
    # Interactive translation loop
    print("\nEnglish to Assamese Translation")
    print("Enter 'q' to quit")
    
    while True:
        text = input("\nEnter English text: ").strip()
        if text.lower() == 'q':
            break
            
        if text:
            translation = translate_text(model, text, source_tokenizer, target_tokenizer, device)
            print(f"Translation: {translation}")
        else:
            print("Please enter some text to translate.")

if __name__ == "__main__":
    main() 