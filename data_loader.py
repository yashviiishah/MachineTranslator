import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import sentencepiece as spm
import tempfile
import os
import codecs

# Check for GPU and set device
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    device = torch.device('cpu')
    print("No GPU available, using CPU")

class TranslationDataset(Dataset):
    def __init__(self, source_texts, target_texts, source_tokenizer, target_tokenizer, max_length=100):
        self.source_texts = source_texts
        self.target_texts = target_texts
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.source_texts)
        
    def __getitem__(self, idx):
        source_text = self.source_texts[idx]
        target_text = self.target_texts[idx]
        
        return {
            'source': source_text,
            'target': target_text
        }

def load_data(csv_file):
    try:
        df = pd.read_csv(csv_file)
        source_texts = df['eng'].tolist()
        target_texts = df['asm'].tolist()
        return source_texts, target_texts
    except Exception as e:
        print(f"Error loading data: {e}")
        return [], []

def create_tokenizers(source_texts, target_texts, vocab_size=8000):
    try:
        # Create temporary files for training
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as source_file, \
             tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as target_file:
            
            # Write source texts
            for text in source_texts:
                if isinstance(text, str):
                    source_file.write(text + '\n')
            source_file_path = source_file.name
            
            # Write target texts
            for text in target_texts:
                if isinstance(text, str):
                    target_file.write(text + '\n')
            target_file_path = target_file.name
        
        try:
            # Train source tokenizer
            print("Training source tokenizer...")
            source_model_prefix = 'english'
            spm.SentencePieceTrainer.train(
                input=source_file_path,
                model_prefix=source_model_prefix,
                vocab_size=vocab_size,
                model_type='bpe',
                input_sentence_size=1000000,
                shuffle_input_sentence=True,
                character_coverage=1.0,
                pad_id=0,
                unk_id=1,
                bos_id=2,
                eos_id=3
            )
            
            # Train target tokenizer
            print("Training target tokenizer...")
            target_model_prefix = 'assamese'
            spm.SentencePieceTrainer.train(
                input=target_file_path,
                model_prefix=target_model_prefix,
                vocab_size=vocab_size,
                model_type='bpe',
                input_sentence_size=1000000,
                shuffle_input_sentence=True,
                character_coverage=1.0,
                pad_id=0,
                unk_id=1,
                bos_id=2,
                eos_id=3
            )
            
            # Load tokenizers
            source_tokenizer = spm.SentencePieceProcessor()
            source_tokenizer.load(f'{source_model_prefix}.model')
            
            target_tokenizer = spm.SentencePieceProcessor()
            target_tokenizer.load(f'{target_model_prefix}.model')
            
            print("Tokenizers created successfully!")
            print(f"Source vocabulary size: {source_tokenizer.get_piece_size()}")
            print(f"Target vocabulary size: {target_tokenizer.get_piece_size()}")
            
            return source_tokenizer, target_tokenizer
            
        finally:
            # Clean up temporary files
            os.unlink(source_file_path)
            os.unlink(target_file_path)
            
    except Exception as e:
        print(f"Error creating tokenizers: {e}")
        return None, None

if __name__ == "__main__":
    # Test the data loader
    print("\nTesting data loader...")
    source_texts, target_texts = load_data('english_assamese.csv')
    
    if source_texts and target_texts:
        print("\nCreating tokenizers...")
        source_tokenizer, target_tokenizer = create_tokenizers(source_texts, target_texts)
        
        if source_tokenizer and target_tokenizer:
            # Create a small test dataset
            test_dataset = TranslationDataset(
                source_texts[:5],
                target_texts[:5],
                source_tokenizer,
                target_tokenizer
            )
            
            print("\nTesting batch creation...")
            test_loader = DataLoader(test_dataset, batch_size=2, shuffle=True)
            for batch in test_loader:
                print(f"Source: {batch['source']}")
                print(f"Target: {batch['target']}")
                break 