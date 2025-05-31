import pandas as pd
from collections import Counter

class Tokenizer:
    def __init__(self, texts, max_vocab_size=10000):
        self.word2idx = {}
        self.idx2word = {}
        self.max_vocab_size = max_vocab_size
        self.build_vocab(texts)
        
    def build_vocab(self, texts):
        # Initialize with special tokens
        self.word2idx = {
            '<pad>': 0,
            '<sos>': 1,
            '<eos>': 2,
            '<unk>': 3
        }
        
        # Count word frequencies
        word_freq = {}
        for text in texts:
            if isinstance(text, str):
                words = text.lower().split()
                for word in words:
                    word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort words by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        # Add most frequent words to vocabulary
        for word, _ in sorted_words:
            if len(self.word2idx) < self.max_vocab_size:
                self.word2idx[word] = len(self.word2idx)
            else:
                break
        
        # Create idx2word mapping
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        
        print(f"Vocabulary size: {len(self.word2idx)}")
        
    def encode(self, text):
        if not isinstance(text, str):
            return [self.word2idx['<unk>']]
        words = text.lower().split()
        return [self.word2idx.get(word, self.word2idx['<unk>']) for word in words]
        
    def decode(self, indices):
        return ' '.join([self.idx2word.get(idx, '<unk>') for idx in indices])
        
    def get_vocab_size(self):
        return self.max_vocab_size
        
    def get_pad_idx(self):
        return self.word2idx['<pad>']
        
    def get_sos_idx(self):
        return self.word2idx['<sos>']
        
    def get_eos_idx(self):
        return self.word2idx['<eos>']
        
    def get_unk_idx(self):
        return self.word2idx['<unk>']

def create_tokenizers(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Filter out non-string values
    source_texts = [str(text) for text in df['eng'].tolist() if isinstance(text, str)]
    target_texts = [str(text) for text in df['asm'].tolist() if isinstance(text, str)]
    
    print(f"Creating tokenizers with {len(source_texts)} source texts and {len(target_texts)} target texts")
    
    # Create tokenizers with fixed vocabulary size
    source_tokenizer = Tokenizer(source_texts, max_vocab_size=10000)
    target_tokenizer = Tokenizer(target_texts, max_vocab_size=10000)
    
    print(f"Source vocabulary size: {source_tokenizer.get_vocab_size()}")
    print(f"Target vocabulary size: {target_tokenizer.get_vocab_size()}")
    
    return source_tokenizer, target_tokenizer 