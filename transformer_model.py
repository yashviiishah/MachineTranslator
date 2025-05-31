import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt
import os

class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention mechanism.
    """
    def __init__(self, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        """
        Forward pass for scaled dot-product attention.
        
        Args:
            query: Query tensor of shape (batch_size, num_heads, seq_len, d_k)
            key: Key tensor of shape (batch_size, num_heads, seq_len, d_k)
            value: Value tensor of shape (batch_size, num_heads, seq_len, d_v)
            mask: Optional mask tensor of shape (batch_size, 1, 1, seq_len)
            
        Returns:
            Tuple of (output, attention_weights)
        """
        d_k = query.size(-1)
        
        # Scaled dot product of query and key
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention weights to values
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        
        # Ensure d_model is divisible by num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model)
        
        # Attention mechanism
        self.attention = ScaledDotProductAttention(dropout)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, query, key, value, mask=None):
        """
        Forward pass for multi-head attention.
        
        Args:
            query: Query tensor of shape (batch_size, seq_len, d_model)
            key: Key tensor of shape (batch_size, seq_len, d_model)
            value: Value tensor of shape (batch_size, seq_len, d_model)
            mask: Optional mask tensor
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size = query.size(0)
        residual = query
        
        # Linear projections and reshape for multi-head attention
        q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply scaled dot-product attention
        output, attention_weights = self.attention(q, k, v, mask)
        
        # Reshape and apply output projection
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(output)
        
        # Apply dropout and residual connection with layer normalization
        output = self.dropout(output)
        output = self.layer_norm(residual + output)
        
        return output, attention_weights

class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """
        Forward pass for position-wise feed-forward network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        residual = x
        
        # Apply feed-forward network
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        # Apply dropout and residual connection with layer normalization
        x = self.dropout(x)
        x = self.layer_norm(residual + x)
        
        return x

class PositionalEncoding(nn.Module):
    """
    Positional Encoding for transformer inputs.
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Forward pass for positional encoding.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class EncoderLayer(nn.Module):
    """
    Encoder Layer for the Transformer.
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
    def forward(self, x, mask=None):
        """
        Forward pass for encoder layer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional mask tensor
            
        Returns:
            Tuple of (output, attention_weights)
        """
        # Self-attention
        x, attention_weights = self.self_attention(x, x, x, mask)
        
        # Feed-forward network
        x = self.feed_forward(x)
        
        return x, attention_weights

class DecoderLayer(nn.Module):
    """
    Decoder Layer for the Transformer.
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
    def forward(self, x, encoder_output, self_mask=None, cross_mask=None):
        """
        Forward pass for decoder layer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            encoder_output: Output from encoder of shape (batch_size, seq_len, d_model)
            self_mask: Mask for self-attention
            cross_mask: Mask for cross-attention
            
        Returns:
            Tuple of (output, self_attention_weights, cross_attention_weights)
        """
        # Self-attention
        x, self_attention_weights = self.self_attention(x, x, x, self_mask)
        
        # Cross-attention with encoder output
        x, cross_attention_weights = self.cross_attention(x, encoder_output, encoder_output, cross_mask)
        
        # Feed-forward network
        x = self.feed_forward(x)
        
        return x, self_attention_weights, cross_attention_weights

class Encoder(nn.Module):
    """
    Transformer Encoder.
    """
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, dropout=0.1, max_len=5000):
        super(Encoder, self).__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # Create encoder layers
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x, mask=None):
        """
        Forward pass for encoder.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len)
            mask: Optional mask tensor
            
        Returns:
            Tuple of (output, attention_weights_list)
        """
        # Get sequence length
        seq_len = x.size(1)
        
        # Apply embedding and positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        
        # Store attention weights from each layer
        attention_weights_list = []
        
        # Apply encoder layers
        for layer in self.layers:
            x, attention_weights = layer(x, mask)
            attention_weights_list.append(attention_weights)
        
        return x, attention_weights_list

class Decoder(nn.Module):
    """
    Transformer Decoder.
    """
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, dropout=0.1, max_len=5000):
        super(Decoder, self).__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # Create decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x, encoder_output, self_mask=None, cross_mask=None):
        """
        Forward pass for decoder.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len)
            encoder_output: Output from encoder of shape (batch_size, seq_len, d_model)
            self_mask: Mask for self-attention
            cross_mask: Mask for cross-attention
            
        Returns:
            Tuple of (output, self_attention_weights_list, cross_attention_weights_list)
        """
        # Get sequence length
        seq_len = x.size(1)
        
        # Apply embedding and positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        
        # Store attention weights from each layer
        self_attention_weights_list = []
        cross_attention_weights_list = []
        
        # Apply decoder layers
        for layer in self.layers:
            x, self_attention_weights, cross_attention_weights = layer(
                x, encoder_output, self_mask, cross_mask
            )
            self_attention_weights_list.append(self_attention_weights)
            cross_attention_weights_list.append(cross_attention_weights)
        
        return x, self_attention_weights_list, cross_attention_weights_list

class Transformer(nn.Module):
    """
    Transformer model for sequence-to-sequence tasks.
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, d_ff=2048,
                 num_encoder_layers=6, num_decoder_layers=6, dropout=0.1, max_len=5000):
        super(Transformer, self).__init__()
        
        self.encoder = Encoder(src_vocab_size, d_model, num_heads, d_ff, num_encoder_layers, dropout, max_len)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_heads, d_ff, num_decoder_layers, dropout, max_len)
        self.final_layer = nn.Linear(d_model, tgt_vocab_size)
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_padding_mask=None, tgt_padding_mask=None):
        """
        Forward pass for transformer.
        
        Args:
            src: Source tensor of shape (batch_size, src_seq_len)
            tgt: Target tensor of shape (batch_size, tgt_seq_len)
            src_mask: Source mask for attention
            tgt_mask: Target mask for attention
            src_padding_mask: Source padding mask
            tgt_padding_mask: Target padding mask
            
        Returns:
            Tuple of (output, encoder_attention, decoder_self_attention, decoder_cross_attention)
        """
        # Encode source sequence
        encoder_output, encoder_attention = self.encoder(src, src_mask)
        
        # Decode target sequence
        decoder_output, decoder_self_attention, decoder_cross_attention = self.decoder(
            tgt, encoder_output, tgt_mask, src_mask
        )
        
        # Apply final linear layer
        output = self.final_layer(decoder_output)
        
        return output, encoder_attention, decoder_self_attention, decoder_cross_attention
    
    def generate_square_subsequent_mask(self, sz):
        """
        Generate a square mask for the sequence.
        
        Args:
            sz: Size of the square mask
            
        Returns:
            Mask tensor of shape (sz, sz)
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def visualize_attention(self, src_tokens, tgt_tokens, attention_weights, layer_idx=-1, head_idx=0, 
                           attention_type='encoder', output_dir='attention_maps'):
        """
        Visualize attention maps.
        
        Args:
            src_tokens: Source tokens
            tgt_tokens: Target tokens
            attention_weights: Attention weights from the model
            layer_idx: Index of the layer to visualize
            head_idx: Index of the attention head to visualize
            attention_type: Type of attention ('encoder', 'decoder_self', 'decoder_cross')
            output_dir: Directory to save attention maps
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if attention_type == 'encoder':
            # Encoder self-attention
            attn = attention_weights[layer_idx][0, head_idx].cpu().detach().numpy()
            plt.figure(figsize=(10, 8))
            plt.imshow(attn, cmap='viridis')
            plt.colorbar()
            plt.xticks(range(len(src_tokens)), src_tokens, rotation=90)
            plt.yticks(range(len(src_tokens)), src_tokens)
            plt.title(f'Encoder Self-Attention (Layer {layer_idx+1}, Head {head_idx+1})')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/encoder_layer{layer_idx+1}_head{head_idx+1}.png')
            plt.close()
            
        elif attention_type == 'decoder_self':
            # Decoder self-attention
            attn = attention_weights[layer_idx][0, head_idx].cpu().detach().numpy()
            plt.figure(figsize=(10, 8))
            plt.imshow(attn, cmap='viridis')
            plt.colorbar()
            plt.xticks(range(len(tgt_tokens)), tgt_tokens, rotation=90)
            plt.yticks(range(len(tgt_tokens)), tgt_tokens)
            plt.title(f'Decoder Self-Attention (Layer {layer_idx+1}, Head {head_idx+1})')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/decoder_self_layer{layer_idx+1}_head{head_idx+1}.png')
            plt.close()
            
        elif attention_type == 'decoder_cross':
            # Decoder cross-attention
            attn = attention_weights[layer_idx][0, head_idx].cpu().detach().numpy()
            plt.figure(figsize=(10, 8))
            plt.imshow(attn, cmap='viridis')
            plt.colorbar()
            plt.xticks(range(len(src_tokens)), src_tokens, rotation=90)
            plt.yticks(range(len(tgt_tokens)), tgt_tokens)
            plt.title(f'Decoder Cross-Attention (Layer {layer_idx+1}, Head {head_idx+1})')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/decoder_cross_layer{layer_idx+1}_head{head_idx+1}.png')
            plt.close()
