import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.2):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(input_size, hidden_size)
        # Using bidirectional LSTM
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers,
                           dropout=dropout if num_layers > 1 else 0,
                           batch_first=True,
                           bidirectional=True)  # Enable bidirectional processing
        self.dropout = nn.Dropout(dropout)
        
        # Linear layer to combine bidirectional outputs
        self.fc = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.dropout(self.embedding(x))
        # embedded shape: (batch_size, seq_len, hidden_size)
        
        # Forward pass through bidirectional LSTM
        outputs, (hidden, cell) = self.lstm(embedded)
        # outputs shape: (batch_size, seq_len, hidden_size * 2)
        # hidden, cell shape: (num_layers * 2, batch_size, hidden_size)
        
        # Combine bidirectional outputs
        outputs = self.fc(outputs)
        # outputs shape: (batch_size, seq_len, hidden_size)
        
        # Combine forward and backward hidden states
        hidden = hidden.view(self.num_layers, 2, -1, self.hidden_size)
        hidden = torch.cat([hidden[:, 0, :, :], hidden[:, 1, :, :]], dim=2)
        hidden = self.fc(hidden)
        
        # Combine forward and backward cell states
        cell = cell.view(self.num_layers, 2, -1, self.hidden_size)
        cell = torch.cat([cell[:, 0, :, :], cell[:, 1, :, :]], dim=2)
        cell = self.fc(cell)
        
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers=2, dropout=0.2):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers,
                           dropout=dropout if num_layers > 1 else 0,
                           batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden, cell):
        # x shape: (batch_size, 1)
        # hidden, cell shape: (num_layers, batch_size, hidden_size)
        
        x = x.unsqueeze(1)  # (batch_size, 1) -> (batch_size, 1, 1)
        embedded = self.dropout(self.embedding(x))
        # embedded shape: (batch_size, 1, hidden_size)
        
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        # output shape: (batch_size, 1, hidden_size)
        
        prediction = self.fc(output.squeeze(1))
        # prediction shape: (batch_size, output_size)
        
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_forcing_ratio=0.5):
        batch_size = target.shape[0]
        target_len = target.shape[1]
        target_vocab_size = self.decoder.output_size
        
        outputs = torch.zeros(batch_size, target_len, target_vocab_size).to(source.device)
        
        hidden, cell = self.encoder(source)
        
        # First input to decoder is <sos> token
        input = target[:, 0]
        
        for t in range(1, target_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:, t] = output
            
            # Teacher forcing
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = target[:, t] if teacher_force else top1
        
        return outputs 