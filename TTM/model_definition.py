import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
import os
import json
from sklearn.model_selection import train_test_split
import time


# Audio processing utilities
class AudioProcessor:
    def __init__(self, sample_rate=22050, n_fft=1024, hop_length=256, n_mels=80, 
                 mel_fmin=0, mel_fmax=8000):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        
        # Create mel filter bank
        self.mel_filter = librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=mel_fmin,
            fmax=mel_fmax
        )
    
    def waveform_to_mel(self, waveform):
        """Convert waveform to mel spectrogram."""
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.numpy()
        
        # Short-time Fourier transform
        D = librosa.stft(
            waveform, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window='hann'
        )
        
        # Convert to power spectrogram
        S = np.abs(D) ** 2
        
        # Apply mel filterbank
        mel = np.dot(self.mel_filter, S)
        
        # Convert to log scale
        mel = np.log10(np.maximum(mel, 1e-10))
        
        # Convert to torch tensor
        mel = torch.FloatTensor(mel)
        
        return mel
    
    def normalize_mel(self, mel, mean=None, std=None):
        """Normalize mel spectrogram."""
        if mean is None and std is None:
            mean = mel.mean()
            std = mel.std()
        
        normalized_mel = (mel - mean) / std
        return normalized_mel, mean, std


# Custom dataset
class TextMelDataset(Dataset):
    def __init__(self, dataframe, processor, text_to_sequence_fn):
        self.df = dataframe
        self.processor = processor
        self.text_to_sequence_fn = text_to_sequence_fn
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        
        # Get text sequence
        text = item['text']
        text_sequence = self.text_to_sequence_fn(text)
        
        # Get waveform and convert to mel
        waveform = item['speech']
        mel = self.processor.waveform_to_mel(waveform)
        
        # Get speaker embedding
        speaker_embedding = item['embedding']
        
        return {
            'text': torch.LongTensor(text_sequence),
            'mel': mel,
            'speaker_embedding': torch.FloatTensor(speaker_embedding),
            'text_length': len(text_sequence),
            'mel_length': mel.shape[1]
        }


# Text processing utilities
class TextProcessor:
    def __init__(self):
        # Simple character-based tokenizer
        self.char_to_id = {c: i+1 for i, c in enumerate(' abcdefghijklmnopqrstuvwxyz.,!?-\'')}
        self.id_to_char = {i: c for c, i in self.char_to_id.items()}
        self.vocab_size = len(self.char_to_id) + 1  # +1 for <pad> token (0)
    
    def text_to_sequence(self, text):
        """Convert text to sequence of token ids."""
        text = text.lower()
        sequence = [self.char_to_id.get(c, 0) for c in text]
        return sequence
    
    def sequence_to_text(self, sequence):
        """Convert sequence of token ids to text."""
        text = ''.join([self.id_to_char.get(id, '') for id in sequence])
        return text


# Model components
class Encoder(nn.Module):
    """Text encoder that converts text to a sequence of encodings."""
    
    def __init__(self, vocab_size, embedding_dim=512, conv_layers=3, conv_channels=512, 
                 kernel_size=5, dropout=0.5, rnn_dim=512):
        super().__init__()
        
        # Text embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Convolutional layers
        self.convs = nn.ModuleList()
        for i in range(conv_layers):
            in_channels = embedding_dim if i == 0 else conv_channels
            self.convs.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels,
                        conv_channels,
                        kernel_size=kernel_size,
                        padding=(kernel_size - 1) // 2
                    ),
                    nn.BatchNorm1d(conv_channels),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            )
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            conv_channels,
            rnn_dim // 2,  # Bidirectional will double this
            batch_first=True,
            bidirectional=True
        )
        
    def forward(self, x, input_lengths=None):
        # x: [batch, text_length]
        
        # [batch, text_length, embedding_dim]
        x = self.embedding(x)
        
        # [batch, embedding_dim, text_length]
        x = x.transpose(1, 2)
        
        # Apply convolutional layers
        for conv in self.convs:
            x = conv(x)
        
        # [batch, conv_channels, text_length]
        x = x.transpose(1, 2)
        # [batch, text_length, conv_channels]
        
        if input_lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, input_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        
        # Apply LSTM
        outputs, _ = self.lstm(x)
        
        if input_lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        
        # [batch, text_length, rnn_dim]
        return outputs


class SpeakerConditionedAttention(nn.Module):
    """Attention mechanism that incorporates speaker embeddings."""
    
    def __init__(self, query_dim, key_dim, attention_dim=128, speaker_emb_dim=256):
        super().__init__()
        
        # Process query (decoder state)
        self.query_layer = nn.Linear(query_dim, attention_dim)
        
        # Process key (encoder outputs)
        self.key_layer = nn.Linear(key_dim, attention_dim)
        
        # Speaker conditioning projection
        self.speaker_projection = nn.Linear(speaker_emb_dim, attention_dim)
        
        # Combined attention
        self.energy_layer = nn.Linear(attention_dim, 1)
        
        # Store attention weights for visualization
        self.attention_weights = None
    
    def forward(self, query, keys, speaker_embedding, mask=None):
        # query: [batch, 1, query_dim]
        # keys: [batch, max_time, key_dim]
        # speaker_embedding: [batch, speaker_emb_dim]
        
        # Project query, keys and speaker embeddings to same dimension
        processed_query = self.query_layer(query)  # [batch, 1, attention_dim]
        processed_keys = self.key_layer(keys)      # [batch, max_time, attention_dim]
        
        # Project and expand speaker embedding
        speaker_proj = self.speaker_projection(speaker_embedding).unsqueeze(1)  # [batch, 1, attention_dim]
        
        # Combine with query to condition attention on speaker
        processed_query = processed_query + speaker_proj  # [batch, 1, attention_dim]
        
        # Attention energy
        alignment = self.energy_layer(torch.tanh(processed_query + processed_keys))  # [batch, max_time, 1]
        alignment = alignment.squeeze(-1)  # [batch, max_time]
        
        # Mask padding values
        if mask is not None:
            alignment.masked_fill_(mask, -float('inf'))
        
        # Softmax to get attention weights
        attention_weights = F.softmax(alignment, dim=1)  # [batch, max_time]
        self.attention_weights = attention_weights
        
        # Context vector
        context = torch.bmm(attention_weights.unsqueeze(1), keys)  # [batch, 1, key_dim]
        
        return context, attention_weights


class Prenet(nn.Module):
    """Prenet for the decoder."""
    
    def __init__(self, input_dim, hidden_dim=256, output_dim=128, dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        ])
        self.dropout = dropout
    
    def forward(self, x):
        for layer in self.layers:
            x = F.dropout(F.relu(layer(x)), p=self.dropout, training=True)
        return x


class SpeakerConditionedDecoder(nn.Module):
    """Decoder that generates mel spectrograms conditioned on speaker embeddings."""
    
    def __init__(self, encoder_dim=512, prenet_dim=256, speaker_emb_dim=256, decoder_dim=1024, 
                 attention_dim=128, mel_dim=80):
        super().__init__()
        
        # Prenet
        self.prenet = Prenet(mel_dim, prenet_dim, prenet_dim//2)
        
        # Attention
        self.attention = SpeakerConditionedAttention(
            decoder_dim, encoder_dim, attention_dim, speaker_emb_dim
        )
        
        # LSTM layers
        self.lstm1 = nn.LSTMCell(
            encoder_dim + prenet_dim//2,  # Context vector + prenet output
            decoder_dim
        )
        self.lstm2 = nn.LSTMCell(
            decoder_dim,
            decoder_dim
        )
        
        # Speaker conditioning projection
        self.speaker_projection = nn.Linear(speaker_emb_dim, decoder_dim)
        
        # Final layers
        self.mel_linear = nn.Linear(decoder_dim + encoder_dim, mel_dim)
        self.gate_linear = nn.Linear(decoder_dim + encoder_dim, 1)
    
    def forward_step(self, encoder_outputs, mel_prev, hidden_states, cell_states, speaker_embedding, t=0):
        # Process previous mel frame
        prenet_out = self.prenet(mel_prev)  # [batch, prenet_dim//2]
        
        # Get attention context first
        attention_context, attention_weights = self.attention(
            hidden_states[0].unsqueeze(1), encoder_outputs, speaker_embedding
        )
        attention_context = attention_context.squeeze(1)  # [batch, encoder_dim]
        
        # First LSTM - concatenate prenet_out and attention_context
        lstm1_input = torch.cat([prenet_out, attention_context], dim=1)
        hidden_states[0], cell_states[0] = self.lstm1(lstm1_input, (hidden_states[0], cell_states[0]))
        
        # Second LSTM
        hidden_states[1], cell_states[1] = self.lstm2(hidden_states[0], (hidden_states[1], cell_states[1]))
        
        # Apply speaker conditioning to LSTM output
        speaker_proj = self.speaker_projection(speaker_embedding)
        hidden_states[1] = hidden_states[1] + speaker_proj
        
        # Output projection
        decoder_output = torch.cat([hidden_states[1], attention_context], dim=1)
        
        # Mel output and gate
        mel_output = self.mel_linear(decoder_output)
        gate_output = self.gate_linear(decoder_output)
        
        return mel_output, gate_output, attention_weights, hidden_states, cell_states
    
    def forward(self, encoder_outputs, mel_targets=None, speaker_embedding=None, max_length=1000):
        """
        Decoder forward pass.
        """
        batch_size = encoder_outputs.size(0)
        mel_dim = self.mel_linear.out_features
        
        # Initialize outputs
        if mel_targets is not None:
            max_length = mel_targets.size(1)
            
        outputs = []
        gate_outputs = []
        alignments = []
        
        # Initialize hidden states and cell states for LSTMs
        hidden_states = [
            torch.zeros(batch_size, self.lstm1.hidden_size, device=encoder_outputs.device),
            torch.zeros(batch_size, self.lstm2.hidden_size, device=encoder_outputs.device)
        ]
        cell_states = [
            torch.zeros(batch_size, self.lstm1.hidden_size, device=encoder_outputs.device),
            torch.zeros(batch_size, self.lstm2.hidden_size, device=encoder_outputs.device)
        ]
        
        # Initial input is zero
        decoder_input = torch.zeros(
            batch_size, mel_dim, device=encoder_outputs.device
        )
        
        # For each time step
        for t in range(max_length):
            if mel_targets is not None and t > 0:
                decoder_input = mel_targets[:, t-1, :]  # Teacher forcing
                
            mel_output, gate_output, attention_weight, hidden_states, cell_states = self.forward_step(
                encoder_outputs, decoder_input, hidden_states, cell_states, 
                speaker_embedding, t
            )
            
            outputs.append(mel_output)
            gate_outputs.append(gate_output)
            alignments.append(attention_weight)
            
            decoder_input = mel_output  # Use generated output as next input
            
            # Stop if model predicts end of sequence
            if mel_targets is None and torch.sigmoid(gate_output.mean()) > 0.5:
                break
        
        # Stack all timesteps
        outputs = torch.stack(outputs, dim=1)  # [batch, time, mel_dim]
        gate_outputs = torch.stack(gate_outputs, dim=1).squeeze(-1)  # [batch, time]
        alignments = torch.stack(alignments, dim=1)  # [batch, time, text_len]
        
        return outputs, gate_outputs, alignments

class PostNet(nn.Module):
    """Refines the mel spectrogram prediction."""
    
    def __init__(self, mel_dim=80, conv_channels=512, conv_layers=5, kernel_size=5, dropout=0.5):
        super().__init__()
        
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(
            nn.Sequential(
                nn.Conv1d(mel_dim, conv_channels, kernel_size, 
                          padding=(kernel_size-1)//2),
                nn.BatchNorm1d(conv_channels),
                nn.Tanh(),
                nn.Dropout(dropout)
            )
        )
        
        # Middle layers
        for i in range(1, conv_layers-1):
            self.convs.append(
                nn.Sequential(
                    nn.Conv1d(conv_channels, conv_channels, kernel_size, 
                              padding=(kernel_size-1)//2),
                    nn.BatchNorm1d(conv_channels),
                    nn.Tanh(),
                    nn.Dropout(dropout)
                )
            )
        
        # Final layer
        self.convs.append(
            nn.Sequential(
                nn.Conv1d(conv_channels, mel_dim, kernel_size, 
                          padding=(kernel_size-1)//2),
                nn.BatchNorm1d(mel_dim),
                nn.Dropout(dropout)
            )
        )
    
    def forward(self, x):
        # x: [batch, time, mel_dim]
        
        # Change to channels first for convolutions
        x = x.transpose(1, 2)  # [batch, mel_dim, time]
        
        for conv in self.convs:
            x = conv(x)
        
        # Back to [batch, time, mel_dim]
        x = x.transpose(1, 2)
        
        return x


class SpeakerConditionedTTS(nn.Module):
    """End-to-end text-to-speech model with speaker conditioning."""
    
    def __init__(self, vocab_size, speaker_emb_dim=256, encoder_dim=512, decoder_dim=1024, 
                 mel_dim=80, max_mel_length=1000):
        super().__init__()
        
        self.encoder = Encoder(vocab_size, embedding_dim=512, rnn_dim=encoder_dim)
        self.decoder = SpeakerConditionedDecoder(
            encoder_dim=encoder_dim, 
            speaker_emb_dim=speaker_emb_dim,
            decoder_dim=decoder_dim,
            mel_dim=mel_dim
        )
        self.postnet = PostNet(mel_dim=mel_dim)
        self.max_mel_length = max_mel_length
    
    def forward(self, text, text_lengths, mel_targets=None, speaker_embedding=None):
        """
        Forward pass for the full model.
        """
        # Encode text
        encoder_outputs = self.encoder(text, text_lengths)
        
        # Generate mel spectrograms
        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, mel_targets, speaker_embedding, self.max_mel_length
        )
        
        # Refine mel spectrograms
        postnet_outputs = self.postnet(mel_outputs)
        postnet_outputs = postnet_outputs + mel_outputs  # Residual connection
        
        return mel_outputs, postnet_outputs, gate_outputs, alignments
    
    def inference(self, text, speaker_embedding):
        """
        Generate mel spectrograms for inference.
        """
        text_lengths = torch.tensor([text.size(1)], device=text.device)
        
        # Encode text
        encoder_outputs = self.encoder(text, text_lengths)
        
        # Generate mel spectrograms
        mel_outputs, _, alignments = self.decoder(
            encoder_outputs, None, speaker_embedding, self.max_mel_length
        )
        
        # Refine mel spectrograms
        postnet_outputs = self.postnet(mel_outputs)
        postnet_outputs = postnet_outputs + mel_outputs  # Residual connection
        
        return postnet_outputs, alignments


# Loss functions
class TTSLoss(nn.Module):
    """Combined loss for TTS training."""
    
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, mel_outputs, postnet_outputs, gate_outputs, 
                mel_targets, gate_targets, mel_lengths=None):
        # Mask for variable-length sequences
        if mel_lengths is not None:
            # Create mask
            mask = torch.arange(mel_targets.size(1)).expand(
                mel_targets.size(0), mel_targets.size(1)
            ).to(mel_targets.device) < mel_lengths.unsqueeze(1)
            
            # Apply mask to outputs and targets
            mel_targets = mel_targets.masked_select(mask.unsqueeze(2))
            mel_outputs = mel_outputs.masked_select(mask.unsqueeze(2))
            postnet_outputs = postnet_outputs.masked_select(mask.unsqueeze(2))
            gate_targets = gate_targets.masked_select(mask)
            gate_outputs = gate_outputs.masked_select(mask)
        
        # Mel loss
        mel_loss = self.mse_loss(mel_outputs, mel_targets)
        postnet_loss = self.mse_loss(postnet_outputs, mel_targets)
        
        # Gate loss
        gate_loss = self.bce_loss(gate_outputs, gate_targets)
        
        # Combined loss
        loss = mel_loss + postnet_loss + gate_loss
        
        return loss, mel_loss, postnet_loss, gate_loss


# Training functions
def collate_fn(batch):
    """Collate function for DataLoader."""
    # Sort by text length for packed sequence
    batch.sort(key=lambda x: x['text_length'], reverse=True)
    
    # Get max lengths
    max_text_len = max([x['text_length'] for x in batch])
    max_mel_len = max([x['mel_length'] for x in batch])
    
    # Initialize tensors
    text_padded = torch.zeros((len(batch), max_text_len), dtype=torch.long)
    mel_padded = torch.zeros((len(batch), max_mel_len, batch[0]['mel'].shape[0]))
    gate_padded = torch.zeros((len(batch), max_mel_len))
    speaker_emb = torch.zeros((len(batch), batch[0]['speaker_embedding'].shape[0]))
    
    text_lengths = torch.LongTensor([x['text_length'] for x in batch])
    mel_lengths = torch.LongTensor([x['mel_length'] for x in batch])
    
    # Fill tensors
    for i, b in enumerate(batch):
        text_padded[i, :b['text_length']] = b['text']
        mel_padded[i, :b['mel_length'], :] = b['mel'].transpose(0, 1)
        gate_padded[i, b['mel_length']-1:] = 1.0  # Set gate to 1 for padding
        speaker_emb[i] = b['speaker_embedding']
    
    return {
        'text': text_padded,
        'text_lengths': text_lengths,
        'mel': mel_padded,
        'mel_lengths': mel_lengths,
        'gate': gate_padded,
        'speaker_embedding': speaker_emb
    }

def evaluate_model(model, val_loader, criterion, device):
    """Evaluate the model on validation set."""
    model.eval()
    total_loss = 0
    mel_loss_total = 0
    postnet_loss_total = 0
    gate_loss_total = 0
    
    with torch.no_grad():
        for batch in val_loader:
            # Move to device
            text = batch['text'].to(device)
            text_lengths = batch['text_lengths'].to(device)
            mel = batch['mel'].to(device)
            mel_lengths = batch['mel_lengths'].to(device)
            gate = batch['gate'].to(device)
            speaker_embedding = batch['speaker_embedding'].to(device)
            
            # Forward pass
            mel_outputs, postnet_outputs, gate_outputs, _ = model(
                text, text_lengths, mel, speaker_embedding
            )
            
            # Compute loss
            loss, mel_loss, postnet_loss, gate_loss = criterion(
                mel_outputs, postnet_outputs, gate_outputs, 
                mel, gate, mel_lengths
            )
            
            total_loss += loss.item()
            mel_loss_total += mel_loss.item()
            postnet_loss_total += postnet_loss.item()
            gate_loss_total += gate_loss.item()
    
    # Calculate average loss
    avg_loss = total_loss / len(val_loader)
    avg_mel_loss = mel_loss_total / len(val_loader)
    avg_postnet_loss = postnet_loss_total / len(val_loader)
    avg_gate_loss = gate_loss_total / len(val_loader)
    
    return avg_loss, avg_mel_loss, avg_postnet_loss, avg_gate_loss

def test_model(model, test_loader, criterion, device):
    """Evaluate the model on test set."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in test_loader:
            # Move to device
            text = batch['text'].to(device)
            text_lengths = batch['text_lengths'].to(device)
            mel = batch['mel'].to(device)
            mel_lengths = batch['mel_lengths'].to(device)
            gate = batch['gate'].to(device)
            speaker_embedding = batch['speaker_embedding'].to(device)
            
            # Forward pass
            mel_outputs, postnet_outputs, gate_outputs, _ = model(
                text, text_lengths, mel, speaker_embedding
            )
            
            # Compute loss
            loss, _, _, _ = criterion(
                mel_outputs, postnet_outputs, gate_outputs, 
                mel, gate, mel_lengths
            )
            
            total_loss += loss.item()
    
    # Calculate average loss
    avg_loss = total_loss / len(test_loader)
    
    return avg_loss

def plot_losses(train_losses, val_losses, save_path="loss_plot.png"):
    """Plot training and validation losses."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs=100, save_dir="models"):
    """Train the model with validation."""
    os.makedirs(save_dir, exist_ok=True)
    
    model.train()
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        mel_loss_total = 0
        postnet_loss_total = 0
        gate_loss_total = 0
        
        for i, batch in enumerate(train_loader):
            # Move to device
            text = batch['text'].to(device)
            text_lengths = batch['text_lengths'].to(device)
            mel = batch['mel'].to(device)
            mel_lengths = batch['mel_lengths'].to(device)
            gate = batch['gate'].to(device)
            speaker_embedding = batch['speaker_embedding'].to(device)
            
            # Forward pass
            mel_outputs, postnet_outputs, gate_outputs, _ = model(
                text, text_lengths, mel, speaker_embedding
            )
            
            # Compute loss
            loss, mel_loss, postnet_loss, gate_loss = criterion(
                mel_outputs, postnet_outputs, gate_outputs, 
                mel, gate, mel_lengths
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            mel_loss_total += mel_loss.item()
            postnet_loss_total += postnet_loss.item()
            gate_loss_total += gate_loss.item()
            
            if i % 10 == 0:
                print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item():.4f}, "
                      f"Mel: {mel_loss.item():.4f}, Postnet: {postnet_loss.item():.4f}, "
                      f"Gate: {gate_loss.item():.4f}")
        
        # Calculate average training loss
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Evaluate on validation set
        val_loss, val_mel_loss, val_postnet_loss, val_gate_loss = evaluate_model(
            model, val_loader, criterion, device
        )
        val_losses.append(val_loss)
        
        print(f"Epoch: {epoch}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Train Mel: {mel_loss_total/len(train_loader):.4f}, Val Mel: {val_mel_loss:.4f}")
        print(f"Train Postnet: {postnet_loss_total/len(train_loader):.4f}, Val Postnet: {val_postnet_loss:.4f}")
        print(f"Train Gate: {gate_loss_total/len(train_loader):.4f}, Val Gate: {val_gate_loss:.4f}")
        
        # Save all checkpoints
        checkpoint_path = os.path.join(save_dir, f"tts_model_epoch_{epoch}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
        }, checkpoint_path)
        
        # Save the best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(save_dir, "tts_model_best.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
            }, best_model_path)
        
        # Plot and save losses after each epoch
        plot_losses(train_losses, val_losses, os.path.join(save_dir, "loss_plot.png"))
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
    }
    
    with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f)
    
    return train_losses, val_losses


def inference(model, text_processor, text, speaker_embedding, device):
    """Generate mel spectrogram for inference."""
    model.eval()
    
    # Encode text
    text_sequence = text_processor.text_to_sequence(text)
    text_tensor = torch.LongTensor([text_sequence]).to(device)
    
    # Speaker embedding
    speaker_emb = torch.FloatTensor(speaker_embedding).unsqueeze(0).to(device)
    
    # Generate mel
    with torch.no_grad():
        mel, alignments = model.inference(text_tensor, speaker_emb)
    
    return mel, alignments


# Function to prepare and train the model with train/val/test split
def prepare_and_train_with_validation(df, batch_size=32, epochs=100, save_dir="models"):
    """Prepare and train the TTS model with validation and testing."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize processors
    audio_processor = AudioProcessor()
    text_processor = TextProcessor()
    
    # Split dataset into train, validation, and test sets
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    print(f"Training set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    print(f"Test set size: {len(test_df)}")
    
    # Create datasets
    train_dataset = TextMelDataset(train_df, audio_processor, text_processor.text_to_sequence)
    val_dataset = TextMelDataset(val_df, audio_processor, text_processor.text_to_sequence)
    test_dataset = TextMelDataset(test_df, audio_processor, text_processor.text_to_sequence)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=4
    )
    
    # Create model
    model = SpeakerConditionedTTS(
        vocab_size=text_processor.vocab_size,
        speaker_emb_dim=256,  # Your speaker embedding dimension
        encoder_dim=512,
        decoder_dim=1024,
        mel_dim=80
    )
    
    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Create optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = TTSLoss()
    
    # Save model configuration and processors
    config = {
        'vocab_size': text_processor.vocab_size,
        'speaker_emb_dim': 256,
        'encoder_dim': 512,
        'decoder_dim': 1024,
        'mel_dim': 80,
        'char_to_id': text_processor.char_to_id,
    }
    
    with open(os.path.join(save_dir, 'model_config.json'), 'w') as f:
        json.dump(config, f)
    
    # Train model
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, optimizer, criterion, 
        device, epochs=epochs, save_dir=save_dir
    )
    
    # Load the best model for testing
    checkpoint = torch.load(os.path.join(save_dir, 'tts_model_best.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test the model
    test_loss = test_model(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")
    
    # Save test results
    with open(os.path.join(save_dir, 'test_results.json'), 'w') as f:
        json.dump({'test_loss': test_loss}, f)
    
    return model, text_processor, audio_processor, train_losses, val_losses, test_loss


# Main function to run the entire pipeline
def main(df_path=None, batch_size=32, epochs=50, save_dir="tts_model"):
    """Main function to run the entire TTS pipeline."""
    # Load your dataframe - adjust this to match how your data is stored
    # if df_path is not None:
    #     df = pd.read_csv(df_path)
    # else:
        # # For demonstration purposes, we'll assume df is already loaded
        # print("Dataframe path not provided. Please ensure df is loaded elsewhere.")
        # return
    df = df_path
    print("Starting model training with validation...")
    
    # Call the prepare_and_train function with your dataframe
    model, text_processor, audio_processor, train_losses, val_losses, test_loss = prepare_and_train_with_validation(
        df,                  # Your dataframe with speaker data
        batch_size=batch_size,  # Adjust based on your GPU memory
        epochs=epochs,         # Number of training epochs
        save_dir=save_dir
    )
    
    print("Training and testing complete!")
    print(f"Final Test Loss: {test_loss:.4f}")
    
    # Plot and save final learning curves
    plot_losses(train_losses, val_losses, os.path.join(save_dir, "final_loss_plot.png"))
    
    print(f"Model saved to {save_dir}")
    print(f"To use this model for inference, load it from {os.path.join(save_dir, 'tts_model_best.pt')}")


# Function to load the trained model for inference
def load_model_for_inference(model_dir="tts_model"):
    """Load the trained model for inference."""
    # Load model configuration
    with open(os.path.join(model_dir, 'model_config.json'), 'r') as f:
        config = json.load(f)
    
    # Create text processor
    text_processor = TextProcessor()
    text_processor.char_to_id = config['char_to_id']
    text_processor.id_to_char = {i: c for c, i in text_processor.char_to_id.items()}
    text_processor.vocab_size = len(text_processor.char_to_id) + 1
    
    # Create model
    model = SpeakerConditionedTTS(
        vocab_size=config['vocab_size'],
        speaker_emb_dim=config['speaker_emb_dim'],
        encoder_dim=config['encoder_dim'],
        decoder_dim=config['decoder_dim'],
        mel_dim=config['mel_dim']
    )
    
    # Load model weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(os.path.join(model_dir, 'tts_model_best.pt'), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, text_processor, device


# Example of how to use the loaded model for inference
def generate_mel_spectrogram(text, speaker_embedding, model_dir="tts_model"):
    """Generate mel spectrogram using the trained model."""
    # Load model
    model, text_processor, device = load_model_for_inference(model_dir)
    
    # Convert text to sequence
    text_sequence = text_processor.text_to_sequence(text)
    text_tensor = torch.LongTensor([text_sequence]).to(device)
    
    # Convert speaker embedding to tensor
    speaker_emb = torch.FloatTensor(speaker_embedding).unsqueeze(0).to(device)
    
    # Generate mel spectrogram
    with torch.no_grad():
        mel, alignments = model.inference(text_tensor, speaker_emb)
    
    return mel.cpu().numpy(), alignments.cpu().numpy()