import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchaudio
from torchaudio.transforms import MelSpectrogram
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence

# Configuration Class
class Config:
    """Model configuration with all parameters in one place for easy adjustment"""
    # Audio parameters
    sampling_rate = 22050
    n_fft = 1024
    hop_length = 256
    win_length = 1024
    n_mels = 80
    mel_fmin = 0
    mel_fmax = 8000
    
    # Model parameters
    embedding_dim = 512
    encoder_dim = 512
    decoder_dim = 1024
    n_heads = 4
    reduction_factor = 1
    
    # Speaker embedding
    speaker_embedding_dim = 256
    
    # Training parameters
    batch_size = 16
    learning_rate = 1e-3
    max_epochs = 100
    grad_clip_thresh = 1.0
    warmup_steps = 4000
    
    # Paths and logging
    checkpoint_dir = "./checkpoints/"
    log_dir = "./logs/"
    
config = Config()

# Dataset Class
class TTSDataset(Dataset):
    def __init__(self, df, config):
        """
        Initialize dataset from a pandas DataFrame containing:
        - text: transcripts
        - speech: audio waveforms or paths to audio files
        - speaker_embedding: pre-computed speaker embeddings
        - speaker_id: speaker identifiers
        """
        self.df = df
        self.config = config
        self.mel_transform = MelSpectrogram(
            sample_rate=config.sampling_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            win_length=config.win_length,
            n_mels=config.n_mels,
            f_min=config.mel_fmin,
            f_max=config.mel_fmax,
        )
    
    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.df)
    
    def __getitem__(self, idx):
        """Get a single training example"""
        row = self.df.iloc[idx]
        
        # Text processing
        text = row['text']
        text_encoded = self.text_to_sequence(text)
        
        # Audio processing
        speech = row['speech']
        if isinstance(speech, str):  # If it's a path to an audio file
            waveform, sr = torchaudio.load(speech)
            if sr != self.config.sampling_rate:
                waveform = torchaudio.functional.resample(waveform, sr, self.config.sampling_rate)
        else:  # If it's already a waveform (numpy array or tensor)
            waveform = torch.tensor(speech) if not isinstance(speech, torch.Tensor) else speech
            
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Generate mel spectrogram
        mel_spectrogram = self.mel_transform(waveform).squeeze(0).T
        
        # Get speaker embedding and ID
        speaker_embedding = torch.tensor(row['speaker_embedding'])
        speaker_id = row['speaker_id']
        
        return {
            "text": torch.LongTensor(text_encoded),
            "text_lengths": torch.LongTensor([len(text_encoded)]),
            "mel_spectrogram": mel_spectrogram,
            "mel_lengths": torch.LongTensor([mel_spectrogram.shape[0]]),
            "speaker_embedding": speaker_embedding,
            "speaker_id": speaker_id
        }
    
    def text_to_sequence(self, text):
        """Convert text string to sequence of character IDs"""
        char_to_id = {char: i+1 for i, char in enumerate(set("abcdefghijklmnopqrstuvwxyz .,?!-"))}
        char_to_id['<pad>'] = 0
        
        sequence = [char_to_id.get(c.lower(), char_to_id.get(' ')) for c in text]
        return sequence

# Collate Function for DataLoader
def collate_fn(batch):
    """
    Collate function for batching variable-length text and mel-spectrogram sequences.
    Sorts by text length for more efficient packing and pads sequences to the same length.
    """
    # Sort by text length (longest to shortest) for packing efficiency
    batch = sorted(batch, key=lambda x: x["text_lengths"], reverse=True)
    
    # Extract individual elements from batch
    text = [item["text"] for item in batch]
    text_lengths = torch.cat([item["text_lengths"] for item in batch])
    
    mel_specs = [item["mel_spectrogram"] for item in batch]
    mel_lengths = torch.cat([item["mel_lengths"] for item in batch])
    
    speaker_embeddings = torch.stack([item["speaker_embedding"] for item in batch])
    speaker_ids = [item["speaker_id"] for item in batch]
    
    # Pad sequences to the same length within the batch
    text_padded = pad_sequence(text, batch_first=True, padding_value=0)
    mel_padded = pad_sequence(mel_specs, batch_first=True, padding_value=0)
    
    return {
        "text": text_padded,
        "text_lengths": text_lengths,
        "mel_spectrogram": mel_padded,
        "mel_lengths": mel_lengths,
        "speaker_embedding": speaker_embeddings,
        "speaker_id": speaker_ids
    }

# Text Encoder
class Encoder(nn.Module):
    def __init__(self, config):
        """
        Text encoder with character embedding, convolutional layers, and bidirectional LSTM.
        """
        super().__init__()
        self.embedding = nn.Embedding(256, config.embedding_dim, padding_idx=0)
        
        self.convs = nn.ModuleList([
            nn.Conv1d(config.embedding_dim, config.encoder_dim, kernel_size=k, padding=(k-1)//2)
            for k in [5, 5, 5]
        ])
        
        self.lstm = nn.LSTM(config.encoder_dim, config.encoder_dim // 2, 
                            num_layers=1, batch_first=True, bidirectional=True)
        
    def forward(self, text, text_lengths):
        """
        Forward pass through the encoder.
        
        Args:
            text: Tensor of character IDs [batch_size, max_text_len]
            text_lengths: Tensor with actual length of each text [batch_size]
            
        Returns:
            outputs: Encoded text features [batch_size, max_text_len, encoder_dim]
        """
        # Convert character IDs to embeddings
        x = self.embedding(text)
        
        # Prepare for 1D convolution
        x = x.transpose(1, 2)
        
        # Apply convolutional layers with ReLU activation
        for conv in self.convs:
            x = F.relu(conv(x))
        
        # Prepare for LSTM
        x = x.transpose(1, 2)
        
        # Pack sequence for efficiency in RNN
        x_packed = nn.utils.rnn.pack_padded_sequence(
            x, text_lengths.cpu(), batch_first=True, enforce_sorted=True
        )
        
        # Apply bidirectional LSTM
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x_packed)
        
        # Unpack sequences
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        
        return outputs

# Attention Mechanism
class LocationSensitiveAttention(nn.Module):
    def __init__(self, query_dim, key_dim, attention_dim):
        """
        Location-sensitive attention that considers both content and location information.
        This helps prevent attention failures like skipping words or repeating.
        
        Args:
            query_dim: Dimension of the query (from decoder)
            key_dim: Dimension of the keys (from encoder)
            attention_dim: Internal dimension of the attention mechanism
        """
        super().__init__()
        self.query_layer = nn.Linear(query_dim, attention_dim, bias=False)
        self.key_layer = nn.Linear(key_dim, attention_dim, bias=False)
        self.location_conv = nn.Conv1d(1, 32, kernel_size=31, padding=15, bias=False)
        self.location_layer = nn.Linear(32, attention_dim, bias=False)
        self.v = nn.Linear(attention_dim, 1, bias=False)
        
    def forward(self, query, keys, attention_weights_prev, mask=None):
        """
        Forward pass of the attention mechanism.
        
        Args:
            query: Decoder state [batch_size, 1, query_dim]
            keys: Encoder outputs [batch_size, max_text_len, key_dim]
            attention_weights_prev: Previous attention weights [batch_size, 1, max_text_len]
            mask: Optional mask for padding [batch_size, max_text_len]
            
        Returns:
            context: Context vector [batch_size, 1, key_dim]
            attention_weights: Attention weights [batch_size, 1, max_text_len]
        """
        # Project query
        processed_query = self.query_layer(query)  # [batch_size, 1, attention_dim]
        
        # Project keys
        processed_keys = self.key_layer(keys)  # [batch_size, max_text_len, attention_dim]
        
        # Process previous attention weights with location features
        attention_weights_prev = attention_weights_prev.transpose(1, 2)
        processed_attention = self.location_conv(attention_weights_prev)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_layer(processed_attention)
        
        # Combine content and location information
        energies = self.v(torch.tanh(
            processed_query + processed_keys + processed_attention))
        energies = energies.transpose(1, 2)
        
        # Apply mask for padding if provided
        if mask is not None:
            energies = energies.masked_fill(mask, -float('inf'))
        
        # Softmax to get attention weights
        attention_weights = F.softmax(energies, dim=2)
        
        # Apply attention weights to keys to get context vector
        context = torch.bmm(attention_weights, keys)
        
        return context, attention_weights

# Decoder with Speaker Conditioning
class Decoder(nn.Module):
    def __init__(self, config):
        """
        Decoder with attention and speaker conditioning.
        Generates mel-spectrograms autoregressively.
        """
        super().__init__()
        self.attention = LocationSensitiveAttention(
            config.decoder_dim, config.encoder_dim, config.decoder_dim)
        
        # Pre-net for input mel processing
        self.prenet = nn.Sequential(
            nn.Linear(config.n_mels * config.reduction_factor, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # LSTM layers for autoregressive generation
        self.lstm1 = nn.LSTMCell(256 + config.encoder_dim + config.speaker_embedding_dim, config.decoder_dim)
        self.lstm2 = nn.LSTMCell(config.decoder_dim, config.decoder_dim)
        
        # Projection layer for mel outputs
        self.mel_projection = nn.Linear(config.decoder_dim + config.encoder_dim, 
                                      config.n_mels * config.reduction_factor)
        
        # Stop token prediction
        self.stop_projection = nn.Linear(config.decoder_dim + config.encoder_dim, 1)
        
    def forward(self, encoder_outputs, mel_targets, speaker_embedding, teacher_forcing_ratio=1.0):
        """
        Forward pass of the decoder.
        
        Args:
            encoder_outputs: Outputs from the encoder [batch_size, max_text_len, encoder_dim]
            mel_targets: Target mel-spectrograms (for teacher forcing) [batch_size, max_mel_len, n_mels]
            speaker_embedding: Speaker embeddings [batch_size, speaker_embedding_dim]
            teacher_forcing_ratio: Probability of using teacher forcing (0.0-1.0)
            
        Returns:
            mel_outputs: Generated mel-spectrograms [batch_size, max_decoder_steps, n_mels]
            stop_outputs: Stop token predictions [batch_size, max_decoder_steps, 1]
            alignments: Attention weights for visualization [batch_size, max_decoder_steps, max_text_len]
        """
        batch_size = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)
        target_len = mel_targets.size(1) if mel_targets is not None else 0
        
        # Initialize LSTM states with zeros
        h1 = torch.zeros(batch_size, self.lstm1.hidden_size, device=encoder_outputs.device)
        c1 = torch.zeros(batch_size, self.lstm1.hidden_size, device=encoder_outputs.device)
        h2 = torch.zeros(batch_size, self.lstm2.hidden_size, device=encoder_outputs.device)
        c2 = torch.zeros(batch_size, self.lstm2.hidden_size, device=encoder_outputs.device)
        
        # Initialize attention weights and context
        attention_weights = torch.zeros(batch_size, 1, seq_len, device=encoder_outputs.device)
        attention_context = torch.zeros(batch_size, 1, encoder_outputs.size(2), device=encoder_outputs.device)
        
        # Initialize decoder input with zero frame
        decoder_input = torch.zeros(batch_size, config.n_mels * config.reduction_factor, 
                                    device=encoder_outputs.device)
        
        # Storage for outputs
        mel_outputs = []
        stop_outputs = []
        alignments = []
        
        # Expand speaker embedding for each time step
        speaker_embedding = speaker_embedding.unsqueeze(1)  # (B, 1, speaker_dim)
        
        # Maximum decoder steps
        max_len = target_len if target_len > 0 else 1000
        
        # Autoregressive decoding
        for t in range(max_len):
            # Teacher forcing
            if t > 0 and np.random.random() < teacher_forcing_ratio and t < target_len:
                decoder_input = mel_targets[:, t-1, :]
            
            # Process through prenet
            prenet_out = self.prenet(decoder_input)  # (B, 256)
            
            # Concatenate with previous attention context
            lstm_input = torch.cat([prenet_out, attention_context.squeeze(1)], dim=1)
            
            # Concatenate with speaker embedding - THIS IS KEY FOR SPEAKER CONDITIONING
            lstm_input = torch.cat([lstm_input, speaker_embedding.squeeze(1)], dim=1)
            
            # LSTM steps
            h1, c1 = self.lstm1(lstm_input, (h1, c1))
            h2, c2 = self.lstm2(h1, (h2, c2))
            
            # Attention calculation
            attention_query = h2.unsqueeze(1)  # (B, 1, decoder_dim)
            attention_context, attention_weights = self.attention(
                attention_query, encoder_outputs, attention_weights)
            
            # Concatenate for output projection
            projection_input = torch.cat([h2, attention_context.squeeze(1)], dim=1)
            
            # Generate output frame and stop token
            mel_output = self.mel_projection(projection_input)
            stop_output = self.stop_projection(projection_input)
            
            # Store outputs
            mel_outputs.append(mel_output.unsqueeze(1))
            stop_outputs.append(stop_output.unsqueeze(1))
            alignments.append(attention_weights)
            
            # Update decoder input for next step
            decoder_input = mel_output
            
            # Stop if stop token predicted (during inference)
            if t > 10 and torch.sigmoid(stop_output) > 0.5 and teacher_forcing_ratio == 0:
                break
        
        # Concatenate outputs along time dimension
        mel_outputs = torch.cat(mel_outputs, dim=1)
        stop_outputs = torch.cat(stop_outputs, dim=1)
        alignments = torch.cat(alignments, dim=1)
        
        return mel_outputs, stop_outputs, alignments

# Full Tacotron2 Model with Speaker Conditioning
class SpeakerConditionedTacotron2(nn.Module):
    def __init__(self, config):
        """
        Full Tacotron 2 model with speaker conditioning.
        
        Args:
            config: Configuration object with model parameters
        """
        super().__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        
        # Speaker projection layer
        self.speaker_projection = nn.Linear(config.speaker_embedding_dim, config.speaker_embedding_dim)
        
        # Reduction factor (how many frames to predict per step)
        self.reduction_factor = config.reduction_factor
        
    def forward(self, text, text_lengths, mel_targets, speaker_embedding, teacher_forcing_ratio=1.0):
        """
        Forward pass of the full model.
        
        Args:
            text: Text input [batch_size, max_text_len]
            text_lengths: Actual text lengths [batch_size]
            mel_targets: Target mel-spectrograms [batch_size, max_mel_len, n_mels]
            speaker_embedding: Speaker embeddings [batch_size, speaker_embedding_dim]
            teacher_forcing_ratio: Probability of using teacher forcing
            
        Returns:
            mel_outputs: Generated mel-spectrograms [batch_size, max_decoder_steps, n_mels]
            stop_outputs: Stop token predictions [batch_size, max_decoder_steps, 1]
            alignments: Attention weights [batch_size, max_decoder_steps, max_text_len]
        """
        # Process speaker embedding
        speaker_embedding = self.speaker_projection(speaker_embedding)
        
        # Encode text
        encoder_outputs = self.encoder(text, text_lengths)
        
        # Decode with attention and speaker conditioning
        mel_outputs, stop_outputs, alignments = self.decoder(
            encoder_outputs, mel_targets, speaker_embedding, teacher_forcing_ratio)
        
        return mel_outputs, stop_outputs, alignments
    
    def inference(self, text, speaker_embedding):
        """
        Inference mode: generate mel-spectrograms for new text with a specified speaker.
        
        Args:
            text: Text input [batch_size, max_text_len]
            speaker_embedding: Speaker embeddings [batch_size, speaker_embedding_dim]
            
        Returns:
            mel_outputs: Generated mel-spectrograms
            alignments: Attention alignments for visualization
        """
        # Encode text
        encoder_outputs = self.encoder(text, torch.LongTensor([text.size(1)]).to(text.device))
        
        # Process speaker embedding
        speaker_embedding = self.speaker_projection(speaker_embedding)
        
        # Decode with attention and speaker conditioning (no teacher forcing)
        mel_outputs, _, alignments = self.decoder(
            encoder_outputs, None, speaker_embedding, teacher_forcing_ratio=0.0)
        
        return mel_outputs, alignments

# Loss Functions
class Tacotron2Loss(nn.Module):
    def __init__(self):
        """
        Combined loss for Tacotron 2 training:
        1. MSE loss for mel-spectrogram reconstruction
        2. BCE loss for stop token prediction
        """
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        
    def forward(self, mel_outputs, mel_targets, stop_outputs, stop_targets):
        """
        Calculate combined loss.
        
        Args:
            mel_outputs: Predicted mel-spectrograms [batch_size, max_len, n_mels]
            mel_targets: Target mel-spectrograms [batch_size, max_len, n_mels]
            stop_outputs: Predicted stop tokens [batch_size, max_len, 1]
            stop_targets: Target stop tokens [batch_size, max_len, 1]
            
        Returns:
            total_loss: Combined loss
            mel_loss: Mel-spectrogram reconstruction loss
            stop_loss: Stop token prediction loss
        """
        # Calculate mel spectrogram reconstruction loss
        mel_loss = self.mse_loss(mel_outputs, mel_targets)
        
        # Calculate stop token prediction loss
        stop_loss = self.bce_loss(stop_outputs, stop_targets)
        
        # Combined loss
        total_loss = mel_loss + stop_loss
        
        return total_loss, mel_loss, stop_loss

# Training Function
def train_model(model, dataloader, optimizer, criterion, device, epoch, log_interval=10):
    """
    Train the model for one epoch.
    
    Args:
        model: The Tacotron 2 model
        dataloader: DataLoader for training data
        optimizer: Optimizer for parameter updates
        criterion: Loss function
        device: Device to run training on (CPU/GPU)
        epoch: Current epoch number
        log_interval: How often to log progress
        
    Returns:
        average_loss: Average loss over the epoch
    """
    model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # Move data to device
        text = batch["text"].to(device)
        text_lengths = batch["text_lengths"].to(device)
        mel_targets = batch["mel_spectrogram"].to(device)
        mel_lengths = batch["mel_lengths"].to(device)
        speaker_embedding = batch["speaker_embedding"].to(device)
        
        # Create stop token targets (1 at the end of sequence)
        max_len = mel_targets.size(1)
        stop_targets = torch.zeros(mel_targets.size(0), max_len, 1).to(device)
        for i, length in enumerate(mel_lengths):
            stop_targets[i, length-1:, 0] = 1.0
        
        # Forward pass
        mel_outputs, stop_outputs, _ = model(
            text, text_lengths, mel_targets, speaker_embedding)
        
        # Calculate loss
        loss, mel_loss, stop_loss = criterion(
            mel_outputs, mel_targets, stop_outputs, stop_targets)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_thresh)
        optimizer.step()
        
        # Logging
        total_loss += loss.item()
        if batch_idx % log_interval == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}/{len(dataloader)}, '
                  f'Loss: {loss.item():.4f}, Mel Loss: {mel_loss.item():.4f}, '
                  f'Stop Loss: {stop_loss.item():.4f}')
    
    return total_loss / len(dataloader)

# Validation Function
def validate_model(model, dataloader, criterion, device):
    """
    Evaluate the model on validation data.
    
    Args:
        model: The Tacotron 2 model
        dataloader: DataLoader for validation data
        criterion: Loss function
        device: Device to run validation on (CPU/GPU)
        
    Returns:
        average_loss: Average loss on validation data
    """
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Move data to device
            text = batch["text"].to(device)
            text_lengths = batch["text_lengths"].to(device)
            mel_targets = batch["mel_spectrogram"].to(device)
            mel_lengths = batch["mel_lengths"].to(device)
            speaker_embedding = batch["speaker_embedding"].to(device)
            
            # Create stop token targets
            max_len = mel_targets.size(1)
            stop_targets = torch.zeros(mel_targets.size(0), max_len, 1).to(device)
            for i, length in enumerate(mel_lengths):
                stop_targets[i, length-1:, 0] = 1.0
            
            # Forward pass
            mel_outputs, stop_outputs, _ = model(
                text, text_lengths, mel_targets, speaker_embedding)
            
            # Calculate loss
            loss, _, _ = criterion(
                mel_outputs, mel_targets, stop_outputs, stop_targets)
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

# Main Training Function
def main(df, config):
    """
    Main training function.
    
    Args:
        df: DataFrame with training data
        config: Configuration object
        
    Returns:
        model: Trained model
    """
    # Create directories for checkpoints and logs
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    
    # Set device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Split data (80% train, 20% validation)
    df_train = df.sample(frac=0.8, random_state=42)
    df_val = df.drop(df_train.index)
    
    # Create datasets and dataloaders
    train_dataset = TTSDataset(df_train, config)
    val_dataset = TTSDataset(df_val, config)
    
    train_dataloader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=4, pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=4, pin_memory=True
    )
    
    # Initialize model, criterion, and optimizer
    model = SpeakerConditionedTacotron2(config).to(device)
    criterion = Tacotron2Loss()
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    
    # Check if pre-trained checkpoint exists (for transfer learning)
    pretrained_path = None  # Set path to pretrained model if available
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"Loading pre-trained model from {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location=device)
        
        # Load pre-trained weights while ignoring mismatched layers
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        # Only initialize the speaker projection layer
        model.speaker_projection.reset_parameters()
        
        print("Transferred model parameters, initialized speaker projection layer")
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(1, config.max_epochs + 1):
        print(f"Epoch {epoch}/{config.max_epochs}")
        
        # Training
        train_loss = train_model(model, train_dataloader, optimizer, criterion, device, epoch)
        
        # Validation
        val_loss = validate_model(model, val_dataloader, criterion, device)
        
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        # Save checkpoint if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(config.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint at {checkpoint_path}")
        
    print("Training complete!")
    return model

# Example Usage
if __name__ == "__main__":
    # Assuming the dataframe 'df' has already been loaded with the necessary columns:
    # - 'speaker_embedding': Pre-extracted speaker embeddings
    # - 'speech': Either audio waveforms or paths to audio files
    # - 'text': Text transcripts
    # - 'speaker_id': Speaker identifiers
    
    # Main training function
    # model = main(df, config)
    
    # Example of how to use the trained model for inference
    def generate_speech(model, text, speaker_embedding, device):
        """
        Generate a mel-spectrogram for given text using a specific speaker's voice.
        
        Args:
            model: Trained Tacotron 2 model
            text: Input text string
            speaker_embedding: Speaker embedding vector
            device: Device to run inference on
            
        Returns:
            mel_spectrogram: Generated mel-spectrogram
            alignment: Attention alignment for visualization
        """
        model.eval()
        
        # Convert text to sequence
        def text_to_sequence(text):
            # Same encoding used in the dataset class
            char_to_id = {char: i+1 for i, char in enumerate(set("abcdefghijklmnopqrstuvwxyz .,?!-"))}
            char_to_id['<pad>'] = 0
            sequence = [char_to_id.get(c.lower(), char_to_id.get(' ')) for c in text]
            return sequence
        
        # Prepare text input
        text_encoded = torch.LongTensor([text_to_sequence(text)]).to(device)
        
        # Prepare speaker embedding
        speaker_embedding = torch.tensor(speaker_embedding).unsqueeze(0).to(device)
        
        # Generate mel-spectrogram
        with torch.no_grad():
            mel_outputs, alignments = model.inference(text_encoded, speaker_embedding)
        
        return mel_outputs[0].cpu().numpy(), alignments[0].cpu().numpy()

    # Example inference code (uncomment to use)
    """
    # Load a trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpeakerConditionedTacotron2(config).to(device)
    checkpoint = torch.load('checkpoints/best_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Select a speaker from your dataset
    example_speaker_idx = 0  # Choose any speaker index from your dataframe
    example_speaker_embedding = df.iloc[example_speaker_idx]['speaker_embedding']
    speaker_id = df.iloc[example_speaker_idx]['speaker_id']
    
    # Generate mel-spectrogram for new text
    text = "This is a test of the speaker-conditioned speech synthesis system."
    mel_spec, alignment = generate_speech(model, text, example_speaker_embedding, device)
    
    # Visualize the results
    plt.figure(figsize=(12, 6))
    
    # Plot mel-spectrogram
    plt.subplot(1, 2, 1)
    plt.imshow(mel_spec, aspect='auto', origin='lower')
    plt.colorbar()
    plt.title(f'Generated Mel-Spectrogram (Speaker: {speaker_id})')
    plt.xlabel('Frame')
    plt.ylabel('Mel Band')
    
    # Plot attention alignment
    plt.subplot(1, 2, 2)
    plt.imshow(alignment, aspect='auto', origin='lower')
    plt.colorbar()
    plt.title('Attention Alignment')
    plt.xlabel('Encoder Step (Text)')
    plt.ylabel('Decoder Step (Audio)')
    
    plt.tight_layout()
    plt.savefig('generated_speech.png')
    plt.show()
    
    # To convert to audio, you would need a vocoder
    # For example, with HiFi-GAN:
    # audio = vocoder(torch.tensor(mel_spec).unsqueeze(0))
    # torchaudio.save('output.wav', audio.cpu(), config.sampling_rate)
    """
    
    # Function to generate speech for multiple speakers
    def generate_multi_speaker_samples(model, df, text, device, output_dir="samples"):
        """
        Generate mel-spectrograms for the same text using different speakers.
        
        Args:
            model: Trained Tacotron 2 model
            df: DataFrame with speaker data
            text: Text to synthesize
            device: Device to run inference on
            output_dir: Directory to save visualizations
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Get unique speakers (limit to 5 for example)
        unique_speakers = df['speaker_id'].unique()[:5]
        
        for speaker_id in unique_speakers:
            # Get speaker row
            speaker_row = df[df['speaker_id'] == speaker_id].iloc[0]
            speaker_embedding = speaker_row['speaker_embedding']
            
            # Generate speech
            print(f"Generating speech for speaker {speaker_id}...")
            mel_spec, alignment = generate_speech(model, text, speaker_embedding, device)
            
            # Visualize
            plt.figure(figsize=(12, 6))
            
            # Plot mel-spectrogram
            plt.subplot(1, 2, 1)
            plt.imshow(mel_spec, aspect='auto', origin='lower')
            plt.colorbar()
            plt.title(f'Mel-Spectrogram (Speaker: {speaker_id})')
            plt.xlabel('Frame')
            plt.ylabel('Mel Band')
            
            # Plot alignment
            plt.subplot(1, 2, 2)
            plt.imshow(alignment, aspect='auto', origin='lower')
            plt.colorbar()
            plt.title('Attention Alignment')
            plt.xlabel('Encoder Step (Text)')
            plt.ylabel('Decoder Step (Audio)')
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/speech_speaker_{speaker_id}.png")
            plt.close()
            
        print(f"Generated samples saved to {output_dir}")
