import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import time
import random
from sklearn.model_selection import train_test_split
from torch.cuda.amp import autocast, GradScaler

warnings.filterwarnings("ignore")

# MelGAN Architecture Components

class ResidualStack(nn.Module):
    def __init__(self, channels, dilation=1):
        super(ResidualStack, self).__init__()
        self.stack = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(dilation),
            nn.utils.weight_norm(nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation)),
            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.Conv1d(channels, channels, kernel_size=1))
        )
        
    def forward(self, x):
        return x + self.stack(x)

class Generator(nn.Module):
    def __init__(self, mel_channels=80, ngf=32, n_residual_layers=3):
        super(Generator, self).__init__()
        self.mel_channels = mel_channels
        
        # Initial conv to get to (ngf*4) channels = 128 channels
        self.conv_in = nn.Sequential(
            nn.ReflectionPad1d(3),
            nn.utils.weight_norm(nn.Conv1d(mel_channels, ngf * 4, kernel_size=7))
        )
        
        # Each upsampling layer reduces channels by half
        # 128 -> 64 -> 32 -> 16 -> 8
        self.ups = nn.ModuleList()
        self.ups.append(nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.ConvTranspose1d(ngf * 4, ngf * 2, kernel_size=8, stride=4, padding=2))
        ))
        self.ups.append(nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.ConvTranspose1d(ngf * 2, ngf, kernel_size=8, stride=4, padding=2))
        ))
        self.ups.append(nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.ConvTranspose1d(ngf, ngf // 2, kernel_size=8, stride=4, padding=2))
        ))
        self.ups.append(nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.ConvTranspose1d(ngf // 2, ngf // 4, kernel_size=8, stride=4, padding=2))
        ))
        
        # Residual stacks
        self.resblocks = nn.ModuleList()
        self.resblocks.append(nn.Sequential(*[
            ResidualStack(ngf * 2, dilation=3**j) for j in range(n_residual_layers)
        ]))
        self.resblocks.append(nn.Sequential(*[
            ResidualStack(ngf, dilation=3**j) for j in range(n_residual_layers)
        ]))
        self.resblocks.append(nn.Sequential(*[
            ResidualStack(ngf // 2, dilation=3**j) for j in range(n_residual_layers)
        ]))
        self.resblocks.append(nn.Sequential(*[
            ResidualStack(ngf // 4, dilation=3**j) for j in range(n_residual_layers)
        ]))
        
        # Final output conv - this expects 8 channels (ngf // 4) as input
        self.conv_out = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(3),
            nn.utils.weight_norm(nn.Conv1d(ngf // 4, 1, kernel_size=7)),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.conv_in(x)
        
        for i, (up, resblock) in enumerate(zip(self.ups, self.resblocks)):
            x = up(x)
            x = resblock(x)
        
        x = self.conv_out(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, ndf=16):
        super(Discriminator, self).__init__()
        
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.utils.weight_norm(nn.Conv1d(1, ndf, kernel_size=15, stride=1, padding=7)),
                nn.LeakyReLU(0.2, inplace=True)
            ),
            nn.Sequential(
                nn.utils.weight_norm(nn.Conv1d(ndf, ndf * 2, kernel_size=41, stride=4, padding=20, groups=4)),
                nn.LeakyReLU(0.2, inplace=True)
            ),
            nn.Sequential(
                nn.utils.weight_norm(nn.Conv1d(ndf * 2, ndf * 4, kernel_size=41, stride=4, padding=20, groups=16)),
                nn.LeakyReLU(0.2, inplace=True)
            ),
            nn.Sequential(
                nn.utils.weight_norm(nn.Conv1d(ndf * 4, ndf * 8, kernel_size=41, stride=4, padding=20, groups=16)),
                nn.LeakyReLU(0.2, inplace=True)
            ),
            nn.Sequential(
                nn.utils.weight_norm(nn.Conv1d(ndf * 8, ndf * 16, kernel_size=41, stride=4, padding=20, groups=16)),
                nn.LeakyReLU(0.2, inplace=True)
            ),
            nn.Sequential(
                nn.utils.weight_norm(nn.Conv1d(ndf * 16, ndf * 16, kernel_size=5, stride=1, padding=2)),
                nn.LeakyReLU(0.2, inplace=True)
            ),
            nn.utils.weight_norm(nn.Conv1d(ndf * 16, 1, kernel_size=3, stride=1, padding=1))
        ])
    
    def forward(self, x):
        feature_maps = []
        
        for layer in self.conv_layers:
            x = layer(x)
            feature_maps.append(x)
        
        return x, feature_maps


class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        
        self.discriminators = nn.ModuleList([
            Discriminator(ndf=16),
            Discriminator(ndf=16),
            Discriminator(ndf=16)
        ])
        
        self.downsample = nn.AvgPool1d(kernel_size=4, stride=2, padding=1)
    
    def forward(self, x):
        outputs = []
        feature_maps = []
        
        x1 = x
        x2 = self.downsample(x)
        x3 = self.downsample(x2)
        
        out1, fmap1 = self.discriminators[0](x1)
        out2, fmap2 = self.discriminators[1](x2)
        out3, fmap3 = self.discriminators[2](x3)
        
        outputs.append(out1)
        outputs.append(out2)
        outputs.append(out3)
        
        feature_maps.append(fmap1)
        feature_maps.append(fmap2)
        feature_maps.append(fmap3)
        
        return outputs, feature_maps


# Dataset for vocoder training directly from dataframe
class DataframeDataset(Dataset):
    def __init__(self, df, segment_size=8192, hop_length=256, max_mel_length=None):
        """
        Dataset for vocoder training from dataframe
        
        Args:
            df: DataFrame containing mel_spectrograms and speech columns
            segment_size: Audio segment size for training
            hop_length: Hop length between frames
            max_mel_length: Maximum length of mel spectrograms (for memory efficiency)
        """
        self.df = df
        self.segment_size = segment_size
        self.hop_length = hop_length
        self.max_mel_length = max_mel_length
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Get data from dataframe
        item = self.df.iloc[idx]
        
        # Get mel spectrogram and audio
        mel = item['mel_spectrograms']
        audio = item['speech']
        speaker_id = item['speaker_id']
        instance_id = item['instance_id']
        
        # Convert to tensor if needed
        if not torch.is_tensor(mel):
            mel = torch.FloatTensor(mel)
        else:
            mel = mel.detach().clone()
        
        if not torch.is_tensor(audio):
            audio = torch.FloatTensor(audio)
        else:
            audio = audio.detach().clone()
        
        # Process mel spectrogram
        if len(mel.shape) == 3 and mel.shape[0] == 1:
            mel = mel.squeeze(0).transpose(0, 1)
        elif len(mel.shape) == 2 and mel.shape[1] == self.mel_channels:
            mel = mel.transpose(0, 1)
        
        # Initialize start variable to None
        start = None
        
        # Limit mel length for memory efficiency if specified
        if self.max_mel_length and mel.size(1) > self.max_mel_length:
            start = random.randint(0, mel.size(1) - self.max_mel_length)
            mel = mel[:, start:start + self.max_mel_length]
        
        # Handle audio segmentation
        if audio.size(0) >= self.segment_size:
            if start is not None:  # If we limited the mel spectrogram
                audio_start = start * self.hop_length
                audio_end = min(audio_start + self.max_mel_length * self.hop_length,
                              audio.size(0))
                audio = audio[audio_start:audio_end]
                
                # Pad if needed
                if audio.size(0) < self.segment_size:
                    audio = F.pad(audio, (0, self.segment_size - audio.size(0)), 'constant')
            else:
                # Otherwise, randomly select an audio segment
                max_audio_start = audio.size(0) - self.segment_size
                audio_start = random.randint(0, max_audio_start)
                audio = audio[audio_start:audio_start + self.segment_size]
        else:
            # Pad audio if too short
            audio = F.pad(audio, (0, self.segment_size - audio.size(0)), 'constant')
        
        return {
            'mel': mel,
            'audio': audio,
            'speaker_id': speaker_id,
            'instance_id': instance_id
        }


# Custom collate function
def custom_collate(batch):
    """
    Custom collate function for variable-length mel spectrograms
    """
    # Get max lengths
    max_mel_time = max([item['mel'].shape[1] for item in batch])
    max_audio_length = max([item['audio'].shape[0] for item in batch])
    
    # Initialize tensors
    batch_size = len(batch)
    mel_channels = batch[0]['mel'].shape[0]
    
    mel_batch = torch.zeros(batch_size, mel_channels, max_mel_time)
    audio_batch = torch.zeros(batch_size, max_audio_length)
    speaker_ids = []
    instance_ids = []
    
    # Fill tensors
    for i, item in enumerate(batch):
        mel = item['mel']
        audio = item['audio']
        
        # Add to batch
        mel_batch[i, :, :mel.shape[1]] = mel
        audio_batch[i, :audio.shape[0]] = audio
        
        speaker_ids.append(item['speaker_id'])
        instance_ids.append(item['instance_id'])
    
    # Convert lists to appropriate format
    speaker_ids = torch.tensor(speaker_ids)
    instance_ids = torch.tensor(instance_ids)
    
    return {
        'mel': mel_batch,
        'audio': audio_batch,
        'speaker_id': speaker_ids,
        'instance_id': instance_ids,
        'mel_lengths': torch.tensor([item['mel'].shape[1] for item in batch]),
        'audio_lengths': torch.tensor([item['audio'].shape[0] for item in batch])
    }


# Loss functions
def feature_matching_loss(real_features, fake_features):
    loss = 0
    for i in range(len(real_features)):
        for j in range(len(real_features[i])):
            # Get minimum size to match dimensions
            if real_features[i][j].shape != fake_features[i][j].shape:
                # Find the minimum size along each dimension
                min_sizes = [min(r, f) for r, f in zip(real_features[i][j].shape, fake_features[i][j].shape)]
                
                # Slice both tensors to the minimum size
                r_tensor = real_features[i][j]
                f_tensor = fake_features[i][j]
                
                # Create slices for each dimension
                slices = tuple(slice(0, s) for s in min_sizes)
                
                # Apply slices
                r_tensor = r_tensor[slices]
                f_tensor = f_tensor[slices]
                
                loss += F.l1_loss(f_tensor, r_tensor.detach())
            else:
                loss += F.l1_loss(fake_features[i][j], real_features[i][j].detach())
    
    return loss


def discriminator_loss(real_outputs, fake_outputs):
    loss = 0
    
    for i in range(len(real_outputs)):
        loss += torch.mean((real_outputs[i] - 1) ** 2) + torch.mean(fake_outputs[i] ** 2)
    
    return loss / len(real_outputs)


def generator_loss(fake_outputs):
    loss = 0
    
    for output in fake_outputs:
        loss += torch.mean((output - 1) ** 2)
    
    return loss / len(fake_outputs)


# Function to split dataframe and create dataloaders
def create_train_val_test_dataloaders(df, batch_size=16, segment_size=8192, 
                                     val_size=0.1, test_size=0.1, max_mel_length=None):
    """
    Split dataframe into train, validation, and test sets and create dataloaders
    """
    # First split into temp and test
    test_fraction = test_size / (1 - val_size)
    temp_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    
    # Then split temp into train and validation
    train_df, val_df = train_test_split(temp_df, test_size=val_size/(1-test_size), random_state=42)
    
    print(f"Training set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    
    # Create datasets
    train_dataset = DataframeDataset(train_df, segment_size=segment_size, max_mel_length=max_mel_length)
    val_dataset = DataframeDataset(val_df, segment_size=segment_size, max_mel_length=max_mel_length)
    test_dataset = DataframeDataset(test_df, segment_size=segment_size, max_mel_length=max_mel_length)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        collate_fn=custom_collate
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=custom_collate
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=custom_collate
    )
    
    return train_loader, val_loader, test_loader


# Evaluation metrics
def compute_mel_reconstruction_error(gt_mel, pred_mel):
    """
    Compute mel-spectrogram reconstruction error
    """
    return F.l1_loss(gt_mel, pred_mel).item()


def evaluate_model(generator, dataloader, device):
    """
    Evaluate the model on a validation or test set
    """
    generator.eval()
    total_mel_error = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            mel = batch['mel'].to(device)
            audio = batch['audio'].to(device)
            
            # Generate audio
            fake_audio = generator(mel)
            
            # Handle dimension mismatch - only compare up to the minimum length
            min_length = min(fake_audio.squeeze(1).size(1), audio.size(1))
            fake_audio_trimmed = fake_audio.squeeze(1)[:, :min_length]
            audio_trimmed = audio[:, :min_length]
            
            # Compute metrics (simplified for efficiency)
            total_mel_error += F.l1_loss(fake_audio_trimmed, audio_trimmed).item()
    
    # Calculate average metrics
    avg_mel_error = total_mel_error / len(dataloader)
    
    return {
        'mel_error': avg_mel_error
    }

# Training function with validation
def train_melgan_with_validation(df, output_dir, epochs=50, batch_size=16, 
                               segment_size=8192, val_size=0.1, test_size=0.1,
                               learning_rate=0.0001, max_mel_length=None,
                               use_mixed_precision=True, grad_accum_steps=1):
    """
    Train the MelGAN vocoder with validation and testing
    
    Args:
        df: DataFrame containing mel_spectrograms and speech columns
        output_dir: Directory to save models and logs
        epochs: Number of training epochs
        batch_size: Batch size for training
        segment_size: Audio segment size for training
        val_size: Fraction of data to use for validation
        test_size: Fraction of data to use for testing
        learning_rate: Learning rate
        max_mel_length: Maximum length of mel spectrograms (for memory efficiency)
        use_mixed_precision: Whether to use mixed precision training
        grad_accum_steps: Number of steps to accumulate gradients
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create train, validation, and test dataloaders
    train_loader, val_loader, test_loader = create_train_val_test_dataloaders(
        df, batch_size, segment_size, val_size, test_size, max_mel_length
    )
    
    # Initialize models
    generator = Generator()
    discriminator = MultiScaleDiscriminator()
    
    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    
    # Optimizers
    optim_g = torch.optim.Adam(generator.parameters(), learning_rate, betas=(0.5, 0.9))
    optim_d = torch.optim.Adam(discriminator.parameters(), learning_rate, betas=(0.5, 0.9))
    
    # Learning rate schedulers
    scheduler_g = torch.optim.lr_scheduler.StepLR(optim_g, step_size=100000, gamma=0.5)
    scheduler_d = torch.optim.lr_scheduler.StepLR(optim_d, step_size=100000, gamma=0.5)
    
    # Initialize mixed precision if requested
    scaler_g = GradScaler() if use_mixed_precision else None
    scaler_d = GradScaler() if use_mixed_precision else None
    
    # Training loop
    generator.train()
    discriminator.train()
    
    # Initialize metrics tracking
    train_g_losses = []
    train_d_losses = []
    val_metrics = []
    best_val_metric = float('inf')
    
    # Training loop
    for epoch in range(epochs):
        print(f"\nEpoch: {epoch+1}/{epochs}")
        start_time = time.time()
        
        # Training
        generator.train()
        discriminator.train()
        
        epoch_g_loss = 0
        epoch_d_loss = 0
        
        # Reset gradients at the beginning of each epoch
        optim_g.zero_grad()
        optim_d.zero_grad()
        
        progress_bar = tqdm(train_loader, desc="Training")
        for i, batch in enumerate(progress_bar):
            mel = batch['mel'].to(device)
            audio = batch['audio'].unsqueeze(1).to(device)  # Add channel dimension
            
            # Train discriminator
            with autocast() if use_mixed_precision else nullcontext():
                # Generate audio
                with torch.no_grad():
                    fake_audio = generator(mel)
                
                # Get discriminator outputs
                real_outputs, real_features = discriminator(audio)
                fake_outputs, fake_features = discriminator(fake_audio.detach())
                
                # Discriminator loss
                d_loss = discriminator_loss(real_outputs, fake_outputs)
                d_loss = d_loss / grad_accum_steps  # Scale for gradient accumulation
            
            # Backward pass with mixed precision if enabled
            if use_mixed_precision:
                scaler_d.scale(d_loss).backward()
                if (i + 1) % grad_accum_steps == 0:
                    scaler_d.step(optim_d)
                    scaler_d.update()
                    optim_d.zero_grad()
            else:
                d_loss.backward()
                if (i + 1) % grad_accum_steps == 0:
                    optim_d.step()
                    optim_d.zero_grad()
            
            # Train generator
            with autocast() if use_mixed_precision else nullcontext():
                # Generate audio
                fake_audio = generator(mel)
                
                # Get discriminator outputs
                fake_outputs, fake_features = discriminator(fake_audio)
                
                # Generator adversarial loss
                adv_loss = generator_loss(fake_outputs)
                
                # Feature matching loss
                fm_loss = feature_matching_loss(real_features, fake_features)
                
                # Combined loss
                g_loss = adv_loss + 10 * fm_loss
                g_loss = g_loss / grad_accum_steps  # Scale for gradient accumulation
            
            # Backward pass with mixed precision if enabled
            if use_mixed_precision:
                scaler_g.scale(g_loss).backward()
                if (i + 1) % grad_accum_steps == 0:
                    scaler_g.step(optim_g)
                    scaler_g.update()
                    optim_g.zero_grad()
            else:
                g_loss.backward()
                if (i + 1) % grad_accum_steps == 0:
                    optim_g.step()
                    optim_g.zero_grad()
            
            # Updating progress bar and accumulate losses
            epoch_g_loss += g_loss.item() * grad_accum_steps
            epoch_d_loss += d_loss.item() * grad_accum_steps
            
            progress_bar.set_description(
                f"G: {g_loss.item():.4f}, D: {d_loss.item():.4f}"
            )
            
            # Updating learning rates
            scheduler_g.step()
            scheduler_d.step()
        
        # Average epoch losses
        epoch_g_loss /= len(train_loader)
        epoch_d_loss /= len(train_loader)
        
        train_g_losses.append(epoch_g_loss)
        train_d_losses.append(epoch_d_loss)
        
        # Validation
        print("Running validation...")
        val_metric = evaluate_model(generator, val_loader, device)
        val_metrics.append(val_metric)
        
        # Printing epoch summary
        time_elapsed = time.time() - start_time
        print(f"Epoch {epoch+1} completed in {time_elapsed:.2f}s")
        print(f"Train G Loss: {epoch_g_loss:.4f}, Train D Loss: {epoch_d_loss:.4f}")
        print(f"Validation Mel Error: {val_metric['mel_error']:.4f}")
        
        # Saving checkpoint
        checkpoint_path = os.path.join(output_dir, f"melgan_epoch_{epoch+1}.pt")
        torch.save({
            'generator': generator.state_dict(),
            'discriminator': discriminator.state_dict(),
            'optim_g': optim_g.state_dict(),
            'optim_d': optim_d.state_dict(),
            'epoch': epoch,
            'g_loss': epoch_g_loss,
            'd_loss': epoch_d_loss,
            'val_metric': val_metric
        }, checkpoint_path)
        
        # Saving the best model
        if val_metric['mel_error'] < best_val_metric:
            best_val_metric = val_metric['mel_error']
            best_model_path = os.path.join(output_dir, "melgan_best.pt")
            torch.save({
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'epoch': epoch,
                'val_metric': val_metric
            }, best_model_path)
            print(f"New best model saved with mel error: {best_val_metric:.4f}")
        
        # Plotting and saving loss curves
        plot_training_curves(train_g_losses, train_d_losses, val_metrics, output_dir)
        
        # Saving training log
        save_training_log(train_g_losses, train_d_losses, val_metrics, output_dir)
    
    # Final evaluation on test set
    print("\nEvaluating on test set...")
    
    # Loading the best model for testing
    checkpoint = torch.load(os.path.join(output_dir, "melgan_best.pt"), map_location=device)
    generator.load_state_dict(checkpoint['generator'])
    generator.eval()
    
    test_metrics = evaluate_model(generator, test_loader, device)
    
    print(f"Test Mel Error: {test_metrics['mel_error']:.4f}")
    
    # Saving the test results
    with open(os.path.join(output_dir, 'test_results.json'), 'w') as f:
        json.dump(test_metrics, f, indent=4)
    
    # Saving the final model
    final_path = os.path.join(output_dir, "melgan_final.pt")
    torch.save({
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict(),
        'test_metrics': test_metrics
    }, final_path)
    
    print(f"Training complete! Final model saved to {final_path}")
    
    return generator, test_metrics


# Utility functions for plotting and logging
def plot_training_curves(train_g_losses, train_d_losses, val_metrics, output_dir):
    """Plot training and validation curves"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extracting the validation metrics
    val_mel_errors = [m['mel_error'] for m in val_metrics]
    
    # Plotting the generator and discriminator losses
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(train_g_losses, label='Generator Loss')
    plt.plot(train_d_losses, label='Discriminator Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Generator and Discriminator Losses')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(val_mel_errors, label='Mel Error', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Metric Value')
    plt.title('Validation Metrics')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))
    plt.close()


def save_training_log(train_g_losses, train_d_losses, val_metrics, output_dir):
    """Save training log to JSON file"""
    training_log = {
        'train_g_losses': train_g_losses,
        'train_d_losses': train_d_losses,
        'val_metrics': val_metrics
    }
    
    with open(os.path.join(output_dir, 'training_log.json'), 'w') as f:
        json.dump(training_log, f, indent=4)


# Function to generate audio with the vocoder
def generate_audio_with_melgan(model_path, mel, device=None):
    """
    Generate audio from mel spectrogram using trained MelGAN
    
    Args:
        model_path: Path to trained model checkpoint
        mel: Mel spectrogram as numpy array or torch tensor [batch_size, n_mels, time]
             or [n_mels, time] for a single example
        device: Device to run generation on
    
    Returns:
        Generated audio as numpy array
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    generator = Generator()
    generator.load_state_dict(checkpoint['generator'])
    generator.to(device)
    generator.eval()
    
    # Convert mel to tensor if needed
    if not torch.is_tensor(mel):
        mel = torch.FloatTensor(mel)
    
    # Ensure proper dimensions
    if len(mel.shape) == 2:  # Single spectrogram
        # Check if mel is in the format [n_mels, time]
        if mel.size(0) < mel.size(1):
            mel = mel.unsqueeze(0)  # [1, n_mels, time]
        else:
            # If it's [time, n_mels], transpose and add batch dim
            mel = mel.transpose(0, 1).unsqueeze(0)  # [1, n_mels, time]
    elif len(mel.shape) == 3 and mel.shape[0] == 1 and mel.shape[2] == 80:
        # If it's [1, time, n_mels], transpose to [1, n_mels, time]
        mel = mel.transpose(1, 2)
    
    mel = mel.to(device)
    
    # Generating the audio
    with torch.no_grad():
        audio = generator(mel)
        audio = audio.squeeze(1)  # Remove channel dimension
    
    return audio.cpu().numpy()


# Function to save audio
def save_audio(audio, path, sr=22050):
    """Save audio to file"""
    # Converting the audio to float32 if needed
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    
    # Normalizing if not done already in [-1, 1]
    if np.max(np.abs(audio)) > 1.0:
        audio = audio / np.max(np.abs(audio))
    
    from scipy.io import wavfile
    wavfile.write(path, sr, audio)


print("MelGAN vocoder implementation is ready!")

# Example usage
if __name__ == "__main__":
    # Example DataFrame structure
    """
    df = pd.DataFrame({
        'mel_spectrograms': [...],  # List of mel spectrograms as numpy arrays
        'speech': [...],            # List of audio waveforms as numpy arrays
        'speaker_id': [...],        # List of speaker IDs
        'instance_id': [...]        # List of instance IDs
    })
    """
    # Training the MelGAN vocoder with memory-efficient settings
    generator, test_metrics = train_melgan_with_validation(
        df=df,    # Dataframe with mel_spectrograms, speech, speaker_id, instance_id
        output_dir="melgan_models",
        epochs=100,
        batch_size=32,    # Smaller batch size for lower memory usage
        segment_size=8192,         
        val_size=0.15,
        test_size=0.1,
        learning_rate=0.0001,
        max_mel_length=700,    # Limit mel length to save memory
        use_mixed_precision=True,    # Further reduce memory usage
        grad_accum_steps=4    # Effectively increases batch size with low memory
    )
    
    # Generating audio with the trained vocoder
    mel = df.iloc[0]['mel_spectrograms']
    audio = generate_audio_with_melgan("melgan_models/melgan_best.pt", mel)
    save_audio(audio[0], "generated_audio.wav")
