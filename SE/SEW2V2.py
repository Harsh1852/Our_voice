import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import os
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.manifold import TSNE
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from dataloader import preprocess_example
import time

df = preprocess_example()

# Creating a mapping of unique values for speakers to continuous numbers
unique_speaker_ids = df['speaker_id'].unique()
speaker_id_mapping = {speaker_id: idx for idx, speaker_id in enumerate(unique_speaker_ids)}

df['speaker_id'] = df['speaker_id'].map(speaker_id_mapping)

# Find a good fixed length (median or max with a safety margin)
waveform_lengths = [len(waveform) for waveform in df['speech']]
# median_length = int(np.median(waveform_lengths))
p95_length = int(np.percentile(waveform_lengths, 95))
target_length = p95_length

print(f"Processing all waveforms to length {target_length}")
print(f"Original length range: {min(waveform_lengths)} to {max(waveform_lengths)}")

# Processing all waveforms
for i in range(len(df)):
    waveform = df.iloc[i]['speech']
    
    # Converting waveforms to tensor (if needed)
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.tensor(waveform, dtype=torch.float32)
    
    current_length = waveform.shape[0]
    
    if current_length > target_length:
        # Either truncating from beginning, center, or with random offset
        # Center truncation:
        start = (current_length - target_length) // 2
        waveform = waveform[start:start + target_length]
    elif current_length < target_length:
        # Padding with zeros
        padding_size = target_length - current_length
        waveform = torch.nn.functional.pad(waveform, (0, padding_size))
    
    # Update the dataframe
    df.at[i, 'speech'] = waveform

# Now your df['speech'] contains fixed-length waveforms
# Continue with your original code without changing the DataLoader

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

# Custom dataset for speaker data
class SpeakerDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        waveform = self.df.iloc[idx]['speech']  
        speaker_id = self.df.iloc[idx]['speaker_id']
        instance_id = self.df.iloc[idx]['instance_id']
        
        # Converting speaker_id to numeric if it's not
        if not isinstance(speaker_id, (int, np.integer)):
            speaker_id = int(speaker_id)
        
        if self.transform:
            waveform = self.transform(waveform)
            
        return waveform, speaker_id

# Data augmentation function to apply random augmentations to a speech waveform
def apply_augmentation(waveform):
    # Ensuring that the waveform is on CPU for torchaudio operations
    device = waveform.device
    waveform = waveform.cpu()
    
    # Random time shift
    if random.random() > 0.5:
        shift = int(random.random() * (waveform.shape[-1] // 10))
        waveform = torch.roll(waveform, shift, dims=-1)
    
    # Random background noise (Gaussian)
    if random.random() > 0.7:
        noise_level = 0.005 * random.random()
        noise = torch.randn_like(waveform) * noise_level
        waveform = waveform + noise
    
    # Random volume change
    if random.random() > 0.7:
        volume_factor = 0.8 + 0.4 * random.random()  # 0.8 to 1.2
        waveform = waveform * volume_factor
        
    # Ensuring that the values are in a valid range
    waveform = torch.clamp(waveform, -1.0, 1.0)
    
    # Return to the original device
    return waveform.to(device)

# Speaker Embedding Model using Transfer Learning
class SpeakerEmbeddingModel(nn.Module):
    def __init__(self, embedding_dim=256, num_speakers=None, unfreeze_layers=2):
        super().__init__()
        # Loading the pretrained wav2vec2 model
        bundle = torchaudio.pipelines.WAV2VEC2_BASE
        self.wav2vec = bundle.get_model()
        
        # Freezing most of the pretrained model
        for param in self.wav2vec.parameters():
            param.requires_grad = False
        
        # Unfreezing the final few transformer layers for fine-tuning
        for i in range(unfreeze_layers):
            for param in self.wav2vec.encoder.transformer.layers[-1-i].parameters():
                param.requires_grad = True
        
        # Getting the output dimension of wav2vec2
        self.wav2vec_dim = self.wav2vec.encoder.transformer.layers[0].final_layer_norm.normalized_shape[0]
        
        # Attention-based pooling
        self.attention = nn.Sequential(
            nn.Linear(self.wav2vec_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Projecting to embedding dimension
        self.projector = nn.Sequential(
            nn.Linear(self.wav2vec_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, embedding_dim)
        )
        
        # (Optional) Classifier for speaker identification
        self.num_speakers = num_speakers
        if num_speakers:
            self.classifier = nn.Linear(embedding_dim, num_speakers)
    
    def forward(self, x):
        """
        Forward pass of the model
        x: tensor of shape [batch_size, time]
        """
        # Extracting features
        with torch.no_grad():
            # wav2vec2 requires 16kHz waveforms
            features, _ = self.wav2vec.extract_features(x)
        
        # The last layer features
        x = features[-1]  # [batch_size, time, feature_dim]
        
        # Applying attention pooling
        attention_weights = torch.softmax(self.attention(x), dim=1)
        x = torch.sum(x * attention_weights, dim=1)  # [batch_size, feature_dim]
        
        # Projecting to the embedding space
        embedding = self.projector(x)  # [batch_size, embedding_dim]
        
        # L2 normalization
        embedding = nn.functional.normalize(embedding, p=2, dim=1)
        
        # Classification (if needed)
        if self.num_speakers:
            logits = self.classifier(embedding)
            return embedding, logits
        
        return embedding

# Triplet selection functions
def create_triplets(embeddings, speaker_ids, mining='random'):
    """
    Create triplets for triplet loss
    mining: 'random', 'semi-hard', or 'hard'
    """
    triplets = []
    emb_matrix = embeddings.detach()
    
    # Get unique speaker ids
    unique_speakers = torch.unique(speaker_ids)
    
    # For each anchor
    for speaker in unique_speakers:
        # Getting indices for this speaker
        speaker_indices = (speaker_ids == speaker).nonzero().squeeze(1)
        
        # Taking at least 2 examples of this speaker
        if speaker_indices.numel() < 2:
            continue
            
        # Getting indices for other speakers
        other_indices = (speaker_ids != speaker).nonzero().squeeze(1)
        
        # Taking at least 1 example of another speaker
        if other_indices.numel() < 1:
            continue
        
        # For each potential anchor from this speaker
        for i in range(speaker_indices.size(0)):
            anchor_idx = speaker_indices[i].item()
            anchor = emb_matrix[anchor_idx]
            
            # Choosing a positive (same speaker, different utterance)
            pos_indices = [idx.item() for idx in speaker_indices if idx.item() != anchor_idx]
            
            if mining == 'hard' or mining == 'semi-hard':
                # Calculating distances to all positives
                pos_distances = []
                for pos_idx in pos_indices:
                    distance = torch.norm(anchor - emb_matrix[pos_idx])
                    pos_distances.append((distance.item(), pos_idx))
                
                # Hard mining: choose furthest positive
                if mining == 'hard':
                    positive_idx = max(pos_distances, key=lambda x: x[0])[1]
                else:
                    # semi-hard: choose random from top half
                    pos_distances.sort(key=lambda x: x[0], reverse=True)
                    positive_idx = pos_distances[random.randint(0, len(pos_distances)//2)][1]
            else:
                # Random mining: choose random positive
                positive_idx = random.choice(pos_indices)
            
            positive = emb_matrix[positive_idx]
            pos_distance = torch.norm(anchor - positive)
            
            if mining == 'hard' or mining == 'semi-hard':
                # Calculating distances to all negatives
                neg_distances = []
                for neg_idx in other_indices:
                    neg_idx = neg_idx.item()
                    distance = torch.norm(anchor - emb_matrix[neg_idx])
                    
                    # For semi-hard: d(a,n) > d(a,p)
                    if mining == 'semi-hard' and distance <= pos_distance:
                        continue
                        
                    neg_distances.append((distance.item(), neg_idx))
                
                if not neg_distances:
                    continue  # No suitable negative found
                
                if mining == 'hard':
                    # Hard mining: choose closest negative
                    negative_idx = min(neg_distances, key=lambda x: x[0])[1]
                else:  
                    # semi-hard: choose random from bottom half
                    neg_distances.sort(key=lambda x: x[0])
                    n_select = max(1, len(neg_distances)//2)
                    negative_idx = neg_distances[random.randint(0, n_select-1)][1]
            else:
                # Random mining: choose random negative
                negative_idx = other_indices[torch.randint(0, other_indices.size(0), (1,))].item()
            
            triplets.append((anchor_idx, positive_idx, negative_idx))
    
    if not triplets:
        return None, None, None
    
    # Extracting the actual embeddings for each index
    anchors = torch.stack([embeddings[i] for i, _, _ in triplets])
    positives = torch.stack([embeddings[j] for _, j, _ in triplets])
    negatives = torch.stack([embeddings[k] for _, _, k in triplets])
    
    return anchors, positives, negatives

# Combined loss function
class CombinedLoss(nn.Module):
    def __init__(self, margin=0.2, lambda_cls=1.0, lambda_triplet=1.0):
        super().__init__()
        self.classification_loss = nn.CrossEntropyLoss()
        self.triplet_loss = nn.TripletMarginLoss(margin=margin)
        self.lambda_cls = lambda_cls
        self.lambda_triplet = lambda_triplet
    
    def forward(self, embeddings, logits, targets, mining='semi-hard'):
        cls_loss = self.classification_loss(logits, targets)
        
        # Creating triplets
        anchors, positives, negatives = create_triplets(embeddings, targets, mining=mining)
        
        # If no triplets could be formed
        if anchors is None:
            return cls_loss
        
        trip_loss = self.triplet_loss(anchors, positives, negatives)
        
        return self.lambda_cls * cls_loss + self.lambda_triplet * trip_loss

# Training function for the model
def train_model(model, train_loader, val_loader, optimizer, scheduler, 
                criterion, device, epochs=30, mining='semi-hard', 
                augmentation=True, grad_clip=3.0):
    """
    Training loop for speaker embedding model
    """
    best_val_loss = float('inf')
    best_model_state = None
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        st = time.time()
        model.train()
        train_loss = 0.0
        batch_count = 0
        
        for waveforms, speaker_ids in train_loader:
            waveforms = waveforms.to(device)
            speaker_ids = speaker_ids.to(device)
            
            # Applying augmentation if enabled
            if augmentation:
                waveforms = torch.stack([apply_augmentation(w) for w in waveforms])
            
            # Forward pass
            embeddings, logits = model(waveforms)
            
            # Loss calculatiion
            loss = criterion(embeddings, logits, speaker_ids, mining=mining)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                
            optimizer.step()
            
            train_loss += loss.item()
            batch_count += 1
        
        avg_train_loss = train_loss / batch_count
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        batch_count = 0
        
        with torch.no_grad():
            for waveforms, speaker_ids in val_loader:
                waveforms = waveforms.to(device)
                speaker_ids = speaker_ids.to(device)
                
                # Forward pass
                embeddings, logits = model(waveforms)
                
                # Loss calculatiion (without triplet during validation for simplicity)
                loss = nn.CrossEntropyLoss()(logits, speaker_ids)
                
                val_loss += loss.item()
                batch_count += 1
        
        avg_val_loss = val_loss / batch_count
        val_losses.append(avg_val_loss)
        
        # Updating learning rate
        scheduler.step(avg_val_loss)
        
        # Saving the best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}')
        print(f'  Val Loss: {avg_val_loss:.4f}')
        print(f'  Time Taken: {time.time() - st}')
        # Computing metrics every 5 epochs or the last epoch
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            accuracy = compute_accuracy(model, val_loader, device)
            print(f'  Validation Accuracy: {accuracy:.2f}%')
    
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.tight_layout()
    plt.savefig('training_curve.png')
    plt.close()
    
    # Restoring the best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses

# Computing accuracy on a dataset
def compute_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for waveforms, speaker_ids in dataloader:
            waveforms = waveforms.to(device)
            speaker_ids = speaker_ids.to(device)
            
            _, logits = model(waveforms)
            _, predicted = torch.max(logits, 1)
            
            total += speaker_ids.size(0)
            correct += (predicted == speaker_ids).sum().item()
    
    return 100 * correct / total

# Computing Equal Error Rate (EER)
def compute_eer(model, dataloader, device):
    model.eval()
    all_embeddings = []
    all_speakers = []
    
    with torch.no_grad():
        for waveforms, speaker_ids in dataloader:
            waveforms = waveforms.to(device)
            
            if model.num_speakers:
                embeddings, _ = model(waveforms)
            else:
                embeddings = model(waveforms)
            
            all_embeddings.append(embeddings.cpu())
            all_speakers.append(speaker_ids)
    
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_speakers = torch.cat(all_speakers, dim=0)
    
    # Computing all pairwise scores
    scores = []
    labels = []
    
    num_embeddings = len(all_embeddings)
    for i in range(num_embeddings):
        for j in range(i+1, num_embeddings):
            sim = torch.cosine_similarity(
                all_embeddings[i].unsqueeze(0), 
                all_embeddings[j].unsqueeze(0)
            ).item()
            scores.append(sim)
            # 1 if same speaker, otherwise 0
            labels.append(1 if all_speakers[i] == all_speakers[j] else 0)
    
    # Computing EER
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    
    # Finding a threshold where FPR = FNR (EER)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    
    return eer

# Extracting and visualizing embeddings
def visualize_embeddings(model, dataloader, device, output_file='embeddings_visualization.png'):
    model.eval()
    all_embeddings = []
    all_speakers = []
    
    with torch.no_grad():
        for waveforms, speaker_ids in dataloader:
            waveforms = waveforms.to(device)
            
            if model.num_speakers:
                embeddings, _ = model(waveforms)
            else:
                embeddings = model(waveforms)
            
            all_embeddings.append(embeddings.cpu().numpy())
            all_speakers.append(speaker_ids.numpy())
    
    all_embeddings = np.vstack(all_embeddings)
    all_speakers = np.concatenate(all_speakers)
    
    # Reducing dimensionality with t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_embeddings)-1))
    embeddings_2d = tsne.fit_transform(all_embeddings)
    
    plt.figure(figsize=(12, 10))
    unique_speakers = np.unique(all_speakers)
    
    for speaker in unique_speakers:
        mask = all_speakers == speaker
        plt.scatter(
            embeddings_2d[mask, 0], 
            embeddings_2d[mask, 1], 
            label=f'Speaker {speaker}',
            alpha=0.7
        )
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('Speaker Embeddings Visualization (t-SNE)')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    return embeddings_2d, all_speakers

# Loading a trained speaker embedding model
def load_model(model_path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Creating a model with the loaded parameters
    model = SpeakerEmbeddingModel(
        embedding_dim=checkpoint['embedding_dim'],
        num_speakers=checkpoint['num_speakers']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    
    return model.to(device)

# Function to extract embedding for a single waveform
def extract_embedding(model, waveform):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        # Ensuring that the waveform has batch dimension
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        waveform = waveform.to(device)
        
        if model.num_speakers:
            embedding, _ = model(waveform)
        else:
            embedding = model(waveform)
            
    return embedding.cpu().numpy()

# Function to extract embeddings for all waveforms in the dataframe
def extract_all_embeddings(model, df, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    dataset = SpeakerDataset(df)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_embeddings = []
    
    model.eval()
    with torch.no_grad():
        for waveforms, _ in dataloader:
            waveforms = waveforms.to(device)
            
            if model.num_speakers:
                embeddings, _ = model(waveforms)
            else:
                embeddings = model(waveforms)
            
            all_embeddings.append(embeddings.cpu().numpy())
    
    all_embeddings = np.vstack(all_embeddings)
    
    result_df = df.copy()
    result_df['embedding'] = list(all_embeddings)
    
    return result_df

# Main function to train model and extract embeddings
def main(df, embedding_dim=256, batch_size=32, epochs=30, learning_rate=0.001, 
         mining='semi-hard', augmentation=True):
    """
    Main function to train speaker embedding model and extract embeddings
    
    Args:
        df: Pandas DataFrame with columns 'speech', 'speaker_id', 'instance_id'
        embedding_dim: Dimension of speaker embeddings
        batch_size: Batch size for training
        epochs: Number of training epochs
        learning_rate: Initial learning rate
        mining: Triplet mining strategy ('random', 'semi-hard', 'hard')
        augmentation: Whether to use data augmentation
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create numeric speaker IDs if needed
    if not pd.api.types.is_numeric_dtype(df['speaker_id']):
        speaker_to_id = {speaker: idx for idx, speaker in enumerate(df['speaker_id'].unique())}
        df['speaker_id_numeric'] = df['speaker_id'].map(speaker_to_id)
        print(f"Mapped {len(speaker_to_id)} speakers to numeric IDs")
    else:
        df['speaker_id_numeric'] = df['speaker_id']
    
    # Split data into train, validation, and test
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, 
                                         stratify=df['speaker_id_numeric'])
    train_df, val_df = train_test_split(train_df, test_size=0.25, random_state=42, 
                                        stratify=train_df['speaker_id_numeric'])
    
    print(f"Train set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    
    # Create datasets
    train_dataset = SpeakerDataset(train_df)
    val_dataset = SpeakerDataset(val_df)
    test_dataset = SpeakerDataset(test_df)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=4, pin_memory=True)
    
    # Initialize model
    num_speakers = len(df['speaker_id_numeric'].unique())
    model = SpeakerEmbeddingModel(embedding_dim=embedding_dim, num_speakers=num_speakers, 
                                 unfreeze_layers=2)
    model = model.to(device)
    
    # Print model summary
    print(f"Model initialized with {embedding_dim} dimensional embeddings")
    print(f"Number of speakers: {num_speakers}")
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.2f}%)")
    
    # Create optimizer with different learning rates for different parts
    optimizer = torch.optim.Adam([
        {'params': model.attention.parameters(), 'lr': learning_rate},
        {'params': model.projector.parameters(), 'lr': learning_rate},
        {'params': model.classifier.parameters(), 'lr': learning_rate},
        {'params': model.wav2vec.parameters(), 'lr': learning_rate * 0.1}
    ])
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=3, factor=0.5, verbose=True
    )
    
    # Loss function
    criterion = CombinedLoss(margin=0.2, lambda_cls=1.0, lambda_triplet=0.5)
    
    # Train model
    print(f"Starting training for {epochs} epochs...")
    model, train_losses, val_losses = train_model(
        model, train_loader, val_loader, optimizer, scheduler, criterion, 
        device, epochs=epochs, mining=mining, augmentation=augmentation
    )
    
    # Evaluate on test set
    test_accuracy = compute_accuracy(model, test_loader, device)
    print(f"Test accuracy: {test_accuracy:.2f}%")
    
    # Try to compute EER (may fail if not enough speakers/samples)
    try:
        eer = compute_eer(model, test_loader, device)
        print(f"Equal Error Rate: {eer:.4f}")
    except Exception as e:
        print(f"Could not compute EER: {e}")
    
    # Visualize embeddings
    print("Generating embedding visualization...")
    visualize_embeddings(model, test_loader, device)
    
    # Save model
    model_path = 'speaker_embedding_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'embedding_dim': embedding_dim,
        'num_speakers': num_speakers,
    }, model_path)
    print(f"Model saved to {model_path}")
    
    return model

# Example usage
# if __name__ == "__main__":
    # Assuming df is already loaded with columns:
    # - speech: preprocessed waveforms (torch tensors or numpy arrays)
    # - speaker_id: ID of the speaker
    # - instance_id: ID of the specific utterance
    # If you need to test with synthetic data, uncomment this:
    
    # # Train the model
    # model = main(
    #     df, 
    #     embedding_dim=256, 
    #     batch_size=32, 
    #     epochs=20, 
    #     learning_rate=0.001,
    #     mining='semi-hard',
    #     augmentation=False
    # )

    # # Load the model
    # model_path = 'SE_W2V2/2/speaker_embedding_model.pth'
    # model = load_model(model_path)
    # print(f"Model loaded from {model_path}")
    
    # # Extract embeddings for all data in the dataframe
    # embeddings_df = extract_all_embeddings(model, df)
    # print(f"Generated embeddings for {len(embeddings_df)} samples")

    # # Extract embeddings for one waveform
    # embeddings = extract_embedding(model, waveform)
    # print(f"Generated embeddings for {len(embeddings_df)}")
    
    # # Save embeddings
    # embeddings_df.to_pickle('speaker_embeddings.pkl')
    # print("Embeddings saved to speaker_embeddings.pkl")