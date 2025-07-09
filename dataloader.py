# import torch
import time
import os
import librosa
import glob
import numpy as np
import re
# from resemblyzer import VoiceEncoder
import pandas as pd

def load_flac(path):
    flacpath =  path
    flacpath = glob.glob(flacpath)
    
    waveforms = []
    ids = []
    for i in flacpath:
        ids.append(i[-18:-10])
        y, _ = librosa.load(i, sr=48000)  # y is the waveform as a NumPy array, sr is the sample rate
        waveforms.append(y)
    return waveforms, ids

def resample_and_trim(waveform, original_rate=48000, target_rate=16000):
    # Converting TensorFlow tensor to numpy
    waveform_np = waveform.squeeze().astype(np.float32)
    # Resampling audio to 16kHz from 48kHz
    resampled = librosa.resample(waveform_np, orig_sr=original_rate, target_sr=target_rate)
    # Trimming silence
    trimmed, _ = librosa.effects.trim(resampled, top_db=30)
    return trimmed.astype(np.float32)

def normalize_amplitude(waveform):
    # Normalizing audio amplitude by scaling it in [-1, 1] range. Ensuring consistent volume levels.
    peak = np.max(np.abs(waveform))
    return waveform / (peak + 1e-9)

# def waveform_to_mel(waveform, sample_rate=16000, n_mels=80, n_fft=1024, hop_length=256):
#     """Converting waveform to mel-spectrogram using librosa."""
#     mel = librosa.feature.melspectrogram(
#         y=waveform,
#         sr=sample_rate,
#         n_fft=n_fft,
#         hop_length=hop_length,
#         n_mels=n_mels
#     )
#     return librosa.power_to_db(mel, ref=np.max).T  # (Time, n_mels)

def load_txt(path): 
    txtfile = path
    txtpath = glob.glob(txtfile)
    
    texts = []
    ids = []
    for i in txtpath:
        ids.append(i[-12:-4])
        with open(i) as file:
            texts.append(file.read())
    return texts, ids

def clean_text(text):
    # Lowercase and striping whitespace
    text = text.lower().strip()
    # Removing special characters except apostrophes and spaces.
    text = re.sub(r"[^a-z' ]", "", text)  
    return text

# # Speaker Embedding (using resemblyzer's VoiceEncoder)

# encoder = VoiceEncoder()

# def extract_speaker_embedding(waveform, sample_rate=16000):
#     """Extracting the speaker embedding using Resemblyzer."""
#     embedding = encoder.embed_utterance(waveform, sample_rate)
#     if isinstance(embedding,tuple):\
#         # Extract the embedding from the tuple
#         embedding = embedding[0]
#     return embedding.astype(np.float32)  # shape = (256, )


# tacotron2 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tacotron2', model_math='fp16')
# tacotron2 = tacotron2.to("cuda" if torch.cuda.is_available() else "cpu")
# # tacotron2.eval()
    
# def text2mel(text):
#     utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tts_utils')
#     sequences, lengths = utils.prepare_input_sequence(text)
#     mel, _, _ = tacotron2.infer(sequences, lengths)
#     text2mel = mel[0].cpu().detach().numpy()
#     return text2mel  # The predicted mel spectrogram with shape (n_batch, n_mels e.g. 80, max of mel_specgram_lengths)

def preprocess_example(path1= 'train/txt/*/*.txt', path2 = 'train/wav48_silence_trimmed/*/*.flac'):
    texts, tids = load_txt(path1)
    waveforms, _ = load_flac(path2)
    
    # First applying the resample_and_trim, then normalized_amplitude on each waveform
    processed_waveforms = [normalize_amplitude(resample_and_trim(waveform)) for waveform in waveforms]
    
    # Standardizing the textual data and then applying tokenization
    processed_text = [clean_text(text) for text in texts]
    
    # mel_spectrogram = [waveform_to_mel(waveform) for waveform in processed_waveforms]
    
    # speaker_embedding = [extract_speaker_embedding(waveform) for waveform in waveforms]

    # txt2mel = [text2mel(text) for text in texts]

    speaker_id = [ids[1:4] for ids in tids]

    instance_id = [ids[-3:] for ids in tids]
    
    # Create a Pandas DataFrame
    data = {
        'speaker_id': speaker_id,
        'instance_id': instance_id,
        'text': processed_text,
        # 'text2mel': txt2mel,
        'speech': waveforms,
        # 'speaker_embedding': speaker_embedding,
        # 'mel_spectrogram': mel_spectrogram
    }
    return pd.DataFrame(data)


# Example usage
# if __name__ == "__main__":
    # Assuming the all the files are stored in folder named train
    # the text files are stored in the folder named txt 
    # the audio samples are stored in the folder named wav48_silence_trimmed and are of the type .flac
    
    # df = preprocess_example()