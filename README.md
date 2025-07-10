# Our Voice

This project is focused on the development of a robust single-shot voice cloning model, with the primary objective to overcome the limitations of the data requirements by creating a system capable of accurately replicating target speaker characteristics using only a minimal audio sample, thereby enabling instant cloning. This approach is intended to dramatically decrease preparation overhead and dataset requirements while preserving the naturalness and authentic identity of the cloned voice.

The core methodology for this project follows a modular, multi-stage pipeline architecture, a common practice in contemporary Text-to-Speech system design. This approach decomposes the task of generating speech from text into several distinct, manageable sub-problems, each addressed by a specialized component. 

The primary stages of this pipeline are: 
- [Dataset Preprocessing](#dataset)
- [Speaker Encoder](#speaker-encoder)
- [Text-to-Mel Synthesis](#text-to-mel)
- [Vocoder (Voice Encoder/Coder)](#vocoder)

## Dataset 
There are various publicly available datasets for training a voice cloning model. For this project, the [CSTR VCTK](https://datashare.ed.ac.uk/handle/10283/3443) corpus is used. It is a multi-speaker dataset with diverse accents and speaking styles, useful for evaluating the system's robustness. It consists of speech data by 110 speakers in various accents, in which every speaker reads about 400 sentences, selected from the Herald Glasgow papers. All recordings were converted into 16-bit and down-sampled to 48 kHz. 

## Speaker Encoder
A crucial aspect of voice cloning is the ability to accurately capture and represent the unique characteristics of a target speaker's voice. This project employed a dedicated speaker encoder module for this purpose, tasked with generating discriminative fixed-length embedding vectors from input speech.

The development involved the Wave2Vec2 model with transfer learning. Only the last two layers of the model were unfrozen and made trainable for the speaker verification task. This specific fine-tuning configuration resulted in approximately 15.6% learnable parameters of the model. This fine-tuning process was conducted for 10 epochs.

## Text to Mel
Tacotron2 architecture was selected for converting the input text sequence into a mel-spectrogram, which serves as an intermediate acoustic representation. which is a well-established end-to-end model in the TTS field, known for its ability to generate high-quality spectrograms.

The standard Tacotron2 architecture was modified to incorporate speaker identity information. Specifically, the speaker embeddings generated were integrated into the model flow. These embeddings were introduced after the Tacotron2 encoder processes the input text and generates its latent representations. The decoder then receives a combined input representation, consisting of both the linguistic information from the text encoder and the speaker identity information from the embedding.

## Vocoder
The MelGAN architecture was chosen for the vocoder part of this project. The selection of MelGAN was explicitly motivated by its computational efficiency and fast inference speed relative to alternative high-fidelity vocoders such as autoregressive models like WaveNet or certain configurations of HiFi-GAN. MelGAN, being a non-autoregressive, Generative Adversarial network-based vocoder, allows for parallel waveform generation directly from the mel-spectrogram, resulting in significantly faster synthesis.

## Result
All three components used in a voice cloning model, namely, the speaker embedding model, the text-to-mel model, and the vocoder model, were individually and independently trained. 
The speaker embedding model was trained on the speaker verification task using the loss function, which was a combination of a classification loss and a triplet loss, for 10 epochs, attaining 
1. Training Loss: 0.1349 
2. Validation Loss: 0.2178

The Text-to-Mel model was trained using the loss combined loss of mel-spectrogram reconstruction loss, which used mean square error to compare generated and targeted mel-spectrograms called Mel Loss, and a stop token prediction loss, which used a binary cross-entropy loss to predict when to stop producing frames, for 10 epochs.
1. Training Loss: 0.3005 
2. Validation Loss: 0.2825

The vocoder (Mel-GAN) model was used with a generator model having a loss that was a combination of adversarial loss and feature matching loss, with adversarial loss being the primary loss for the model. Whereas the discriminator model used a simple binary cross entropy, which penalized the model for misclassifying the generated audio files. The model was trained for 50 epochs.
1. Mel-reconstruction Error attained: 0.0161

## Conclusion
Despite attaining good results after hours of training on each component in the voice cloning model, the synthesized speech was not similar to the reference audio, but it was not a monotonic audio file, suggesting that more training on a variety of data is needed to improve the result.

### Poster

![Our Voice](https://github.com/user-attachments/assets/7aed5f3d-80fc-4859-a0cc-6dfc573f70b6)

