import gradio as gr
import torch
import torchaudio
import numpy as np
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import torchaudio.transforms as T

MODEL_ID = "Zeyadd-Mostaffa/wav2vec_checkpoints"

# 1) Load model & feature extractor
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
model = AutoModelForAudioClassification.from_pretrained(MODEL_ID)
model.eval()

# Optionally use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

label_names = ["fake", "real"]  # According to your label2id = {"fake": 0, "real": 1}


def classify_audio(audio_file):
    """
    audio_file: path to the uploaded file (WAV, MP3, etc.)
    Returns: "fake" or "real"
    """

    # 2) Load the audio file
    # torchaudio returns (waveform, sample_rate)
    waveform, sr = torchaudio.load(audio_file)

    # If stereo, pick one channel or average
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    waveform = waveform.squeeze()  # (samples,)

    # 3) Resample if needed
    if sr != 16000:
        resampler = T.Resample(sr, 16000)
        waveform = resampler(waveform)
        sr = 16000


    # 3) Preprocess with feature_extractor
    inputs = feature_extractor(
        waveform.numpy(),
        sampling_rate=sr,
        return_tensors="pt",
        truncation=True,
        max_length=int(16000* 6.0),  # 6 second max
    )

    # Move everything to device
    input_values = inputs["input_values"].to(device)

    with torch.no_grad():
        logits = model(input_values).logits
        pred_id = torch.argmax(logits, dim=-1).item()

    # 4) Return label text
    predicted_label = label_names[pred_id]
    return predicted_label


# 5) Build Gradio interface
demo = gr.Interface(
    fn=classify_audio,
    inputs=gr.Audio( type="filepath"),
    outputs="text",
    title="Wav2Vec2 Deepfake Detection",
    description="Upload an audio sample to check if it is fake or real."
)

if __name__ == "__main__":
    demo.launch()
