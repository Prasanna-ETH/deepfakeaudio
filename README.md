# Sigma One - Deepfake Audio Detection  

🔍 **Project Overview**  
Sigma One is an AI-powered Deepfake Audio Detection system. It allows users to upload an audio file and determine whether it is "real" or "fake" using a pre-trained AI model. The model is built using state-of-the-art deep learning techniques and hosted with a simple **Gradio** interface for easy access.  

---

## 🚀 Technologies Used & Their Purpose  

| Technology                  | Purpose |
|-----------------------------|---------|
| **Python**                  | Main programming language for the project. |
| **PyTorch**                 | Deep learning framework used for loading the model and making predictions. |
| **Hugging Face Transformers** | Provides pre-trained deep learning models for audio classification. |
| **Torchaudio**              | Library for loading and processing audio files. |
| **Gradio**                  | Creates a user-friendly web interface for uploading and classifying audio files. |

---

## 📌 Breakdown of the Code & Its Purpose  

### 1️⃣ Model Selection & Loading  

```python
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

MODEL_ID = "Zeyadd-Mostaffa/wav2vec_checkpoints"
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
model = AutoModelForAudioClassification.from_pretrained(MODEL_ID)
model.eval()
```
- **feature_extractor**: Converts raw audio into a format suitable for deep learning.  
- **model**: Loads the pre-trained deepfake detection model.  
- **model.eval()**: Puts the model in evaluation mode, meaning it won’t learn new data but only make predictions.  

---

### 2️⃣ Audio Processing  

```python
import torchaudio.transforms as T
waveform, sr = torchaudio.load(audio_file)

if sr != 16000:
    resampler = T.Resample(sr, 16000)
    waveform = resampler(waveform)
```
- **Torchaudio** loads audio files (WAV, MP3, etc.).  
- **waveform**: The audio data in numerical form.  
- **sr**: Sample rate (how many times per second the audio is sampled).  
- **Resampling** ensures the audio matches the model’s expected 16,000 Hz sample rate.  

---

### 3️⃣ Feature Extraction & Prediction  

```python
inputs = feature_extractor(
    waveform.numpy(),
    sampling_rate=sr,
    return_tensors="pt",
    truncation=True,
    max_length=int(16000 * 6.0),  # 6-second max
)

with torch.no_grad():
    logits = model(inputs.input_values).logits
    pred_id = torch.argmax(logits, dim=-1).item()
```
- Converts the audio into numerical features that the model understands.  
- Truncates input to **6 seconds** to keep it within model limits.  
- **Predicts** whether the audio is fake or real using the highest probability label.  

---

### 4️⃣ User Interface with Gradio  

```python
import gradio as gr

def classify_audio(audio_file):
    # Function that processes and classifies audio

demo = gr.Interface(
    fn=classify_audio,
    inputs=gr.Audio(type="filepath"),
    outputs="text",
    title="Sigma One - Deepfake Audio Detection",
    description="Upload an audio sample to check if it is fake or real."
)

if __name__ == "__main__":
    demo.launch()
```
- **Gradio** provides a simple web interface.  
- Users can **upload an audio file** and get a classification result.  
- The model predicts whether the **audio is real or fake**.  

---

## 🎯 Why This Project is Useful?  

✅ **Detects Deepfake Audio**: Helps prevent misinformation by identifying fake voices.  
✅ **AI-Powered**: Uses a pre-trained deep learning model for high accuracy.  
✅ **Easy to Use**: No programming knowledge needed – just upload an audio file!  
✅ **Fast & Lightweight**: Runs efficiently on both **CPU & GPU**.  

---

## 🛠️ How to Use?  

1️⃣ **Clone the repository**  

```sh
git clone https://github.com/Prasanna-ETH/deepfakeaudio.git
```

2️⃣ **Install dependencies**  

```sh
pip install torch torchaudio transformers gradio
```

3️⃣ **Run the project**  

```sh
python app.py
```

4️⃣ **Open the Gradio interface and upload an audio file** to check if it’s fake or real!  

---


🔗 **Check out the project**: [GitHub Repository](https://github.com/Prasanna-ETH/deepfakeaudio)  


📌 **Developed by** Prasanna-ETH 🎯  
