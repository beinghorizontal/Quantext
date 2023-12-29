import torch
from TTS.api import TTS

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# List available 🐸TTS models
print(TTS().list_models())

# Init TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# Run TTS
# ❗ Since this model is multi-lingual voice cloning model, we must set the target speaker_wav and language
# Text to speech list of amplitude values as output
#wav = tts.tts(text="Hello world!", speaker_wav="my/cloning/audio.wav", language="en")
# Text to speech to a file
tts.tts_to_file(text="It is a capital mistake to theorize before one has data. Insensibly one begins to twist "
                     "facts to suit theories, instead of theories to suit facts",
                speaker_wav=r"D:\demos\ttstest.wav", language="en", file_path="d:/demos/sherlock_output.wav")

tts.tts_to_file(text="जिंदगी में अगर सफल होना है तो अच्छे दिमाग के साथ तंदरुस्त भी रहना जरुरी है. और आप देख रहे है मेरा यूट्यूब चैनल कॉन्टेक्स्ट",
                speaker_wav=r"D:\demos\amitabh_clean_voice.wav", language="hi", file_path="d:/demos/amitabh_output.wav")
