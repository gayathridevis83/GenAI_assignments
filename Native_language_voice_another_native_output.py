# create an a program that takes native language voice as input and replies in another native language voice

!pip install -q openai-whisper deep-translator gtts ffmpeg-python
!apt-get -qq install ffmpeg

import whisper
print("Loading model...")
model = whisper.load_model("base")
print("Model loaded")

from google.colab import files

!pip install gtts

from gtts import gTTS

tamil_text = "??????? ?????? ??????????????."

tts = gTTS(text=tamil_text, lang='ta')

tts.save("tamil_sample.mp3")

print("Tamil sample audio created: tamil_sample.mp3")

from IPython.display import Audio

Audio("tamil_sample.mp3")

from google.colab import files

files.download("tamil_sample.mp3")


uploaded = files.upload()
audio_file = list(uploaded.keys())[0]

print(audio_file)


result = model.transcribe(audio_file)

print(result["language"])
print(result["text"])

from deep_translator import GoogleTranslator

translated = GoogleTranslator(
    source="auto",
    target="en"
).translate(result["text"])

print(translated)

from gtts import gTTS
from IPython.display import Audio

tts = gTTS(translated, lang="en")
tts.save("output.mp3")

Audio("output.mp3")
