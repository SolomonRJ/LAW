import os
import io
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import vertexai
from vertexai.generative_models import GenerativeModel, SafetySetting
from google.cloud import speech, translate_v2 as translate, texttospeech
from pydub import AudioSegment

# Set Google Cloud authentication
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "services_account.json"

# Google Cloud Project Configuration
PROJECT_ID = "text-to-speech-452109"
LOCATION = "us-central1"

# Initialize FastAPI
app = FastAPI()

# Initialize Google Clients
speech_client = speech.SpeechClient()
translate_client = translate.Client()
tts_client = texttospeech.TextToSpeechClient()

# Initialize Vertex AI
vertexai.init(
    project=PROJECT_ID,
    location=LOCATION,
    api_endpoint="us-central1-aiplatform.googleapis.com"    
)

# Safety settings (Optional)
safety_settings = [
    SafetySetting(category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=SafetySetting.HarmBlockThreshold.OFF),
    SafetySetting(category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=SafetySetting.HarmBlockThreshold.OFF),
    SafetySetting(category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=SafetySetting.HarmBlockThreshold.OFF),
    SafetySetting(category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=SafetySetting.HarmBlockThreshold.OFF),
]

gemini_model = GenerativeModel("gemini-1.5-pro-002")

def speech_to_text(audio_bytes, language_code="kn-IN"):
    audio = speech.RecognitionAudio(content=audio_bytes)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=48000,
        language_code=language_code,
    )
    response = speech_client.recognize(config=config, audio=audio)
    return response.results[0].alternatives[0].transcript if response.results else None

def translate_text(text, target_language="en"):
    translation = translate_client.translate(text, target_language=target_language)
    return translation["translatedText"]

def process_with_gemini(input_text):
    template = f"""
    You are a legal adviser helping users with their problems.
    Keep the response under 5000 bytes, provide clear and simple guidance,
    reference relevant laws and amendments, and communicate in a caring manner.
    Respond in a single paragraph with a conversational tone.
    
    Legal Question:
    "{input_text}"
    """
    chat = gemini_model.start_chat()
    response = chat.send_message(template)
    return response.text

def text_to_speech(text, output_file="output.mp3", language_code="kn-IN"):
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code=language_code,
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL,
    )
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
    response = tts_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
    with open(output_file, "wb") as out:
        out.write(response.audio_content)
    return output_file

@app.post("/process-audio/")
async def process_audio(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    text = speech_to_text(audio_bytes)
    if not text:
        return {"error": "No speech detected!"}
    
    translated_text = translate_text(text, "en")
    legal_response = process_with_gemini(translated_text)
    final_translation = translate_text(legal_response, "kn")
    output_audio_path = text_to_speech(final_translation)
    
    return FileResponse(output_audio_path, media_type="audio/mpeg", filename="translated_audio.mp3")