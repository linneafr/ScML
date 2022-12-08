from transformers import pipeline
import gradio as gr

pipe = pipeline(model="TeoJM/whisper-small-se")

def transcribe(audio):
    text = pipe(audio)["text"]
    return text

iface = gr.Interface(
    fn=transcribe, 
    inputs=gr.Audio(source="microphone", type="filepath"), 
    outputs="text",
    title="Swedish Transcription using Whisper",
    description="Demo for Swedish ASR using a fine-tuned Whisper small model.",
)


iface.launch()