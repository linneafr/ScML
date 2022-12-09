from transformers import pipeline
import gradio as gr
from pytube import YouTube

pipe = pipeline(model="TeoJM/whisper-small-se")  # change to "your-username/the-name-you-picked"




def transcribe(audio):
    text = pipe(audio)["text"]
    return text


def get_audio_from_youtube(url):
    streams = YouTube(url).streams.filter(only_audio=True, file_extension='mp4')
    path = streams.first().download()

    text = transcribe(path)
    return text



with gr.Blocks() as iface:
  with gr.TabItem("Record from microphone"):
        filefrom_mic = gr.Audio(source="microphone", type="filepath")
        button_mic = gr.Button("Submit audio")
        outputs_mic = [
            gr.Textbox(label="Outputs"),
        ]
  with gr.TabItem("Transcribe from a Youtube video"):
        url = gr.Text(max_lines=1, label="Insert YouTube URL", value="https://www.youtube.com/watch?v=db2cCDUJBNk&t=1s")
        button_yt = gr.Button("Submit video")
        outputs_yt = [
            gr.Textbox(label="Outputs")
        ]    
  button_mic.click(
        fn=transcribe,
        inputs=filefrom_mic,
        outputs=outputs_mic,
    )
  button_yt.click(
    fn=get_audio_from_youtube,
    inputs=url,
    outputs=outputs_yt,
    )



iface.launch()
