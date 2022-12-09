from transformers import pipeline
import gradio as gr
from pytube import YouTube

import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

models = ["text-curie-001", "text-davinci-002", "text-davinci-003"]
max_tokens = 2048

session_token = os.environ.get('SessionToken')


pipe = pipeline(model="sanchit-gandhi/whisper-small-hi")  # change to "your-username/the-name-you-picked"


desc = "Copy paste download link for a news radio programme in Swedish (e.g. from sr.se), summary in other languages than English only available in davinci-003"

def transcribe(audio):
    text = pipe(audio)["text"]
    return text


def get_audio_from_youtube(url):
    streams = YouTube(url).streams.filter(only_audio=True, file_extension='mp4')
    path = streams.first().download()

    text = transcribe(path)
    return text



def transcribe_to_summary(url, language, model):
    text = pipe(url, chunk_length_s=20)["text"]
    prompt = 'The following is a transcription from a Swedish news program: "'
    prompt += text
    prompt += '"'+" " + "summarize the main news stories briefly using bullet points in "
    prompt += language

    response = openai.Completion.create(
    engine=model,
    prompt=prompt,
    temperature=0.1,
    max_tokens=max_tokens,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    completed_text = response.choices[0].text
    completed_text = completed_text.replace(prompt, "")
    return completed_text



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
  with gr.TabItem("Transcribe news from radio program"):
        url_radio = gr.Textbox(placeholder="Insert url ending with .mp3 here")
        button_radio = gr.Button("Submit news program")
        outputs_radio = [
            gr.Textbox(label="Outputs")
        ]
        description=desc
        languages = ["Swedish", "English", "Mandarin Chinese", "Hindi", "Spanish", "Arabic",
             "Bengali", "Russian", "Portuguese", "Japanese", "German",
             "Javanese", "Wu Chinese", "Korean", "French", "Vietnamese",
             "Telugu", "Marathi", "Tamil", "Turkish", "Urdu",
             "Italian", "Cantonese", "Thai", "Gujarati", "Jin Chinese",
             "Min Nan Chinese", "Persian", "Polish", "Punjabi", "Romanian",
             "Ukrainian", "Dutch", "Kannada", "Malayalam", "Oriya",
             "Serbo-Croatian", "Sundanese", "Czech", "Hausa",
             "Swahili", "Norwegian", "Finnish", "Hungarian", "Lithuanian",
             "Latvian", "Estonian", "Slovenian", "Bulgarian", "Danish",
             "Slovak", "Macedonian", "Georgian", "Armenian", "Azerbaijani"]

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
  button_radio.click(
      fn=transcribe_to_summary,
      inputs=[url_radio, gr.Radio(languages, value="English"),
           gr.Radio(models, value="text-davinci-003")],
      outputs=outputs_radio
  )



iface.launch()
