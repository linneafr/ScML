from transformers import pipeline
import gradio as gr
import os
import openai

pipe = pipeline(model="TeoJM/whisper-small-se")

import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

models = ["text-curie-001", "text-davinci-002", "text-davinci-003"]
max_tokens = 2048

session_token = os.environ.get('SessionToken')  
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

desc = "Copy paste download link for a news radio programme in Swedish (e.g. from sr.se), summary in other languages than English only available in davinci-003"
    
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


iface = gr.Interface(
    fn=transcribe_to_summary, 
    inputs=[gr.Textbox(placeholder="Insert url ending with .mp3 here"),
           gr.Radio(languages, value="English"),
           gr.Radio(models, value="text-davinci-003")],
    outputs="text",
    title="Summary of Swedish news using fine-tuned whisper-small",
    description=desc
)


iface.launch()