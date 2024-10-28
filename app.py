import warnings
warnings.filterwarnings("ignore")
from flask import Flask, render_template, request, url_for, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from summarizer import Summarizer
import moviepy.editor as mp
from time import sleep
import os
import requests
from googletrans import Translator
from langdetect import detect
from fpdf import FPDF  # Import the fpdf module
from flask import send_file  # Import send_file to allow file downloads

# Ensure to include your existing imports at the top
app = Flask(__name__)

# Load models
def load_models():
    global model
    global tokenizer
    global bert_model
    model = AutoModelForSeq2SeqLM.from_pretrained('t5-base')
    tokenizer = AutoTokenizer.from_pretrained('t5-base')
    bert_model = Summarizer()

# AssemblyAI API settings
API_key = "057a0adabc0e4d25ba5e9db2d155c8b0"
headers = {'authorization': API_key, 'content-type': 'application/json'}
upload_endpoint = 'https://api.assemblyai.com/v2/upload'
transcription_endpoint = "https://api.assemblyai.com/v2/transcript"
translator = Translator()

# Ensure the uploads directory exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Functions
def translate_text(text, target_lang):
    """Function to translate text into the target language using Googletrans."""
    try:
        translation = translator.translate(text, dest=target_lang)
        return translation.text
    except Exception as e:
        #return f"Translation error: {str(e)}"
        return f"Could not translate to {target_lang}. Showing summary in English:\n\n" + text

def detect_language(text):
    """Detect the language of the given text."""
    try:
        return detect(text)
    except Exception as e:
        return "Language detection error: " + str(e)

def upload(file_path):
    """Upload audio/video file to AssemblyAI API."""
    with open(file_path, 'rb') as f:
        response = requests.post(upload_endpoint, headers=headers, data=f)

    if response.status_code == 200:
        return response.json()['upload_url']
    else:
        raise Exception(f"Error uploading file: {response.status_code}, {response.text}")

def transcribe(upload_url):
    """Send the uploaded file for transcription to AssemblyAI."""
    transcript_request = {'audio_url': upload_url}
    response = requests.post(transcription_endpoint, json=transcript_request, headers=headers)
    
    if response.status_code == 200:
        return response.json()['id']
    else:
        raise Exception(f"Error starting transcription: {response.status_code}, {response.text}")

def get_result(transcription_id):
    """Get the result of the transcription from AssemblyAI."""
    polling_endpoint = f"{transcription_endpoint}/{transcription_id}"
    
    # Polling for the transcription result
    while True:
        response = requests.get(polling_endpoint, headers=headers)
        result = response.json()

        status = result.get('status', None)  # Use get to avoid KeyError

        if status == 'completed':
            return result
        elif status == 'failed':
            raise Exception(f"Transcription failed: {result}")

        # Wait before trying again
        sleep(5)


ALLOWED_EXTENSIONS = {'mp3', 'wav', 'flac', 'aac', 'mp4', 'avi', 'mov', 'mkv'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Routes
@app.route("/download_text", methods=['GET'])
def download_text():
    """Download summary as a text file."""
    summary = request.args.get('summary', default='', type=str)
    
    # Create a text file and write the summary into it
    file_path = "uploads/summary.txt"
    with open(file_path, 'w') as file:
        file.write(summary)
    
    return send_file(file_path, as_attachment=True)

@app.route("/download_pdf", methods=['GET'])
def download_pdf():
    """Download summary as a PDF file."""
    summary = request.args.get('summary', default='', type=str)

    # Create a PDF object
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Add a cell with the summary text
    pdf.multi_cell(0, 10, summary)
    
    # Save the PDF to a file
    pdf_file_path = "uploads/summary.pdf"
    pdf.output(pdf_file_path)

    return send_file(pdf_file_path, as_attachment=True)
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/options")
def options():
    return render_template("options.html")

@app.route("/text_input")
def text_input():
    return render_template("text_input.html")

@app.route("/audio_input")
def audio_input():
    return render_template("audio_input.html")

@app.route("/result")
def result():
    # Get the summary and extended summary from the query parameters
    summary = request.args.get('summary', default='', type=str)
    ext_summary = request.args.get('ext_summary', default='', type=str)
    
    # Render the result page with the summary and extended summary
    return render_template('result.html', summary=summary, ext_summary=ext_summary)

@app.route("/loading", methods=['POST'])
def loading():
    if request.form.get('text'):  # Text summarization
        form_data = {
            'url': url_for('summarize_text'),
            'data': request.form.to_dict(),
            'contentType': 'application/x-www-form-urlencoded',
            'processData': True
        }

    return render_template('loading.html', form_data=form_data)

@app.route("/summarize_text", methods=['POST'])
def summarize_text():
    text = request.form['text']
    target_language = request.form["lang1"]  # The language the user wants the summary in

    # Step 1: Detect the input language
    detected_language = detect_language(text)

    # Step 2: Translate to English if the detected language is not English
    if detected_language != 'en':
        text = translate_text(text, 'en')

    # Step 3: Summarize the text (in English)
    if len(text.split()) < 10:
        summary = text  # If text is too short, return as-is
    else:
        tokens_input = tokenizer.encode("summarize: " + text, return_tensors='pt', max_length=512, truncation=True)
        summary_ids = model.generate(tokens_input, min_length=20, max_length=150, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # Step 4: Translate the summary back to the target language (if necessary)
    if target_language != 'en':
        summary = translate_text(summary, target_language)

    # Generate extended summary using BERT
    ext_summary = bert_model(text, ratio=0.5)
    if target_language != 'en':
        ext_summary = translate_text(ext_summary, target_language)

    return jsonify(summary=summary, ext_summary=ext_summary)

@app.route("/summarize_audio", methods=['POST'])
def summarize_audio():
    file = request.files['file']
    
    # Check if the uploaded file has a valid format
    if not allowed_file(file.filename):
        return render_template("error.html", message="Unsupported file format! Please upload a valid audio or video file.")
    
    file_path = f"uploads/{file.filename}"
    file.save(file_path)

    try:
        # Prepare form data to be passed to the loading page for AJAX processing
        form_data = {
            'url': url_for('process_audio'),
            'data': {
                'file': file.filename,
                'lang1': request.form['lang1']
            },
            'contentType': 'application/json',
            'processData': False
        }

        # Redirect to the loading page
        return render_template('loading_audio.html', form_data=form_data)

    except Exception as e:
        return render_template("error.html", message=f"An error occurred: {str(e)}")

# New route for processing audio in the background (AJAX call)
@app.route("/process_audio", methods=['POST'])
def process_audio():
    try:
        file_name = request.json.get('file')
        target_language = request.json.get('lang1')  # The language the user wants the summary in
        file_path = f"uploads/{file_name}"

        if file_path.endswith(('.mp3', '.wav', '.flac', '.aac')):
            upload_url = upload(file_path)
        else:
            # Convert video to audio if necessary
            clip = mp.VideoFileClip(file_path)
            audio_file_path = "uploads/converted_audio.mp3"
            clip.audio.write_audiofile(audio_file_path)
            upload_url = upload(audio_file_path)

        transcription_id = transcribe(upload_url)
        response = get_result(transcription_id)

        # Get the transcription result
        transcript = response.get("text", "")[:5000000]  # Limit for demonstration purposes

        # Step 3: Detect the language of the transcription
        detected_language = detect_language(transcript)

        # Step 4: Translate the transcript to English if it's not in English
        if detected_language != 'en':
            transcript = translate_text(transcript, 'en')

        # Step 5: Summarize the transcription (in English)
        summary = transcript[:500]  # Summarization logic
        headline = "No headline available"
        if 'chapters' in response and response['chapters']:
            headline = response['chapters'][0].get('headline', "No headline")

        # Step 6: Translate the summary back to the requested language if necessary
        if target_language != 'en':
            summary = translate_text(summary, target_language)
            headline = translate_text(headline, target_language)

        return jsonify({'summary': summary, 'headline': headline})

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route("/result_audio")
def result_audio():
    summary = request.args.get('summary', default='', type=str)
    headline = request.args.get('headline', default='', type=str)

    return render_template("result2.html", summary=summary, headline=headline)

if __name__ == '__main__':
    load_models()
    app.run(port=8080, debug=True)
