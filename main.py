import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import tempfile
from helper import parse_invitees
from transcriber import transcribe_audio
from summarizer import summarize
from dotenv import load_dotenv

app = FastAPI()

load_dotenv()

@app.get('/')
async def welcome():
    return{"Hello!": "Test via /docs"}

@app.post("/process-audio")
async def process_audio(file: UploadFile = File(...), invitees_list: str = Form(...)):

    # Error if unsupported file type
    if file.content_type not in ["audio/mpeg", "audio/wav", "audio/x-wav"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}"
        )
    
    # Error if unrecorded invitee

    invitees = parse_invitees(invitees_list)

    missing_samples = []
    for invitee in invitees:
        invitee_folder = os.path.join(os.environ.get('samples_directory'), invitee)
        if not os.path.isdir(invitee_folder) or not os.listdir(invitee_folder):
            missing_samples.append(invitee)

    if missing_samples:
        raise HTTPException(
            status_code=400,
            detail=f"No sample recordings found for: {', '.join(missing_samples)}"
        )
    
    # Moving on if no errors found
    contents = await file.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(contents)
        tmp_path = tmp.name  # Full path to the saved file

    transcript= transcribe_audio(tmp_path, invitees) # run whisper_model = large for better transcription (note that it will take longer!)
    summary, action_items = summarize(transcript)

    return {
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items
    }