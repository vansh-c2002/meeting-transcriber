import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import tempfile
from helper import parse_invitees
from transcriber import transcribe_audio
from summarizer import summarize
from dotenv import load_dotenv
import logging

app = FastAPI()

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

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
    
    invitees = parse_invitees(invitees_list)

    # Error if no invitees
    if not invitees:
        raise HTTPException(
            status_code=400,
            detail = "No invitees detected."
        )
    
    # Error if invitee doesn't have sample
    missing_samples = []
    for invitee in invitees:
        invitee_folder = os.path.join(os.environ.get('samples_directory'), invitee)
        if not os.path.isdir(invitee_folder) or not os.listdir(invitee_folder):
            missing_samples.append(invitee)

    if missing_samples:
        logger.error('Invitee(s) lack sample recording(s).')
        raise HTTPException(
            status_code=400,
            detail=f"No sample recordings found for: {', '.join(missing_samples)}"
        )
    
    # Moving on if no errors found
    contents = await file.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(contents)
        tmp_path = tmp.name  # Full path to the saved file

    try:
        transcript = transcribe_audio(tmp_path, invitees)
    except Exception as e:
        logger.error(f'Failed to transcribe: {e}')
        raise 

    logger.info('Transcription complete! Starting summarization')

    try:
        summary, action_items = summarize(transcript)
    except Exception as e:
        logger.exception('Failed summarization/action item extraction')
        raise HTTPException(
            status_code=400,
            detail=f"Couldn't extract summary/action items"
        )

    return {
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items
    }