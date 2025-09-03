from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import tempfile
from transcriber import transcribe_audio
from summarizer import summarize

app = FastAPI()

@app.get('/')
async def welcome():
    return{"Hello!": "Test via /docs"}

@app.post("/process-audio")
async def process_audio(file: UploadFile = File(...), invitees_list: str = Form(...)):

    if file.content_type not in ["audio/mpeg", "audio/wav", "audio/x-wav"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}"
        )
    
    # TODO: Raise error if you don't have embeddings for a particular invitee

    contents = await file.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(contents)
        tmp_path = tmp.name  # Full path to the saved file

    transcript= transcribe_audio(tmp_path, invitees_list) # run whisper_model = large for better transcription (note that it will take longer!)
    summary, action_items = summarize(transcript)

    return {
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items
    }