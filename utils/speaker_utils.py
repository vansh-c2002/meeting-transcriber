import os
import torchaudio
import tqdm
from pydub import AudioSegment
import torchaudio
from io import BytesIO
import tempfile
import torch
import logging
import torch.nn.functional as F
from pathlib import Path
from speechbrain.inference.speaker import SpeakerRecognition


logger = logging.getLogger(__name__)

# Load ECAPA model
spkrec = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="data/pretrained_models/spkrec"
)

def normalize_and_convert_to_mono(audio_path):
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_channels(1).set_frame_rate(16000) # mono and 16khz (speaker dientification pipeline expects 16k)

    # Normalizing volume
    normalized = audio.apply_gain(-20-audio.dbFS)

    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    normalized.export(temp.name, format="wav")
    return temp.name


def extract_embedding(audio_path, embedding_path):
    if os.path.exists(embedding_path):
        logger.info(f"Skipping (exists): {embedding_path}")
        return

    temp_path = None
    try:
        # Normalize vol, convert to mono and downsample to 16kHz
        temp_path = normalize_and_convert_to_mono(audio_path)

        signal, _ = torchaudio.load(temp_path)

        # Extract embedding from normalized temp file
        embedding = spkrec.encode_batch(signal).squeeze().detach().cpu()

        os.makedirs(os.path.dirname(embedding_path), exist_ok=True)
        torch.save(embedding, embedding_path)
        logging.info(f"Saved: {embedding_path}")

    except Exception as e:
        logging.exception(f"Error processing {audio_path}: {e}")

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)  # Cleanup temp file


def process_samples_folder(samples_dir, embeddings_dir):
    samples_dir = Path(samples_dir)
    embeddings_dir = Path(embeddings_dir)

    for audio_file in samples_dir.rglob("*"):
        if audio_file.suffix.lower() not in ['.wav', '.flac', '.mp3', '.m4a', '.aac', '.ogg']:
            continue

        rel_path = audio_file.relative_to(samples_dir)
        embedding_path = embeddings_dir / rel_path.with_suffix(".pt")

        extract_embedding(str(audio_file), embedding_path)

def embedding_matrix(embedding_root, invitees):
    
    embedding_root = Path(embedding_root)
    embeddings = []
    speakers = []
    for folder in embedding_root.iterdir():
        if folder.name in invitees:
            for file in folder.rglob('*.pt'):
                embeddings.append(torch.load(file))
                speakers.append(folder.name)

    matrix = torch.stack(embeddings)
    return matrix, speakers


def get_speaker_from_matrix(segment_embedding, embedding_matrix, speakers):
    similarity = F.cosine_similarity(segment_embedding.unsqueeze(0), embedding_matrix)
    return speakers[torch.argmax(similarity).item()]

def get_segment_embedding(audio_file, segment_start, segment_end):
    audio = AudioSegment.from_file(audio_file)
    audio = audio[segment_start*1000:segment_end*1000]

    if len(audio) < 200:
        padding = AudioSegment.silent(duration=50)
        audio = padding + audio + padding

    audio = audio.set_channels(1).set_frame_rate(16000) # mono and 16khz (speaker identification pipeline expects 16k)
    normalized = audio.apply_gain(-20.0 - audio.dBFS)

    buffer = BytesIO()
    normalized.export(buffer, format="wav")
    buffer.seek(0)

    signal, _ = torchaudio.load(buffer)
    embedding = spkrec.encode_batch(signal).squeeze().detach().cpu()
    return embedding