import os
import re
from pydub import AudioSegment
import tempfile
import torch
import tqdm
from speechbrain.inference.speaker import SpeakerRecognition
from pathlib import Path
import torchaudio
import torch.nn.functional as F
from io import BytesIO

# Load ECAPA model
spkrec = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec"
)

def normalize_and_convert_to_mono(audio_path):
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_channels(1).set_frame_rate(16000) # mono and 16khz (speaker dientification pipeline expects 16k)

    # Normalizing volume
    target_dBFS = -20.0
    change_dBFS = target_dBFS - audio.dBFS
    normalized = audio.apply_gain(change_dBFS)

    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    normalized.export(temp.name, format="wav")
    return temp.name

def parse_invitees(invitees_list: str):
    list_of_invitees = re.findall(r'<([^<>@\s]+@[^<>@\s]+\.[^<>@\s]+)>', invitees_list)
    return list_of_invitees


def extract_embedding(audio_path, embedding_path):
    if os.path.exists(embedding_path):
        print(f"Skipping (exists): {embedding_path}")
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
        print(f"Saved: {embedding_path}")

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)  # Cleanup temp file

def process_samples_folder(samples_dir, embeddings_dir):
    samples_dir = Path(samples_dir)
    embeddings_dir = Path(embeddings_dir)

    for audio_file in tqdm.tqdm(samples_dir.rglob("*")):
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

def combine_segments(segments):
    combined_segments = []
    previous_segment = None

    for segment in segments:
        # Remove 'words' key
        segment_data = {
            'start': segment['start'],
            'end': segment['end'],
            'text': segment['text'],
            'speaker': segment['speaker']
        }
        
        if previous_segment is None:
            previous_segment = segment_data
        elif segment['speaker'] == previous_segment['speaker']:
            # Same speaker: combine
            previous_segment['text'] += ' ' + segment['text']
            previous_segment['end'] = segment['end']
        else:
            # Different speaker: save previous and start new
            combined_segments.append(previous_segment)
            previous_segment = segment_data

    # Add the last one
    if previous_segment is not None:
        combined_segments.append(previous_segment)
    return combined_segments