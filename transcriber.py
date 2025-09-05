from helper import get_segment_embedding, embedding_matrix, get_speaker_from_matrix, combine_segments, process_samples_folder, parse_invitees
from dotenv import load_dotenv
import os
import whisper

# device = "cpu"
# audio_file = "/Users/vanshc/Desktop/delete.mp3"
# batch_size = 4 # increase for higher speed
# compute_type = "int8" # change to "int8" if low on GPU mem (may reduce accuracy)
# invitees_list = 'Full name <don@gmail.com>, Other name <isaac@cursor.ai>, Some Thing <mo@yale.edu>'

load_dotenv()

def transcribe_audio(audio_file, invitees, whisper_model = os.environ.get('WHISPER_MODEL'), device='cpu', batch_size=4, compute_type='int8'):

    num_speakers = len(invitees)

    # Compute embeddings for all samples
    process_samples_folder('./data/samples/', './data/embeddings/')

    # Use Whisper for transcription
    model = whisper.load_model(whisper_model)
    result = model.transcribe(audio_file, word_timestamps=True)

    segments = result['segments']

    matrix, speakers = embedding_matrix("./data/embeddings/", invitees)    # TODO: Have an intial matrix and speakers list on startup, and select
                                                                    # speakers from those, instead of recomputing
    # Replace speaker names
    for segment in segments:
        embedding = get_segment_embedding(audio_file, segment['start'], segment['end'])
        segment['speaker'] = get_speaker_from_matrix(embedding, matrix, speakers)

    combined_segments = combine_segments(segments) #TODO: for higher efficiency, don't run this, replace combined_segments with segments lower

    # Creating transcript
    transcript = []

    for i in combined_segments:
        transcript.append(f"{i['speaker']}: {i['text'].strip()}")

    return transcript