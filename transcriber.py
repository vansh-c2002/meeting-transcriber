from utils.speaker_utils import get_segment_embedding, embedding_matrix, get_speaker_from_matrix, process_samples_folder
from utils.transcription_utils import combine_segments 
from dotenv import load_dotenv
import os
import whisper
import logging

# device = "cpu"
# audio_file = "/Users/vanshc/Desktop/delete.mp3"
# batch_size = 4 # increase for higher speed
# compute_type = "int8" # change to "int8" if low on GPU mem (may reduce accuracy)
# invitees_list = check format in readme.md

logger = logging.getLogger(__name__)

load_dotenv()

def transcribe_audio(audio_file, invitees, whisper_model = os.environ.get('whisper_model'), device='cpu', batch_size=4, compute_type='int8'):
    
    # This is not optimized for GPUs yet

    num_speakers = len(invitees)
    logger.info(f'{num_speakers} speakers detected.')

    samples_directory = os.environ.get('samples_directory')
    embeddings_directory = os.environ.get('embeddings_directory')
    logger.info(f'Embeddings will be uploaded to {embeddings_directory}')

    # Compute embeddings for all samples
    process_samples_folder(samples_directory, embeddings_directory)
    logger.info('Extracted embeddings')

    # Use Whisper for transcription
    logger.info('Loading transcription model')
    try:
        model = whisper.load_model(whisper_model)
        logger.info('Successfully loaded Whisper!')
    except Exception as e:
        logger.exception('Failed to load Whisper')

    result = model.transcribe(audio_file, word_timestamps=True)
    logger.info('Transcription finished!')

    segments = result['segments']

    matrix, speakers = embedding_matrix(embeddings_directory, invitees) # TODO: Have an intial matrix and speakers list on startup, and select
                                                                        # speakers from those, instead of recomputing
    # Replace speaker names
    for segment in segments:
        embedding = get_segment_embedding(audio_file, segment['start'], segment['end'])
        segment['speaker'] = get_speaker_from_matrix(embedding, matrix, speakers)

    segments = combine_segments(segments)  # TODO: for slightly higher efficiency for slightly lower accuracy, move this line above segments = reuslt['segments']

    # Creating transcript
    transcript = []

    for i in segments:
        transcript.append(f"{i['speaker']}: {i['text'].strip()}")

    return transcript