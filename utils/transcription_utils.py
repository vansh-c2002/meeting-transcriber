
import re
import logging

logger = logging.getLogger(__name__)

def parse_invitees(invitees_list: str):
    list_of_invitees = re.findall(r'([^<>@\s]+@[^<>@\s]+\.[^<>@\s]+)', invitees_list)
    return list_of_invitees

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
            previous_segment['text'] += segment['text']
            previous_segment['end'] = segment['end']
        else:
            # Different speaker: save previous and start new
            combined_segments.append(previous_segment)
            previous_segment = segment_data

    # Add the last one
    if previous_segment is not None:
        combined_segments.append(previous_segment)
    return combined_segments