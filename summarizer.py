import os
from dotenv import load_dotenv
from groq import Groq
import logging

load_dotenv()

logger = logging.getLogger(__name__)

def summarize(final_transcript):
    
    try: 
        client = Groq(
            api_key=os.environ.get('groq'),
        )
        logger.info('Found Groq API key')
    except Exception as e:
        logger.exception(f'Failed to load API key: {e}')
        raise

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": f"""
        You are an assistant that summarizes meetings and also extracts action items. Do not create action items if there were none in the meeting. \n
        Critically think about what work was assigned to whom. 

        For the following transcript, do the following:
        1. Provide a summary of the discussion.
        2. List action items, and assign each to an invitee (with a deadline if applicable).
        Output format:
        - Summary:
        ...
        - Action Items:
        - [Name]: [Action item]

        Here's the transcript:
        {final_transcript}
        """
                }
            ],
            model="llama-3.3-70b-versatile",
        )
        logger.info('Summary and action item extraction complete')
    except Exception as e:
        logger.error('Could not communicate with Groq')

    output = chat_completion.choices[0].message.content

    # Separating one output into two 
    summary = output.split('- Action Items:')[0].replace('- Summary:', '').strip()
    action_items = output.split('- Action Items:')[1].strip()

    return summary, action_items