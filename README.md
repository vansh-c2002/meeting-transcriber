# Backend Meeting Transcriber

This is a backend API for transcribing in-person meetings with speaker labeling, summarizing them and extracting action items from them. It takes meeting audio and email IDs of attendees, then splits the meeting into segments, and compares each segment to pre-enrolled speaker voice embeddings to generate a multi-speaker transcription.

> **Note:** The `data/` folder contains sensitive employee data and is excluded from this repository. Placeholder `.gitkeep` files are present to maintain folder structure.

---

## Setup Instructions

1. **Python Version**

This project uses Python 3.12.3. We recommend using [`pyenv`](https://github.com/pyenv/pyenv) to manage your Python versions.

To install Python 3.12.3 and set it locally:

```bash
brew install pyenv #if you don't have pyenv
pyenv install 3.12.3
pyenv local 3.12.3
```
Check via
```bash 
pyenv version 
```
This should return 3.12.3 (set by .../backend-meeting-transcriber/.python-version)

2. **Virtual Environment**

Once you've set your directory to backend-meeting-transcriber,
create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```
4. **Enter data**

Insert 2 10-15s voice samples from each employee in data/samples like this

```
data/
├── samples/
│ ├── sam@organization.com
│ │ ├── sam-1.wav
│ │ └── sam-2.wav
│ └── jess@organization.com
│   ├── jess-1.wav
│   └── jess-2.wav
│
├── embeddings/ # Auto-generated!
│ ├── sam@organization.com
│ │ ├── sam-1.wav
│ │ └── sam-2.wav
│ └── jess@organization.com
│   ├── jess-1.wav
│   └── jess-2.wav
```
5. **Prepare .env file**

```bash
cp .env.example .env
```
Input your [Groq](https://console.groq.com/keys) API key in the .env file.

6. **Launch the API**

To start the transcription API, run:

```bash
uvicorn main:app --reload
```

7. **Input the audio and list of attendees**

You can copy paste from Google Calendar.

Maya Thompson <<mayat@spotify.com>>, Elias Navarro <<eliasn@spotify.com>>, Zoe Chen <<zoec@spotify.com>>, Leo Armstrong <<leoa@spotify.com>>, Aria Delgado <<ariad@espotify.com>>, Kai Matsuda <<kaim@espotify.com>>, Vansh Chugh <<vanshc@spotify.com>>

OR

<<mayat@spotify.com>> <<eliasn@spotify.com>> <<zoec@spotify.com>> <<leoa@spotify.com>> <<ariad@espotify.com>> <<kaim@espotify.com>> <<vanshc@spotify.com>>

OR even 

mayat@spotify.com, eliasn@spotify.com, zoec@spotify.com, leoa@spotify.com, ariad@espotify.com, kaim@espotify.com, vanshc@spotify.com

Method uses regex; looks for @ and . to parse a list of email IDs that match - person@company.name

## To-do List

Replace Whisper with [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper) for faster processing.