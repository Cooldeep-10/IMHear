import os
import time
import requests
import numpy as np
import whisper
import torchaudio


# ============================================================
# NUMPY COMPATIBILITY
# ============================================================

# Older PyAnnote versions use np.NaN.
# np.NaN was removed in NumPy 2.0.
if not hasattr(np, "NaN"):
    np.NaN = np.nan


# ============================================================
# TORCHAUDIO COMPATIBILITY
# ============================================================

if not hasattr(torchaudio, "set_audio_backend"):
    torchaudio.set_audio_backend = lambda *args, **kwargs: None

if not hasattr(torchaudio, "get_audio_backend"):
    torchaudio.get_audio_backend = lambda: "soundfile"


# IMPORTANT:
# Import PyAnnote AFTER the NumPy compatibility fix.
from pyannote.audio import Pipeline


# ============================================================
# CONFIGURATION
# ============================================================

# WAV file located next to this Python file
AUDIO_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "live_recording.wav"
)


# ============================================================
# WHISPER

WHISPER_MODEL = "large-v3"
# "en" = English
WHISPER_LANGUAGE = None


HF_TOKEN = os.getenv("            ")

# ============================================================
# OLLAMA

OLLAMA_URL = "      "
OLLAMA_MODEL = "gemma3:4b"


# ============================================================
# CHECK AUDIO FILE

print("=" * 60)
print("🎵 CHECKING SAVED AUDIO")
print("=" * 60)

if not os.path.exists(AUDIO_FILE):

    print("❌ Audio file not found!")
    print()
    print("Expected location:")
    print(AUDIO_FILE)
    print()
    print("Make sure live_recording.wav is inside the Analysis folder.")

    exit()


print("✅ Audio file found:")
print(AUDIO_FILE)


# ============================================================
# HELPER FUNCTION

def find_speaker(start_time, end_time, diarization):
    """
    Finds which speaker has the largest amount
    of overlapping time with a Whisper segment.
    """

    speaker_overlap = {}

    for turn, _, speaker in diarization.itertracks(
        yield_label=True
    ):

        overlap_start = max(
            start_time,
            turn.start
        )

        overlap_end = min(
            end_time,
            turn.end
        )

        overlap = max(
            0,
            overlap_end - overlap_start
        )

        if overlap > 0:

            if speaker not in speaker_overlap:
                speaker_overlap[speaker] = 0

            speaker_overlap[speaker] += overlap


    # No matching speaker found
    if not speaker_overlap:
        return "UNKNOWN"


    # Return speaker with maximum overlap
    return max(
        speaker_overlap,
        key=speaker_overlap.get
    )


# ============================================================
# STEP 1: LOAD WHISPER

print("\n" + "=" * 60)
print("🧠 LOADING WHISPER LARGE-v3")
print("=" * 60)

start = time.time()

whisper_model = whisper.load_model(
    WHISPER_MODEL
)

print(
    f"✅ Whisper loaded in "
    f"{time.time() - start:.2f} seconds"
)


# ============================================================
# STEP 2: WHISPER TRANSCRIPTION

print("\n" + "=" * 60)
print("🧠 WHISPER SPEECH-TO-TEXT")
print("=" * 60)

print("Transcribing saved audio...")

start = time.time()


if WHISPER_LANGUAGE is None:

    result = whisper_model.transcribe(
        AUDIO_FILE,
        task="transcribe"
    )

else:

    result = whisper_model.transcribe(
        AUDIO_FILE,
        language=WHISPER_LANGUAGE,
        task="transcribe"
    )


print(
    f"✅ Transcription completed in "
    f"{time.time() - start:.2f} seconds"
)


# ============================================================
# GET WHISPER SEGMENTS

segments = result.get(
    "segments",
    []
)

transcript = result.get(
    "text",
    ""
).strip()


print("\n" + "-" * 60)
print("RAW TRANSCRIPT")
print("-" * 60)

print(transcript)

print("-" * 60)


# ============================================================
# STEP 3:  PYANNOTE

print("\n" + "=" * 60)
print("👥 LOADING SPEAKER DIARIZATION")
print("=" * 60)

start = time.time()

try:

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HF_TOKEN
    )

except Exception as e:

    print("\n❌ PyAnnote failed to load.")
    print()
    print("Error:")
    print(e)

    exit()


print(
    f"✅ Pyannote loaded in "
    f"{time.time() - start:.2f} seconds"
)

# STEP 4: AUTOMATIC SPEAKER DETECTION

print("\n" + "=" * 60)
print("👥 DETECTING SPEAKERS")
print("=" * 60)

print(
    "Automatically estimating number of speakers..."
)

start = time.time()

try:

    diarization = pipeline(
        AUDIO_FILE,
        min_speakers=1,
        max_speakers=10
    )

except Exception as e:

    print("\n❌ Speaker diarization failed.")
    print()
    print("Error:")
    print(e)

    exit()


print(
    f"✅ Diarization completed in "
    f"{time.time() - start:.2f} seconds"
)


# ============================================================
# STEP 5: FIND DETECTED SPEAKERS

speakers = set()

for turn, _, speaker in diarization.itertracks(
    yield_label=True
):

    speakers.add(speaker)


print("\n" + "-" * 60)
print("DETECTED SPEAKERS")
print("-" * 60)

for speaker in sorted(speakers):
    print(speaker)

print("-" * 60)

print(
    f"Total speakers detected: "
    f"{len(speakers)}"
)


# ============================================================
# STEP 6: MATCH WHISPER + PYANNOTE

print("\n" + "=" * 60)
print("🔗 COMBINING TRANSCRIPTION + SPEAKERS")
print("=" * 60)

speaker_transcript = []


for segment in segments:

    start_time = segment["start"]
    end_time = segment["end"]

    text = segment["text"].strip()

    if not text:
        continue

    speaker = find_speaker(
        start_time,
        end_time,
        diarization
    )

    speaker_transcript.append(
        {
            "speaker": speaker,
            "start": start_time,
            "end": end_time,
            "text": text
        }
    )


# ============================================================
# STEP 7: DISPLAY SPEAKER-LABELED TRANSCRIPT

print("\n" + "=" * 60)
print("📝 SPEAKER-LABELED TRANSCRIPT")
print("=" * 60)


for item in speaker_transcript:

    print()

    print(
        f"[{item['start']:.2f}s - "
        f"{item['end']:.2f}s]"
    )

    print(
        f"{item['speaker']}: "
        f"{item['text']}"
    )


print("\n" + "-" * 60)


# ============================================================
# STEP 8: PREPARE TRANSCRIPT FOR GEMMA

gemma_transcript = ""

for item in speaker_transcript:

    gemma_transcript += (
        f"{item['speaker']}: "
        f"{item['text']}\n"
    )


# ============================================================
# STEP 9: GEMMA CATEGORIZATION

print("\n" + "=" * 60)
print("🤖 GEMMA 3 CATEGORIZATION")
print("=" * 60)


prompt = f"""
You are an AI Conversation Analyzer.

Analyze the following conversation.

Choose categories that best describe the actual topic
of the conversation.

Possible categories include:

Academic
Meeting
Shopping
Finance
Travel
Health
Reminder
Ideas
Technical Discussion
Office
Personal
Entertainment
Emergency
General Conversation

Rules:

1. Choose a maximum of 3 categories.
2. Rank them from most relevant to least relevant.
3. Do not invent unrelated categories.
4. Return ONLY valid JSON.
5. No explanation.
6. No markdown.
7. If the conversation is a song, poem, or entertainment content,
   categorize it accordingly instead of treating it as a normal
   conversation.

JSON format:

{{
    "primary_category": "",
    "secondary_category": "",
    "tertiary_category": ""
}}

Conversation:

{gemma_transcript}
"""


print("Sending speaker-labeled transcript to Gemma...")


# ============================================================
# STEP 10: SEND TO OLLAMA

try:

    response = requests.post(

        OLLAMA_URL,

        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        },

        timeout=300
    )


    # ========================================================
    # CHECK RESPONSE
    if response.status_code != 200:

        print("\n❌ Ollama Error:")
        print(response.text)

    else:

        answer = response.json()["response"]

        print("\n" + "=" * 60)
        print("✅ FINAL AI RESULT")
        print("=" * 60)

        print(answer)


except requests.exceptions.ConnectionError:

    print("\n❌ OLLAMA CONNECTION ERROR")

    print(
        "Ollama is not running on "
        "http://localhost:11434"
    )

    print(
        "\nStart Ollama and run the program again."
    )


except requests.exceptions.Timeout:

    print("\n❌ Ollama request timed out.")


# ============================================================
# STEP 11: FINAL SUMMARY

print("\n" + "=" * 60)
print("📊 ANALYSIS SUMMARY")
print("=" * 60)

print(
    f"Audio file: "
    f"{os.path.basename(AUDIO_FILE)}"
)

print(
    f"Speakers detected: "
    f"{len(speakers)}"
)

print(
    f"Whisper segments: "
    f"{len(speaker_transcript)}"
)

print(
    f"Whisper language: "
    f"{result.get('language', 'unknown')}"
)

print("\n🎉 COMPLETE PIPELINE FINISHED")
print("=" * 60)
