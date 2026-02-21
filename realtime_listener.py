import os
import glob
import argparse
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from faster_whisper import WhisperModel
from datetime import datetime
import subprocess
import sys
import json
import time
import gc
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(
    filename="autonotes.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

class UsageTracker:
    def __init__(self):
        self.tokens = 0
    def add(self, t):
        self.tokens += t
    def report(self):
        return self.tokens

USAGE = UsageTracker()
PROMPT_CACHE = {}


# ================= CONFIG =================
SAMPLE_RATE = 16000
CHUNK_DURATION = 10
MODEL_SIZE = "base"
DEVICE_INDEX = 0
LLM_MODEL = "llama3"
# ==========================================

# ================= DIRECTORIES =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(BASE_DIR, "audio")
TRANSCRIPT_DIR = os.path.join(BASE_DIR, "transcripts")
CLASSIFICATION_DIR = os.path.join(BASE_DIR, "classification_logs")
NOTES_DIR = os.path.join(BASE_DIR, "notes")
AGGREGATE_DIR = os.path.join(BASE_DIR, "aggregate_notes")
PROCESSED_FILE = "processed_sessions.txt"
CHECKPOINT_FILE = os.path.join(BASE_DIR, "session_checkpoint.json")

DEFAULT_PROMPTS = {
    "general": """
Create structured study notes from the transcript below.

Focus on:
- Clear conceptual explanations
- Definitions
- Key takeaways
- Important terminology
- Practical understanding
""",

    "int": """
Generate interview preparation material from the transcript below.

Include:
- Conceptual questions
- Scenario-based questions
- Deep-dive explanations
- Model answers
""",

    "cert": """
Generate certification-style preparation content from the transcript below.

Include:
- Scenario-based questions
- Objective-based questions
- Key exam traps
- Clear explanations
""",

    "exams": """
Generate exam-style preparation material from the transcript below.

Create:
- Multiple-choice questions
- Conceptual clarity questions
- Application-based problems
- Clear answers with explanation
"""
}


for folder in [AUDIO_DIR, TRANSCRIPT_DIR, CLASSIFICATION_DIR, NOTES_DIR, AGGREGATE_DIR]:
    os.makedirs(folder, exist_ok=True)

if not os.path.exists(PROCESSED_FILE):
    open(PROCESSED_FILE, "w").close()

# ===== Whisper Singleton (Load Once) =====
WHISPER_MODEL = None

def get_whisper():
    global WHISPER_MODEL
    if WHISPER_MODEL is None:
        print("Loading Faster-Whisper model once...")
        WHISPER_MODEL = WhisperModel(MODEL_SIZE, device="auto", compute_type="int8")
    return WHISPER_MODEL


# ================= METRICS =================

def reset_metrics():
    return {
        "total_chunks": 0,
        "accepted_chunks": 0,
        "transcription_scores": []
    }

# ================= UTILITIES =================

def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def read_processed_sessions():
    with open(PROCESSED_FILE, "r") as f:
        return set(line.strip() for line in f.readlines())

def mark_sessions_processed(files):
    with open(PROCESSED_FILE, "a") as f:
        for file in files:
            f.write(file + "\n")

def record_chunk():
    device_info = sd.query_devices(DEVICE_INDEX, 'input')
    max_channels = device_info['max_input_channels']
    audio = sd.rec(
        int(CHUNK_DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=max_channels,
        device=DEVICE_INDEX
    )
    sd.wait()
    return audio

def save_audio(audio_data):
    filename = os.path.join(AUDIO_DIR, f"chunk_{timestamp()}.wav")
    wav.write(filename, SAMPLE_RATE, audio_data)
    return filename

def transcribe_audio(audio_file, metrics):
    model = get_whisper()
    segments, _ = model.transcribe(audio_file)
    text = ""
    scores = []

    for segment in segments:
        text += segment.text + " "
        if hasattr(segment, "avg_logprob"):
            score = max(0, min(1, 1 + segment.avg_logprob))
            scores.append(score)

    if scores:
        metrics["transcription_scores"].append(sum(scores) / len(scores))

    return text.strip()

def cleanup_chunks():
    for file in glob.glob(os.path.join(AUDIO_DIR, "*.wav")):
        os.remove(file)
    #for file in glob.glob(os.path.join(TRANSCRIPT_DIR, "*.txt")):
    #    os.remove(file)

def save_checkpoint(session_id, session_text, metrics, classification_log):
    data = {
        "session_id": session_id,
        "session_text": session_text,
        "metrics": metrics,
        "classification_log": classification_log
    }
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(data, f)
    logging.info("Checkpoint saved.")

def load_checkpoint():
    if not os.path.exists(CHECKPOINT_FILE):
        return None

    try:
        with open(CHECKPOINT_FILE, "r") as f:
            data = json.load(f)
        logging.info("Checkpoint loaded.")
        return data
    except Exception:
        return None

def clear_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        logging.info("Checkpoint cleared.")


# ================= AI CLASSIFICATION =================

def classify_chunk(text):
    prompt = f"""
Classify into one category:
EDUCATIONAL
ADVERTISEMENT
SMALL_TALK
IRRELEVANT
NOISE

Return ONLY the category word.

Content:
{text}
"""
    try:
        result = subprocess.run(
            ["ollama", "run", LLM_MODEL, "-p", prompt],
            capture_output=True,
            text=True,
            timeout=30
        )
        raw = result.stdout.strip().upper()

        if "EDUCATIONAL" in raw:
            return "EDUCATIONAL"
        if "ADVERTISEMENT" in raw:
            return "ADVERTISEMENT"
        if "SMALL" in raw:
            return "SMALL_TALK"
        if "IRRELEVANT" in raw:
            return "IRRELEVANT"
        if "NOISE" in raw:
            return "NOISE"

        return "EDUCATIONAL"

    except Exception:
        return "EDUCATIONAL"

# ================= CONFIDENCE =================

def calculate_confidence(metrics):
    total_chunks = metrics["total_chunks"]
    accepted_chunks = metrics["accepted_chunks"]
    scores = metrics["transcription_scores"]

    if total_chunks == 0:
        return {
            "transcription": 0,
            "signal_purity": 0,
            "stability": 0,
            "completeness": 0,
            "overall": 0
        }

    # Transcription Quality
    transcription = sum(scores) / len(scores) if scores else 0

    # Signal Purity
    signal_purity = accepted_chunks / total_chunks

    # Stability (variance of transcription scores)
    if len(scores) > 1:
        mean_score = transcription
        variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
        stability = max(0, 1 - variance)
    else:
        stability = transcription

    # Completeness Guard
    completeness = min(1, accepted_chunks / max(5, total_chunks))

    # Weighted Confidence v2
    overall = (
        transcription * 0.4 +
        signal_purity * 0.3 +
        stability * 0.2 +
        completeness * 0.1
    )

    return {
        "transcription": round(transcription * 100, 2),
        "signal_purity": round(signal_purity * 100, 2),
        "stability": round(stability * 100, 2),
        "completeness": round(completeness * 100, 2),
        "overall": round(overall * 100, 2)
    }


# ================= NOTE RESOLUTION =================

def resolve_note_types(args):
    if not args.enablenotes:
        note_types = ["general"]
    else:
        note_types = [n.strip() for n in args.enablenotes.split(",")]

    allowed = {"general", "int", "cert", "exams"}

    for n in note_types:
        if n not in allowed:
            print(f"Invalid note type: {n}")
            sys.exit(1)

    if "exams" in note_types and not args.level:
        print("ERROR: --level is required when exams is enabled.")
        sys.exit(1)

    return note_types

# ================= LLM =================

def run_llm(prompt):
    try:
        result = subprocess.run(
            ["ollama", "run", LLM_MODEL],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=120
        )
        output = result.stdout
        estimated_tokens = len(prompt.split()) + len(output.split())
        return output, estimated_tokens
    except Exception as e:
        logging.error(f"LLM error: {e}")
        return "", 0

def run_llm_with_retry(prompt, retries=3):
    for attempt in range(retries):
        try:
            start = time.time()
            output, tokens = run_llm(prompt)
            latency = round(time.time() - start, 2)
            logging.info(f"LLM | Tokens={tokens} | Latency={latency}s")
            return output, tokens
        except Exception:
            time.sleep(2)
    return "", 0

def cached_llm(prompt):
    if prompt in PROMPT_CACHE:
        return PROMPT_CACHE[prompt], 0
    output, tokens = run_llm_with_retry(prompt)
    PROMPT_CACHE[prompt] = output
    return output, tokens


# ================= NOTE GENERATION =================

def build_prompt(note_type, transcript, level=None, args=None):
    default_prompt = DEFAULT_PROMPTS.get(note_type, "")

    custom_prompt_map = {}

    if args is not None:
        custom_prompt_map = {
            "general": getattr(args, "generalprompt", None),
            "cert": getattr(args, "certprompt", None),
            "int": getattr(args, "intprompt", None),
            "exams": getattr(args, "examsprompt", None),
        }

    base_prompt = custom_prompt_map.get(note_type) or default_prompt
    level_text = f"\nGenerate output at {level} level." if level else ""

    return f"{base_prompt}{level_text}\n\nTranscript:\n{transcript}"

def chunk_text(text, max_words=1800):
    words = text.split()
    for i in range(0, len(words), max_words):
        yield " ".join(words[i:i+max_words])


def generate_notes(transcript, name, folder, note_types, level, args=None):
    os.makedirs(folder, exist_ok=True)

    def process(note_type):
        try:
            outputs = []
            total_tokens = 0

            for chunk in chunk_text(transcript):
                prompt = build_prompt(note_type, chunk, level, args)
                output, tokens = cached_llm(prompt)
                outputs.append(output)
                total_tokens += tokens

            final_output = "\n".join(outputs)
            file_path = os.path.join(folder, f"{name}_{note_type}_notes.txt")

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(final_output)

            USAGE.add(total_tokens)
            logging.info(f"Session={name} | Note={note_type} | Tokens={total_tokens}")
            print(f"Saved: {file_path}")

        except Exception as e:
            logging.error(f"Error generating {note_type}: {e}")
            print(f"Error generating {note_type}: {e}")

    with ThreadPoolExecutor(max_workers=min(4, len(note_types))) as executor:
        futures = [executor.submit(process, nt) for nt in note_types]
        for future in as_completed(futures):
            future.result()

    print(f"Total Tokens Used: {USAGE.report()}")


# ================= CUMULATIVE PER-TYPE =================

def create_cumulative_files(name, note_types):

    mapping = {
        "general": "*_general_notes.txt",
        "int": "*_interview_notes.txt",
        "cert": "*_certification_notes.txt",
        "exams": f"*_exams_*.txt"
    }

    for note_type in note_types:
        pattern = mapping[note_type]
        files = glob.glob(os.path.join(NOTES_DIR, pattern))

        if not files:
            continue

        cumulative_path = os.path.join(
            AGGREGATE_DIR,
            f"{name}_cumulative_{note_type}_notes.txt"
        )

        with open(cumulative_path, "w") as out:
            for file in files:
                with open(file, "r") as f:
                    out.write(f.read() + "\n\n")

# ================= MODES =================

def recording_mode(args):
    checkpoint = load_checkpoint()
    if checkpoint:
        print(f"Resumed session {session_id}")
        print(f"Recovered {metrics['total_chunks']} chunks so far.")

    if checkpoint:
        print("Resuming previous session...")
        session_id = checkpoint["session_id"]
        session_text = checkpoint["session_text"]
        metrics = checkpoint["metrics"]
        classification_log = checkpoint["classification_log"]
    else:
        metrics = reset_metrics()
        session_text = ""
        classification_log = []
        session_id = timestamp() 

    note_types = resolve_note_types(args)

    try:
        while True:
            audio_data = record_chunk()
            audio_file = save_audio(audio_data)

            text = transcribe_audio(audio_file, metrics)
            metrics["total_chunks"] = metrics.get("total_chunks", 0) + 1

            category = classify_chunk(text)
            # Always append transcription
            session_text += text + "\n"
            if category == "EDUCATIONAL":
                metrics["accepted_chunks"] = metrics.get("accepted_chunks", 0) + 1

            # Store classification metadata
            classification_log.append({
                "chunk_time": datetime.now().isoformat(),
                "category": category,
                "preview": text[:120]
            })
            save_checkpoint(session_id, session_text, metrics, classification_log)

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        confidence = calculate_confidence(metrics)

        session_file = os.path.join(TRANSCRIPT_DIR, f"session_{session_id}.txt")
        with open(session_file, "w") as f:
            bytes_written=f.write(session_text)

        classification_file = os.path.join(
            CLASSIFICATION_DIR,
            f"session_{session_id}_classification.json"
        )

        with open(classification_file, "w") as f:
            json.dump(classification_log, f, indent=4)

        generate_notes(session_text, session_id, NOTES_DIR, note_types, args.level,args)
        clear_checkpoint()
        cleanup_chunks()
        print("\nSession saved.")
        print("Confidence Breakdown:")
        for k, v in confidence.items():
            print(f"  {k}: {v}%")
        gc.collect()

def generate_mode(args):
    note_types = resolve_note_types(args)

    processed = read_processed_sessions()
    all_sessions = sorted(glob.glob(os.path.join(TRANSCRIPT_DIR, "*.txt")))
    unread = [s for s in all_sessions if s not in processed]

    if not unread:
        print("No new sessions.")
        return

    combined = ""
    for file in unread:
        with open(file, "r") as f:
            combined += f.read() + "\n"

    name = timestamp()
    generate_notes(combined, name, NOTES_DIR, note_types, args.level,args)
    mark_sessions_processed(unread)

def aggregate_mode(args):
    note_types = resolve_note_types(args)

    sessions = sorted(glob.glob(os.path.join(TRANSCRIPT_DIR, "*.txt")))
    if not sessions:
        print("No sessions available.")
        return

    combined = ""
    for file in sessions:
        with open(file, "r") as f:
            combined += f.read() + "\n"

    name = args.name if args.name else timestamp()
    generate_notes(combined, name, AGGREGATE_DIR, note_types, args.level,args)
    create_cumulative_files(name, note_types)

# ================= MAIN =================

def main():
    parser = argparse.ArgumentParser(description="Local AI Study Engine v9")

    parser.add_argument("--mode", choices=["record", "generate", "aggregate"])
    parser.add_argument("--name")
    parser.add_argument("--enablenotes", help="Comma-separated note types: general,int,cert,exams")
    parser.add_argument("--level", choices=["easy", "medium", "hard"])
    parser.add_argument("--generalprompt", help="Custom prompt for general notes")
    parser.add_argument("--certprompt", help="Custom prompt for certification notes")
    parser.add_argument("--intprompt", help="Custom prompt for interview notes")
    parser.add_argument("--examsprompt", help="Custom prompt for exam notes")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    if args.mode == "record":
        recording_mode(args)
    elif args.mode == "generate":
        generate_mode(args)
    elif args.mode == "aggregate":
        aggregate_mode(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

