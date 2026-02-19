AutoNotes-AI
AutoNotes-AI is a local AI-powered knowledge extraction engine that converts live audio into structured study material. The system records audio in real time, transcribes it using on-device speech recognition, filters non-educational content, and generates structured notes tailored to specific learning goals such as general understanding, interview preparation, certification exams, and ranked exam practice.
The entire system runs locally. No external APIs are required.

Overview
AutoNotes-AI was designed to transform passive listening into structured knowledge artifacts. It combines speech-to-text processing, AI-based classification, and local large language model reasoning into a cohesive workflow.
The project demonstrates how multiple AI components can be orchestrated into a practical knowledge distillation pipeline.

Key Capabilities
	•	Real-time audio recording
	•	On-device speech-to-text transcription (Faster-Whisper)
	•	AI-based filtering of non-educational content
	•	Structured study note generation
	•	Interview question generation with answers
	•	Certification-focused question generation
	•	Ranked exam preparation content (Easy / Medium / Hard)
	•	Per-session confidence scoring
	•	Per-type cumulative knowledge aggregation
	•	Fully local execution using Ollama and Llama3
	•	CLI-based control of output types

System Architecture
The system follows a multi-stage processing pipeline:
Audio Input → Speech-to-Text Transcription → AI Classification (Educational Filtering) → Session Transcript Creation → LLM-Based Knowledge Generation → Structured Notes Output → Per-Type Aggregation
Each stage is modular and independently testable.

Default Behavior
If no note type is specified, AutoNotes-AI generates general study notes only.
Example:

python realtime_listener.py --mode record
This produces structured notes focused on understanding the topic clearly.

Note Type Control
Users can control the type of notes generated per execution.
Supported options:
	•	general
	•	int (interview preparation)
	•	cert (certification preparation)
	•	exams (ranked exam preparation)
Examples:
Generate interview and certification notes:

python realtime_listener.py --mode record --enablenotes int,cert
Generate hard-level exam preparation:

python realtime_listener.py --mode record --enablenotes exams --level hard
If exams are enabled, a difficulty level must be specified (easy, medium, or hard).

Aggregation
Aggregate mode allows combining multiple session transcripts into consolidated knowledge files.
Example:

python realtime_listener.py --mode aggregate --enablenotes general,int
This generates cumulative files per enabled note type.

Confidence Scoring
Each session includes a confidence score derived from:
	•	Transcription quality metrics
	•	Signal purity (ratio of accepted educational content to total captured content)
This provides an indication of session reliability.

Technology Stack
	•	Python 3
	•	Faster-Whisper (Speech-to-Text)
	•	Ollama (Local LLM Runtime)
	•	Llama3 (Reasoning Model)
	•	SoundDevice
	•	NumPy
	•	SciPy

Why This Project Matters
AutoNotes-AI demonstrates practical AI system design:
	•	Local LLM orchestration
	•	Multi-stage AI pipeline construction
	•	Structured knowledge synthesis
	•	Content filtering using language models
	•	CLI-based AI workflow control
	•	Deterministic execution without cloud dependency
It is particularly relevant for roles involving AI engineering, applied machine learning, cloud-based AI systems, and LLM orchestration.

Project Structure

AutoNotes-AI/
├── realtime_listener.py
├── requirements.txt
├── README.md
├── audio/
├── transcripts/
├── full_sessions/
├── notes/
└── aggregate_notes/

Prerequisites
	•	Python 3.10+
	•	Ollama installed
	•	Llama3 model pulled
	•	Microphone or virtual audio device configured

License
Add appropriate license information before public release.
