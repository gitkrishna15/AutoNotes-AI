# AutoNotes-AI
AutoNotes-AI is a local AI-powered knowledge extraction engine that converts live audio into structured study material.
The system records audio in real time, transcribes it using on-device speech recognition, classifies content, and generates structured notes tailored to specific learning goals such as:
	•	General understanding
	•	Interview preparation
	•	Certification exams
	•	Ranked exam practice (Easy / Medium / Hard)
The entire system runs locally. No external cloud APIs are required.

# Overview
AutoNotes-AI transforms passive listening into structured knowledge artifacts.
It combines:
	•	Real-time audio capture
	•	On-device speech-to-text (Faster-Whisper)
	•	AI-based classification
	•	Local LLM reasoning (Ollama + Llama3)
	•	Structured knowledge synthesis
The project demonstrates how multiple AI components can be orchestrated into a practical multi-stage knowledge distillation pipeline.

# Key Capabilities
	•	Real-time audio recording
	•	On-device speech-to-text transcription (Faster-Whisper)
	•	AI-based content classification
	•	Structured study note generation
	•	Interview question generation with answers
	•	Certification-focused question generation
	•	Ranked exam preparation (Easy / Medium / Hard)
	•	Per-session confidence scoring
	•	Per-type cumulative aggregation
	•	Fully local execution
	•	CLI-based workflow control

# System Architecture
AutoNotes-AI follows a modular processing pipeline:

```
Audio Input
   ↓
Speech-to-Text Transcription
   ↓
AI Classification
   ↓
Session Transcript Creation
   ↓
LLM-Based Knowledge Generation
   ↓
Structured Notes Output
   ↓
Per-Type Aggregation
```
Each stage is independently testable and modular.


# Installation

1. Clone the repository

git clone https://github.com/gitkrishna15/AutoNotes-AI.git
cd AutoNotes-AI

2. Create a virtual environment

python3 -m venv venv
source venv/bin/activate

3. Install dependencies

pip install -r requirements.txt

4. Install Ollama
Download from:https://ollama.com
Then pull Llama3:

ollama pull llama3

# Basic Usage
Record and generate general notes (default)

python realtime_listener.py --mode record

If no note type is specified, general study notes are generated.

# Note Type Control
Supported note types:
	•	general
	•	int (interview preparation)
	•	cert (certification preparation)
	•	exams (ranked exam preparation)

# Generate interview and certification notes

python realtime_listener.py --mode record --enablenotes int,cert

# Generate hard-level exam preparation

python realtime_listener.py --mode record --enablenotes exams --level hard

If exams is enabled, a difficulty level must be specified:
	•	easy
	•	medium
	•	hard

# Aggregate Mode
Aggregate mode combines multiple session transcripts into consolidated knowledge files.
Example:

python realtime_listener.py --mode aggregate --enablenotes general,int

This generates cumulative files per enabled note type.

# Confidence Scoring
Each session includes a confidence score derived from:
	•	Transcription quality metrics
	•	Signal purity ratio
This provides an indicator of session reliability.

# Project Structure

AutoNotes-AI/
├── realtime_listener.py
├── requirements.txt
├── README.md
├── audio/
├── transcripts/
├── classification_logs/
├── notes/
└── aggregate_notes/

Runtime-generated folders are excluded via .gitignore.

# Technology Stack
	•	Python 3.10+
	•	Faster-Whisper
	•	Ollama
	•	Llama3
	•	SoundDevice
	•	NumPy
	•	SciPy

# Why This Project Matters
AutoNotes-AI demonstrates:
	•	Local LLM orchestration
	•	Multi-stage AI pipeline design
	•	Structured knowledge synthesis
	•	AI-based content classification
	•	CLI-driven AI workflows
	•	Deterministic local execution
It is particularly relevant for:
	•	AI engineering roles
	•	Applied machine learning
	•	Cloud AI architecture
	•	LLM orchestration systems

# Roadmap
Phase 1 (Current):
	•	Stable transcription + classification + structured notes
Phase 2 (Planned):

# License
This project is released under the MIT License.

