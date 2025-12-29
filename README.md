# Agent1

A Python project featuring AI agents for educational content generation.

## Setup Instructions

### 1. Create Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
python main.py
```

## Project Structure

- `main.py` - Entry point that demonstrates the TeacherAgent
- `agents/teacher.py` - TeacherAgent implementation for generating educational content
- `config.py` - Configuration settings
- `requirements.txt` - Python dependencies

## Usage

The main script runs a TeacherAgent that generates educational content about "Transformers in machine learning". You can modify the topic in `main.py` to explore different subjects.

## Deactivating Virtual Environment

When you're done working on the project:

```bash
deactivate
```