Agentic AI for Clinical Discharge Summaries

This repository contains an agentic AI system designed to transform messy, handwritten, and contradictory clinical source notes into structured, clinically safe discharge summary drafts.

Role: AI Engineer 

Tech Stack: Python, Streamlit, LangGraph, Groq (Llama 3.3 70B), EasyOCR, PyMuPDF.

🚀 Live Demo-https://agentic-clinical-scribe-cn74zu3y73whcqqcupwpbh.streamlit.app/
Click here to view the Live App on Streamlit Cloud

🧠 System Architecture
Unlike a standard linear LLM pipeline, this system uses a State Machine (LangGraph) to implement a true agentic loop.

1. The Agent Loop (Requirement #1)
The system follows a ReAct (Reasoning + Action) pattern:
Planner Node: Analyzes the current state of the summary and identifies missing required fields (Demographics, Meds, etc.).
Extractor Node: Executes targeted "searches" across the OCR-processed text to find specific facts.
Re-planning: If the extractor finds conflicting data (e.g., two different discharge dates), the planner detects the conflict and instructs the agent to flag it for review rather than guessing.

2. Clinical Safety & Guardrails (Requirement #3 & #4)
Zero Fabrication: The system prompt enforces a strict "No Invention" rule. If data is not present in the source notes, it is explicitly marked as MISSING or PENDING.
Conflict Resolution: If the agent detects contradictory information (e.g., different primary diagnoses in progress notes vs. admission records), it surfaces both and flags the section for clinician reconciliation.
Medication Reconciliation: The agent compares admission medication lists against discharge orders, flagging any changes that lack a documented reason in the hospital course.

3. Observability & Trace (Requirement #10)
Every decision made by the agent—from the initial plan to the final safety check—is recorded in a Step Trace. This audit trail is visible in the UI, allowing clinicians to see exactly why the agent chose to flag a specific piece of information.
🛠️ Installation & Local Setup
If you wish to run this locally, follow these steps:
Clone the repository:
code
Bash
git clone https://github.com/shrirajjadhav5/agentic-clinical-scribe.git
cd agentic-clinical-scribe
Install dependencies:
code
Bash
pip install -r requirements.txt
Install System Dependencies (For OCR/PDF):
Windows: No extra steps usually needed.
Mac/Linux: Ensure libgl1 is installed (required by OpenCV/EasyOCR).
Run the application:
code
Bash
streamlit run app.py

👨‍⚕️ Part 2: Learning from Doctor Edits
This project implements a Simulated Feedback Loop to improve agent performance over time:
Feedback Capture: Clinicians can provide style or clinical corrections in the sidebar.
Contextual Memory: These corrections are stored in the session state and injected into the "Extractor" and "Finalizer" nodes as a "Doctor Preference" layer.
Result: On subsequent runs, the agent adjusts its formatting and reasoning based on previous edits (e.g., "Always use formal ICD-10 terminology"), effectively lowering the "Edit Distance" for the clinician.
📋 Data Disclosure & Safety
Synthetic Data: All data processed during the development of this tool is synthetic.
Privacy: This application does not store uploaded PDFs. Processing happens in memory, and the Groq API is used for inference only.
Disclaimer: This is a technical proof-of-concept for an assignment and is not for real-world clinical use.

🎥 Video Demo Requirements

As per the assignment, the video demo includes:

Live run on a patient with conflicting/missing data.
Walkthrough of the Step Trace to show agent reasoning.
Demonstration of Part 2 learning where the agent adapts to a stylistic correction.
Author: [Shriraj Jadhav]
Date: June 2026
