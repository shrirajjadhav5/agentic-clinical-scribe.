import streamlit as st
import os
import json
import operator
from typing import Annotated, List, TypedDict
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
import easyocr
import numpy as np
from PIL import Image
import fitz  # PyMuPDF to handle PDF files

# --- CONFIGURATION ---
st.set_page_config(page_title="Agentic Clinical Scribe", layout="wide")

st.sidebar.title("Settings")
GROQ_API_KEY = st.sidebar.text_input("Enter Free Groq API Key", type="password")

@st.cache_resource
def load_ocr():
    # This downloads the model only once
    return easyocr.Reader(['en'])

reader = load_ocr()

# --- AGENT STATE ---
class AgentState(TypedDict):
    raw_text: str
    summary: dict
    plan: List[str]
    trace: Annotated[List[dict], operator.add]
    iteration: int
    doctor_memory: str 

# --- AGENT NODES ---
def planner_node(state: AgentState):
    llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
    prompt = f"""
    Current Summary: {json.dumps(state['summary'])}
    Tasks: Extract Demographics, Admission/Discharge dates, Diagnoses, Medication Changes.
    Text snippet: {state['raw_text'][:2000]}...
    What are the next 2 clinical items to verify?
    """
    res = llm.invoke(prompt)
    return {
        "plan": [res.content],
        "iteration": state["iteration"] + 1,
        "trace": [{"reasoning": "Analyzing document structure", "action": "Planning", "result": res.content}]
    }

def extractor_node(state: AgentState):
    llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile", temperature=0)
    prompt = f"""
    CONTEXT: {state['raw_text']}
    GOAL: {state['plan'][-1]}
    PREFERENCE: {state['doctor_memory']}
    
    RULES:
    1. NEVER invent clinical facts. If not found, write "MISSING".
    2. Identify conflicts (e.g. different discharge diagnoses) and flag them.
    
    Return findings in structured format.
    """
    res = llm.invoke(prompt)
    return {
        "summary": {**state["summary"], "findings": res.content},
        "trace": [{"reasoning": "Extracting facts from OCR text", "action": "Extraction", "result": "Batch processed"}]
    }

# --- GRAPH ---
builder = StateGraph(AgentState)
builder.add_node("planner", planner_node)
builder.add_node("extractor", extractor_node)
builder.set_entry_point("planner")
builder.add_edge("planner", "extractor")
builder.add_conditional_edges("extractor", lambda x: END if x["iteration"] >= 2 else "planner")
agent_app = builder.compile()

# --- UI LOGIC ---
st.title("🏥 Agentic AI for Clinical Discharge Summaries")

if "memory" not in st.session_state:
    st.session_state.memory = "Use formal medical terminology."

uploaded_files = st.file_uploader("Upload Patient Records (PDF or Images)", accept_multiple_files=True, type=["pdf", "png", "jpg", "jpeg"])

if st.button("Generate Discharge Summary"):
    if not GROQ_API_KEY:
        st.error("Please enter your Groq API Key in the sidebar.")
    elif not uploaded_files:
        st.warning("Please upload a file.")
    else:
        all_text = ""
        with st.status("Processing Documents (OCR)..."):
            for f in uploaded_files:
                if f.type == "application/pdf":
                    # CONVERT PDF PAGES TO IMAGES
                    pdf_bytes = f.read()
                    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                    for page_num in range(len(doc)):
                        page = doc.load_page(page_num)
                        pix = page.get_pixmap()
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        result = reader.readtext(np.array(img), detail=0)
                        all_text += f"\n--- {f.name} (Page {page_num+1}) ---\n" + " ".join(result)
                else:
                    # PROCESS IMAGES DIRECTLY
                    img = Image.open(f)
                    result = reader.readtext(np.array(img), detail=0)
                    all_text += f"\n--- Image {f.name} ---\n" + " ".join(result)
        
        # Run Agent
        inputs = {
            "raw_text": all_text,
            "summary": {},
            "plan": [],
            "trace": [],
            "iteration": 0,
            "doctor_memory": st.session_state.memory
        }
        
        with st.status("Agentic Reasoning..."):
            final_state = agent_app.invoke(inputs)
        
        st.subheader("Clinical Summary Draft")
        st.markdown(final_state["summary"].get("findings", "Error in generation."))
        
        with st.expander("View Audit Trail (Agent Steps)"):
            st.write(final_state["trace"])

# Learning Loop
st.sidebar.divider()
st.sidebar.subheader("Clinical Learning (Part 2)")
feedback = st.sidebar.text_area("Correct the agent style:")
if st.sidebar.button("Save Feedback"):
    st.session_state.memory += f" | {feedback}"
    st.sidebar.success("Updated agent memory.")