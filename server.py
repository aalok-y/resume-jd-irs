import os
import json
import tempfile
from typing import List, Optional

import faiss
import fitz  # PyMuPDF
import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from google import generativeai as genai
from pydantic import BaseModel

# Import the functions the server actually uses from main.py
from main import match_resume_to_jobs, preprocess

# --- Configuration ---
API_KEY = "AIzaSyD7wc4FAcbvLp7M1hYciW2_0qiYlVmtZYo" 

VECTOR_INDEX_PATH = "vector.index"
VECTORIZER_PATH = "tfidf_vectorizer.joblib"
METADATA_PATH = "metadata.csv"

# --- App Initialization ---
app = FastAPI(title="Resume Matcher API")

# --- CORS Middleware ---
# This allows your React frontend to communicate with this backend.
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:5173", # Default for Vite React projects
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Safety Check & GenAI Configuration ---
if not API_KEY or API_KEY == "YOUR_GOOGLE_API_KEY_HERE":
    raise ValueError("CRITICAL: GOOGLE_API_KEY is not set. Please replace the placeholder in server.py.")

genai.configure(api_key=API_KEY)

# --- Load Models on Startup ---
try:
    vectorizer = joblib.load(VECTORIZER_PATH)
    index = faiss.read_index(VECTOR_INDEX_PATH)
    metadata_df = pd.read_csv(METADATA_PATH)
    print("Server ready: Models and data loaded successfully.")
except FileNotFoundError as e:
    print(f"ERROR: A required file was not found: {e}")
    print("Please make sure you have run 'python main.py build' successfully before starting the server.")
    exit()

# --- Pydantic Models ---
class ResumeRequest(BaseModel):
    resume_text: str
    top_n: Optional[int] = 5

# --- Helper Functions ---
def extract_text_from_pdf(file_path: str) -> str:
    """Extracts and returns all text from a PDF file."""
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text.strip()

async def get_structured_data_from_gemini(text: str) -> dict:
    """Sends text to Gemini and returns structured JSON."""
    prompt = (
        "Extract a structured summary from the following resume text containing these fields: "
        "'Core Responsibilities', 'Required Skills', 'Educational Requirements', 'Experience Level', "
        "'Preferred Qualifications', and 'Compensation and Benefits'. "
        f'Resume text: """{text}""" '
        "Please respond only with a compact JSON object. Use 'N/A' if details are not found."
    )
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        clean_response = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(clean_response)
    except Exception as e:
        print(f"Error communicating with GenAI or parsing JSON: {e}")
        raise HTTPException(status_code=500, detail="Failed to get structured data from AI model.")

# --- API Endpoints ---
@app.post("/upload_and_match/", summary="Upload a PDF resume to get job matches")
async def upload_and_match(top_n: int = 5, file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    temp_pdf_path = temp_pdf.name
    
    try:
        temp_pdf.write(await file.read())
        temp_pdf.close()
        resume_text = extract_text_from_pdf(temp_pdf_path)
        structured_resume = await get_structured_data_from_gemini(resume_text)
        structured_resume_text = json.dumps(structured_resume)
        matches = match_resume_to_jobs(resume_text=structured_resume_text, top_n=top_n)
        return {"structured_resume": structured_resume, "matches": matches}
    finally:
        os.unlink(temp_pdf_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
