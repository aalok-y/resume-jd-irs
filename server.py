from fastapi import FastAPI, HTTPException,File, UploadFile
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import pandas as pd
import joblib
import faiss
import httpx
import json
import fitz
import tempfile

from google import genai


API_KEY = "AIzaSyDUi9Nz6"

client = genai.Client(api_key=API_KEY)


from main import (
    get_wordnet_pos,
    preprocess,
    preprocess_and_vectorize,
    match_resume_to_jobs,
    add_documents,
)

app = FastAPI()


# Load vectorizer, index, and metadata on startup
VECTOR_INDEX_PATH = "vector.index"
VECTOR_PATH = "tfidf_vectorizer.joblib"
METADATA_PATH = "metadata.csv"

vectorizer = joblib.load(VECTOR_PATH)
index = faiss.read_index(VECTOR_INDEX_PATH)
metadata_df = pd.read_csv(METADATA_PATH)

# Reuse your existing preprocess function here
# Include the preprocess function and other helper methods here or import them


class ResumeRequest(BaseModel):
    resume_text: str
    top_n: Optional[int] = 5


class AddDocumentsRequest(BaseModel):
    new_texts: List[str]
    new_metadata: List[dict]


class ResumeInput(BaseModel):
    resume_text: str


def extract_text_from_pdf(file_path):
    """
    Extracts and returns all text from a PDF file.

    Args:
        file_path (str or file-like object): Path to the PDF file or an open file object.

    Returns:
        str: The extracted text from all pages of the PDF.
    """
    text = ""

    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()

    return text.strip()

@app.post("/match_resume/")
def match_resume(data: ResumeRequest):
    preprocessed_resume = preprocess(data.resume_text)
    resume_vec = vectorizer.transform([preprocessed_resume])
    resume_dense = resume_vec.toarray().astype("float32")
    distances, indices = index.search(resume_dense, data.top_n)

    results = []
    for rank, idx in enumerate(indices[0]):
        job_info = metadata_df.iloc[idx].to_dict()
        results.append(
            {
                "rank": rank + 1,
                "job_id": job_info.get("job_id"),
                "company": job_info.get("company_name"),
                "score": float(distances[0][rank]),
                "job_description": job_info.get("model_response"),
            }
        )
    return {"matches": results}


@app.post("/add_documents/")
def add_documents(data: AddDocumentsRequest):
    global index, vectorizer, metadata_df

    # Calculate last job id
    if "job_id" in metadata_df.columns and not metadata_df.empty:
        last_job_id = metadata_df["job_id"].max()
    else:
        last_job_id = 0

    # Assign job_id for new metadata entries
    for i, meta in enumerate(data.new_metadata):
        meta["job_id"] = last_job_id + i + 1

    preprocessed_new = [preprocess(text) for text in data.new_texts]
    new_tfidf = vectorizer.transform(preprocessed_new)
    new_dense = new_tfidf.toarray().astype("float32")

    # Add to FAISS index and save
    index.add(new_dense)
    faiss.write_index(index, VECTOR_INDEX_PATH)

    # Append to metadata dataframe and save
    df_new = pd.DataFrame(data.new_metadata)
    metadata_df = pd.concat([metadata_df, df_new], ignore_index=True)
    metadata_df.to_csv(METADATA_PATH, index=False)

    return {"message": f"Added {len(data.new_texts)} documents."}


@app.post("/extract_resume_structured/")
async def extract_resume_structured(data: ResumeInput):
    # Craft prompt for structured extraction from raw resume text

    prompt = (
    "Extract a structured summary from the following resume text containing these fields: "
    "Core Responsibilities, Required Skills, Educational Requirements, Experience Level, Preferred Qualifications, and Compensation and Benefits. "
    f'Resume text: """{data.resume_text}""" '
    "Please respond only with a compact JSON object containing these fields. Write 'N/A' if specific details are not found. "
    "Do not format the JSON with indentation or line breaks.\n"
    "Example output:\n"
    '{"Core Responsibilities":"as an asc you will be highly influential in growing mind and market share of apple products while building longterm relationships with those who share your passion customer experiences are driven through you and your partner team growing in an ever changing and challenging environment you strive for perfection whether its maintaining visual merchandising or helping to grow and develop your partner team","Required Skills":"a passion to help people understand how apple products can enrich their livesexcellent communication skills allowing you to be as comfortable in front of a small group as you are speaking with individuals years preferred working in a dynamic sales andor results driven environment as well as proven success developing customer loyaltyability to encourage a partner team and grow apple business","Educational Requirements":"N/A","Experience Level":"years preferred","Preferred Qualifications":"N/A","Compensation and Benefits":"N/A"}'
)


    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash", contents=prompt
        )
        # The response.text should contain the structured JSON
        raw_response = response.text
        parsed = json.loads(raw_response)
        print(parsed)
        return parsed
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload_resume/")
async def upload_resume(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Save uploaded file to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(await file.read())
        temp_pdf_path = temp_pdf.name

    try:
        # Extract text
        resume_text = extract_text_from_pdf(temp_pdf_path)

        # Prepare payload
        payload = {"resume_text": resume_text}

        # Send POST request to the other API endpoint
        async with httpx.AsyncClient() as client:
            response = await client.post("http://localhost:8000/extract_resume_structured/", json=payload)

        # Handle response
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to extract structured resume data.")

        return response.json()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
