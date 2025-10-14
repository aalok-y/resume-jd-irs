# main.py

import argparse
import json
import pandas as pd
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
import numpy as np
from joblib import dump, load

# Define constants for file paths for easier management
TRAINING_DATA_PATH = "training_data.csv"
VECTORIZER_PATH = "tfidf_vectorizer.joblib"
VECTOR_INDEX_PATH = "vector.index"
METADATA_PATH = "metadata.csv"

# Download necessary NLTK data
print("Downloading NLTK assets...")
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True) # Fix for the second LookupError
print("NLTK assets are ready.")


def get_wordnet_pos(tag: str) -> str:
    """Maps POS tag to a format recognized by the WordNet lemmatizer."""
    if tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("N"):
        return wordnet.NOUN
    elif tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def preprocess(text: str) -> str:
    """
    Cleans and preprocesses a text string by lowercasing, tokenizing,
    removing stopwords/punctuation, and lemmatizing.
    """
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [
        word
        for word in tokens
        if word.isalpha() and word not in stopwords.words("english")
    ]
    tagged_tokens = pos_tag(tokens)
    lemmatizer = WordNetLemmatizer()
    lemmas = [
        lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in tagged_tokens
    ]
    return " ".join(lemmas)


def build_index():
    """
    Loads training data, preprocesses it, and builds the TF-IDF vectorizer
    and FAISS index, saving them to disk.
    """
    print("Starting index build process...")
    # Load data
    df = pd.read_csv(TRAINING_DATA_PATH)

    # Add job_id based on row index
    df["job_id"] = (df.index + 1).astype(int)

    # Preprocess model_response column
    print("Preprocessing texts...")
    preprocessed_texts = df["model_response"].apply(preprocess).tolist()

    # Vectorize texts using TF-IDF
    print("Vectorizing texts with TF-IDF...")
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocessed_texts)

    # Save fitted vectorizer for future use
    dump(vectorizer, VECTORIZER_PATH)
    print(f"Vectorizer saved to: {VECTORIZER_PATH}")

    # Convert sparse matrix to dense float32 for FAISS
    tfidf_dense = tfidf_matrix.toarray().astype("float32")

    # Create and populate FAISS index
    print("Building and saving FAISS index...")
    index = faiss.IndexFlatL2(tfidf_dense.shape[1])
    index.add(tfidf_dense)
    faiss.write_index(index, VECTOR_INDEX_PATH)
    print(f"FAISS index saved to: {VECTOR_INDEX_PATH}")

    # Save metadata
    df[["job_id", "company_name", "model_response"]].to_csv(METADATA_PATH, index=False)
    print(f"Metadata saved to: {METADATA_PATH}")
    print("\nIndex build complete!")


def match_resume_to_jobs(resume_text: str, top_n: int = 5) -> list:
    """
    Matches a given resume text against the indexed jobs and returns the top N matches.
    """
    print(f"Loading index and searching for top {top_n} matches...")
    vectorizer = load(VECTORIZER_PATH)
    index = faiss.read_index(VECTOR_INDEX_PATH)
    metadata_df = pd.read_csv(METADATA_PATH)

    preprocessed_resume = preprocess(resume_text)
    resume_vec = vectorizer.transform([preprocessed_resume])
    resume_dense = resume_vec.toarray().astype("float32")

    distances, indices = index.search(resume_dense, top_n)

    results = []
    for rank, idx in enumerate(indices[0]):
        job_info = metadata_df.iloc[idx].to_dict()
        results.append(
            {
                "rank": rank + 1,
                "job_id": int(job_info.get("job_id")),
                "company": job_info.get("company_name"),
                "score": float(distances[0][rank]),
                "job_description": job_info.get("model_response"),
            }
        )
    return results


def add_documents(new_texts: list, new_metadata: list):
    """
    Adds new documents to an existing FAISS index and updates the metadata.
    """
    print(f"Adding {len(new_texts)} new documents to the index...")
    # Load existing components
    vectorizer = load(VECTORIZER_PATH)
    index = faiss.read_index(VECTOR_INDEX_PATH)
    df_meta = pd.read_csv(METADATA_PATH)

    # Determine start job_id
    last_job_id = df_meta['job_id'].max() if 'job_id' in df_meta.columns and not df_meta.empty else 0

    # Assign new job IDs
    for i, meta in enumerate(new_metadata):
        meta['job_id'] = last_job_id + i + 1

    # Preprocess and vectorize new texts
    preprocessed_new = [preprocess(text) for text in new_texts]
    new_tfidf = vectorizer.transform(preprocessed_new)
    new_dense = new_tfidf.toarray().astype("float32")

    # Add to FAISS index and save
    index.add(new_dense)
    faiss.write_index(index, VECTOR_INDEX_PATH)

    # Append to metadata and save
    df_new = pd.DataFrame(new_metadata)
    df_updated = pd.concat([df_meta, df_new], ignore_index=True)
    df_updated.to_csv(METADATA_PATH, index=False)

    print(f"Successfully added {len(new_texts)} documents. Index and metadata updated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resume-JD Matching Engine CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Command to build the initial index
    build_parser = subparsers.add_parser("build", help="Build the FAISS index and vectorizer from the training data.")

    # Command to match a resume
    match_parser = subparsers.add_parser("match", help="Match a resume text against the index.")
    match_parser.add_argument("resume", type=str, help="The resume text to match (enclose in quotes).")
    match_parser.add_argument("--top_n", type=int, default=5, help="Number of top matches to return.")

    args = parser.parse_args()

    if args.command == "build":
        build_index()
    elif args.command == "match":
        resume_str = args.resume
        matches = match_resume_to_jobs(resume_str, args.top_n)
        print("\n--- Top Matches Found ---")
        print(json.dumps(matches, indent=2))