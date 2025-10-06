# import nltk
# from nltk.corpus import stopwords, wordnet
# from nltk.stem import WordNetLemmatizer
# from nltk import pos_tag, word_tokenize


# nltk.download("punkt")
# nltk.download("stopwords")
# nltk.download("wordnet")
# nltk.download("averaged_perceptron_tagger")
# nltk.download("averaged_perceptron_tagger_eng")


# def get_wordnet_pos(tag):
#     if tag.startswith("J"):
#         return wordnet.ADJ
#     elif tag.startswith("V"):
#         return wordnet.VERB
#     elif tag.startswith("N"):
#         return wordnet.NOUN
#     elif tag.startswith("R"):
#         return wordnet.ADV
#     else:
#         return wordnet.NOUN


# def preprocess(text):
#     # Lowercase
#     text = text.lower()
#     # Tokenization
#     tokens = word_tokenize(text)
#     # Remove punctuation and stopwords
#     tokens = [
#         word
#         for word in tokens
#         if word.isalpha() and word not in stopwords.words("english")
#     ]
#     # POS tagging
#     tagged_tokens = pos_tag(tokens)
#     # Lemmatization with correct POS
#     lemmatizer = WordNetLemmatizer()
#     lemmas = [
#         lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in tagged_tokens
#     ]
#     # Rejoin tokens
#     return " ".join(lemmas)


# # Example
# job_desc = """{
# "Core Responsibilities": "Training program to become an entry level software engineer. Work on real world projects in a team environment. No prior professional experience required.",
# "Required Skills": "College degree (associates or bachelors). Authorized to work in the US. Desire to learn to code. Problem solving skills. Team player. Adaptable. Communication and interpersonal skills.",
# "Educational Requirements": "College degree (associates or bachelors).",
# "Experience Level": "No prior professional experience required.",
# "Preferred Qualifications": "N/A",
# "Compensation and Benefits": "Competitive salary. Relocation & housing assistance. Paid time off. Industry certifications. Mentoring program. Experience with large US companies. Career acceleration opportunities."
# }"""
# preprocessed = preprocess(job_desc)
# print(preprocessed)


import pandas as pd
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
import numpy as np
from joblib import dump, load

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")


def get_wordnet_pos(tag):
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


def preprocess(text):
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


def preprocess_and_vectorize(
    csv_path="training_data.csv",
    vectorizer_path="tfidf_vectorizer.joblib",
    vector_index_path="vector.index",
    metadata_path="metadata.csv",
):
    # Load data
    df = pd.read_csv(csv_path)

    # Add job_id based on row index
    df["job_id"] = (df.index + 1).astype(int)

    # Preprocess model_response column
    preprocessed_texts = df["model_response"].apply(preprocess).tolist()

    # Vectorize texts using TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocessed_texts)

    # Save fitted vectorizer for future use
    dump(vectorizer, vectorizer_path)

    # Convert sparse matrix to dense float32 for FAISS
    tfidf_dense = tfidf_matrix.toarray().astype("float32")

    # Create and populate FAISS index for L2 similarity search
    index = faiss.IndexFlatL2(tfidf_dense.shape[1])
    index.add(tfidf_dense)
    faiss.write_index(index, vector_index_path)

    # Save metadata with job_id and company_name
    df[["job_id", "company_name", "model_response"]].to_csv(metadata_path, index=False)

    print(
        f"Preprocessing, vectorization done.\nVectorizer saved to: {vectorizer_path}\nIndex saved to: {vector_index_path}\nMetadata saved to: {metadata_path}"
    )


def match_resume_to_jobs(
    resume_text,
    top_n=5,
    vectorizer_path="tfidf_vectorizer.joblib",
    vector_index_path="vector.index",
    metadata_path="metadata.csv",
):
    vectorizer = load(vectorizer_path)
    index = faiss.read_index(vector_index_path)
    metadata_df = pd.read_csv(metadata_path)

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
                "job_id": job_info.get("job_id"),
                "company": job_info.get("company_name"),
                "score": float(distances[0][rank]),
                "job_description": job_info.get("model_response"),
            }
        )
    return results


# Usage:
# preprocess_and_vectorize()


resume_str = """{
  "Core Responsibilities": "Help drive business by supporting licensing managers on tasks related to category management, facilitating information between stakeholders, maintaining communication plans. Coordinate with internal teams to share brand and marketing updates with partners. Maintain and update title strategies and licensing plans. Collaborate with partners on product launches. Assist with licensing recaps, meetings, and agreements.",
  "Required Skills": "2+ years experience in preferably outbound licensing. Understanding of category manufacturing and sales cycles for toys/food/beverage preferred. Experience with entertainment/lifestyle brands. Self-starter, proactive, flexible. Thrives under pressure. Superb organizational and multitasking skills. Excellent communication skills.",
  "Educational Requirements": "N/A",
  "Experience Level": "2+ years experience in preferably outbound licensing",
  "Preferred Qualifications": "Understanding of category manufacturing and sales cycles for toys and/or food and beverage preferred. Experience working with reputable entertainment and/or lifestyle brands.",
  "Compensation and Benefits": "N/A"
}"""
# matches = match_resume_to_jobs(resume_str)
# print(matches)



def add_documents(
    new_texts,
    vectorizer_path="vectorizer.joblib",
    vector_index_path="vector.index",
    metadata_path="metadata.csv",
    new_metadata=[],
):
    # Load existing vectorizer and FAISS index
    vectorizer = joblib.load(vectorizer_path)
    index = faiss.read_index(vector_index_path)

    # Load existing metadata
    df_meta = pd.read_csv(metadata_path)
    
    # Determine start job_id from existing metadata
    if 'job_id' in df_meta.columns and not df_meta.empty:
        last_job_id = df_meta['job_id'].max()
    else:
        last_job_id = 0
    
    # Assign job IDs to new metadata starting after last_job_id
    for i, meta in enumerate(new_metadata):
        meta['job_id'] = last_job_id + i + 1  # job_id starts from 1

    # Preprocess new texts (make sure preprocess function is imported/defined globally)
    preprocessed_new = [preprocess(text) for text in new_texts]

    # Vectorize new documents using loaded vectorizer
    new_tfidf = vectorizer.transform(preprocessed_new)
    new_dense = new_tfidf.toarray().astype("float32")

    # Add new vectors to FAISS index
    index.add(new_dense)
    faiss.write_index(index, vector_index_path)

    # Append new metadata info (including job_id) to existing metadata dataframe
    df_new = pd.DataFrame(new_metadata)
    df_updated = pd.concat([df_meta, df_new], ignore_index=True)
    df_updated.to_csv(metadata_path, index=False)

    print(f"Added {len(new_texts)} documents. Updated index and metadata saved.")
