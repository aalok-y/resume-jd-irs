from sklearn.feature_extraction.text import TfidfVectorizer

# 1. List of cleaned strings (job descriptions + resumes)
documents = [
    """{
    "Core Responsibilities": "Training program to become an entry level software engineer. Work on real world projects in a team environment. No prior professional experience required.",
    "Required Skills": "College degree (associates or bachelors). Authorized to work in the US. Desire to learn to code. Problem solving skills. Team player. Adaptable. Communication and interpersonal skills.",
    "Educational Requirements": "College degree (associates or bachelors).",
    "Experience Level": "No prior professional experience required.",
    "Preferred Qualifications": "N/A",
    "Compensation and Benefits": "Competitive salary. Relocation & housing assistance. Paid time off. Industry certifications. Mentoring program. Experience with large US companies. Career acceleration opportunities."
    }"""
]  # Replace with your preprocessed text

# 2. Create vectorizer and fit_transform
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

# 3. To see TF-IDF features:
print(tfidf_vectorizer.get_feature_names_out())
print(tfidf_matrix.shape)  # (num_documents, num_terms)

# Each row = 1 document (job or resume). Use this matrix for similarity calculation.
