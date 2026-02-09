import re
import numpy as np
import spacy
import PyPDF2
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load spaCy model once
nlp = spacy.load("en_core_web_md")


# -------------------------------
# TEXT EXTRACTION
# -------------------------------
def extract_text(file):
    text = ""

    if file.name.endswith(".pdf"):
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text

    elif file.name.endswith(".docx"):
        doc = Document(file)
        for para in doc.paragraphs:
            text += para.text + " "

    elif file.name.endswith(".txt"):
        text = file.read().decode("utf-8")

    return text


# -------------------------------
# TEXT CLEANING
# -------------------------------
def clean_text(text):
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = text.lower()

    doc = nlp(text)

    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop and len(token.text) > 2
    ]

    return tokens


# -------------------------------
# VECTOR FUNCTIONS
# -------------------------------
def spacy_vector(tokens, job_keywords):
    vectors = []

    for word in tokens:
        vec = nlp(word).vector
        if np.any(vec):
            if word in job_keywords:
                vectors.append(vec * 2.0)
            else:
                vectors.append(vec * 0.5)

    if not vectors:
        return np.zeros(nlp.vocab.vectors_length)

    return np.mean(vectors, axis=0)


def tfidf_vectors(documents):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    return tfidf_matrix.toarray()


def combine_vectors(v1, v2):
    combined = np.concatenate((v1, v2))
    norm = np.linalg.norm(combined)
    return combined if norm == 0 else combined / norm


# -------------------------------
# MAIN RANKING FUNCTION
# -------------------------------
def rank_resumes(job_text, resume_files):

    # Process Job Description (text input)
    job_tokens = clean_text(job_text)
    job_keywords = list(set(job_tokens))

    resume_texts = []
    resume_tokens = []
    resume_names = []

    # Process Resumes (files)
    for file in resume_files:
        text = extract_text(file)
        tokens = clean_text(text)

        resume_texts.append(" ".join(tokens))
        resume_tokens.append(list(set(tokens)))
        resume_names.append(file.name)

    # TF-IDF
    all_docs = [" ".join(job_tokens)] + resume_texts
    tfidf_matrix = tfidf_vectors(all_docs)

    job_tfidf = tfidf_matrix[0]
    resume_tfidf = tfidf_matrix[1:]

    # Create Job Vector
    job_vector = combine_vectors(
        spacy_vector(job_tokens, job_keywords),
        job_tfidf
    )

    # Create Resume Vectors
    resume_vectors = []
    for tokens, tfidf_vec in zip(resume_tokens, resume_tfidf):
        vec = combine_vectors(
            spacy_vector(tokens, job_keywords),
            tfidf_vec
        )
        resume_vectors.append(vec)

    # Cosine Similarity
    similarity_scores = cosine_similarity(
        [job_vector],
        resume_vectors
    )[0]

    ranked = sorted(
        zip(resume_names, similarity_scores),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked
