import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from ranking_engine import rank_resumes

st.set_page_config(page_title="ATS Resume Ranker", layout="wide")

st.title("📄Resume Ranking System")

st.markdown("### 📌 Enter Job Description")

job_text = st.text_area(
    "Paste or type the Job Description below:",
    height=200,
    placeholder="Enter required skills, qualifications, responsibilities..."
)

st.markdown("### 📂 Upload Candidate Resumes")

resume_files = st.file_uploader(
    "Upload Resumes (PDF, DOCX, TXT)",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

# Rank Button
if st.button("🚀 Rank Resumes"):

    if job_text.strip() != "" and resume_files:

        with st.spinner("Analyzing resumes..."):

            results = rank_resumes(job_text, resume_files)

        df = pd.DataFrame(
            results,
            columns=["Resume", "Similarity Score"]
        )

        df["Similarity Score (%)"] = (df["Similarity Score"] * 100).round(2)
        df = df.drop(columns=["Similarity Score"])

        df.index = df.index + 1
        df.index.name = "Rank"

        st.success("✅ Ranking Complete!")
        st.dataframe(df, use_container_width=True)

        # -------------------------------
        # Bar Chart
        # -------------------------------
        st.subheader("📊 Similarity Scores")

        fig1, ax1 = plt.subplots()
        ax1.barh(df["Resume"], df["Similarity Score (%)"])
        ax1.invert_yaxis()
        ax1.set_xlabel("Similarity Score (%)")
        st.pyplot(fig1)

        # -------------------------------
        # Score Curve
        # -------------------------------
        st.subheader("📉 Similarity Drop-off")

        fig2, ax2 = plt.subplots()
        ax2.plot(
            range(1, len(df) + 1),
            df["Similarity Score (%)"],
            marker="o"
        )
        ax2.set_xlabel("Rank")
        ax2.set_ylabel("Similarity Score (%)")
        ax2.grid(True)
        st.pyplot(fig2)

    else:
        st.warning("⚠ Please enter Job Description and upload at least one Resume.")
