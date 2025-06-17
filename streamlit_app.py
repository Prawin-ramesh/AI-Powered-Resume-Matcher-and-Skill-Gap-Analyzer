import streamlit as st
import fitz  # PyMuPDF
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as gen_ai

# === Gemini Setup ===
gen_ai.configure(api_key=st.secrets["GEMINI_API_KEY"])
gemini_model = gen_ai.GenerativeModel('gemini-1.5-flash')

# === UI Config ===
st.set_page_config(page_title="Resume Matcher AI", layout="centered")

st.markdown("""
    <style>
        .stButton>button {
            background-color: #FF4B4B;
            color: white;
            font-weight: bold;
        }
        .centered-title {
            text-align: center;
            color: #FF4B4B;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <h1 class='centered-title'>📄 Resume Matcher AI</h1>
    <p style='text-align: center; font-size:18px;'>Compare your resume with any job description and find missing skills</p>
""", unsafe_allow_html=True)

# === Utility Functions ===
def load_skills(file_path="resume_matcher/skills_list.txt"):
    with open(file_path, "r") as f:
        return [line.strip() for line in f.readlines()]

def extract_text_from_pdf(uploaded_file):
    if uploaded_file is None:
        return ""
    doc = fitz.open(stream=uploaded_file.getvalue(), filetype="pdf")
    return "".join(page.get_text() for page in doc)

def extract_skills(text, skills_list):
    return list({skill for skill in skills_list if skill.lower() in text.lower()})

def compute_match_score(resume_text, jd_text):
    if not resume_text.strip() or not jd_text.strip():
        return 0.0
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, jd_text])
    score = cosine_similarity(vectors[0], vectors[1])[0][0]
    return round(score * 100, 2)

def get_gemini_suggestions(resume_text, jd_text):
    prompt = f"""
You're an AI Resume Advisor. Based on the resume and job description below, give exactly 3 short, bullet-point suggestions to improve the resume. Be concise (1 line each).

Resume:
{resume_text}

Job Description:
{jd_text}
"""
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error from Gemini: {e}"

# === Main App ===
try:
    skills_list = load_skills()
except FileNotFoundError:
    st.error("Skills list not found. Please upload resume_matcher/skills_list.txt.")
    st.stop()

resume_file = st.file_uploader("📄 Upload Resume (PDF)", type=["pdf"])
jd_text = st.text_area("📝 Paste Job Description Here")

if st.button("🔍 Analyze") and resume_file and jd_text.strip():
    with st.spinner("Analyzing..."):
        resume_text = extract_text_from_pdf(resume_file)
        resume_skills = extract_skills(resume_text, skills_list)
        jd_skills = extract_skills(jd_text, skills_list)
        match_score = compute_match_score(resume_text, jd_text)
        missing_skills = list(set(jd_skills) - set(resume_skills))
        gemini_tips = get_gemini_suggestions(resume_text, jd_text)

    st.metric(label="🎯 Match Score", value=f"{match_score}%")
    st.subheader("✅ Skills Found in Resume")
    st.markdown(", ".join([f"`{skill}`" for skill in resume_skills]) or "None")

    st.subheader("❌ Missing Skills from Resume")
    st.markdown(", ".join([f"`{skill}`" for skill in missing_skills]) or "None 🎉")

    st.subheader("💡 Gemini AI Suggestions")
    st.markdown(gemini_tips or "No suggestions returned.")
else:
    st.info("📂 Upload a resume and paste a job description to get started.")
