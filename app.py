import os
import tempfile
import streamlit as st

from typing import TypedDict, List
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)


# ============================================================
# Custom CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 18px;
    color: #666;
    margin-bottom: 30px;
}

.section-title {
    font-size: 24px;
    font-weight: 600;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# Title
# ============================================================

st.markdown(
    '<div class="main-title">📄 AI Resume Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Analyze your resume using Generative AI and get professional HR feedback.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# API Key
# ============================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("⚠️ GROQ_API_KEY environment variable is not set.")
    st.stop()


# ============================================================
# Groq Model
# ============================================================

model = ChatGroq(
    model="openai/gpt-oss-120b",
    groq_api_key=GROQ_API_KEY,
)


# ============================================================
# Structured Output
# ============================================================

class Analyzer(TypedDict):
    summary: str
    skills: List[str]
    education: List[str]
    experience: List[str]
    projects: List[str]
    missing_skills: List[str]
    recommendation: str


structured_model = model.with_structured_output(Analyzer)


# ============================================================
# Prompt
# ============================================================

prompt = PromptTemplate(
    template="""
You are an experienced HR Manager and Technical Recruiter
with more than 10 years of experience in recruitment,
resume screening, candidate evaluation, and hiring.

You have extensive knowledge of CV/resume analysis,
Applicant Tracking Systems (ATS), technical skills,
job requirements, candidate strengths and weaknesses,
and recruitment best practices.

Analyze the following candidate resume carefully.

RESUME:
{resume}

Focus on:

- Candidate's professional summary
- Technical and soft skills
- Educational qualifications
- Work experience
- Projects
- Missing or outdated skills
- Suitable job roles
- Overall employability

Do not invent information that is not present in the resume.

If some information is missing, clearly mention:

"Not provided in the resume."

Provide an objective and professional assessment.
""",
    input_variables=["resume"]
)


# ============================================================
# Create Chain
# ============================================================

chain = prompt | structured_model


# ============================================================
# File Upload
# ============================================================

uploaded_file = st.sidebar.file_uploader(
    "📤 Upload your Resume",
    type=["pdf"],
    help="Upload your resume in PDF format."
)


# ============================================================
# Analyze Button
# ============================================================

if uploaded_file is not None:

    st.sidebar.success(f"Uploaded: {uploaded_file.name}")

    if st.sidebar.button("🔍 Analyze Resume", use_container_width=True):

        with st.sidebar.spinner("Analyzing your resume..."):

            try:

                # ------------------------------------------------
                # Save uploaded PDF temporarily
                # ------------------------------------------------

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as temp_file:

                    temp_file.write(uploaded_file.getbuffer())

                    temp_pdf_path = temp_file.name


                # ------------------------------------------------
                # Load PDF
                # ------------------------------------------------

                loader = PyPDFLoader(temp_pdf_path)

                documents = loader.load()


                # ------------------------------------------------
                # Extract text
                # ------------------------------------------------

                resume_text = "\n".join(
                    doc.page_content
                    for doc in documents
                )


                # ------------------------------------------------
                # Check PDF text
                # ------------------------------------------------

                if not resume_text.strip():

                    st.error(
                        "❌ Could not extract text from this PDF. "
                        "Please upload a text-based PDF."
                    )

                    st.stop()


                # ------------------------------------------------
                # Invoke LangChain chain
                # ------------------------------------------------

                result = chain.invoke({
                    "resume": resume_text
                })


                # ------------------------------------------------
                # Display Results
                # ------------------------------------------------

                st.markdown(
                    '<div class="section-title">👤 Candidate Summary</div>',
                    unsafe_allow_html=True
                )

                st.info(result["summary"])


                # ------------------------------------------------
                # Skills
                # ------------------------------------------------

                st.markdown(
                    '<div class="section-title">🛠️ Skills</div>',
                    unsafe_allow_html=True
                )

                cols = st.columns(3)

                for i, skill in enumerate(result["skills"]):

                    with cols[i % 3]:
                        st.success(f"✓ {skill}")


                # ------------------------------------------------
                # Education
                # ------------------------------------------------

                st.markdown(
                    '<div class="section-title">🎓 Education</div>',
                    unsafe_allow_html=True
                )

                for education in result["education"]:
                    st.write(f"• {education}")


                # ------------------------------------------------
                # Experience
                # ------------------------------------------------

                st.markdown(
                    '<div class="section-title">💼 Experience</div>',
                    unsafe_allow_html=True
                )

                for experience in result["experience"]:
                    st.write(f"• {experience}")


                # ------------------------------------------------
                # Projects
                # ------------------------------------------------

                st.markdown(
                    '<div class="section-title">🚀 Projects</div>',
                    unsafe_allow_html=True
                )

                for project in result["projects"]:
                    st.write(f"• {project}")


                # ------------------------------------------------
                # Missing Skills
                # ------------------------------------------------

                st.markdown(
                    '<div class="section-title">⚠️ Missing / Weak Skills</div>',
                    unsafe_allow_html=True
                )

                if result["missing_skills"]:

                    for skill in result["missing_skills"]:
                        st.warning(skill)

                else:

                    st.success("No major missing skills identified.")


                # ------------------------------------------------
                # Recommendation
                # ------------------------------------------------

                st.markdown(
                    '<div class="section-title">💡 HR Recommendation</div>',
                    unsafe_allow_html=True
                )

                st.info(result["recommendation"])


                # ------------------------------------------------
                # Success
                # ------------------------------------------------

                st.success(
                    "✅ Resume analysis completed successfully!"
                )


            except Exception as e:

                st.error(
                    f"❌ Something went wrong: {str(e)}"
                )
