import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
import json
import tempfile

# Page Configuration

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)


# Custom CSS

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


# Title

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


# Model Creation
import json
import streamlit as st
from langchain_groq import ChatGroq
api_key = st.secrets["GROQ_API_KEY"]
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=GROQ_API_KEY,
    temperature=0
)

json_llm = llm.bind(
    response_format={"type": "json_object"}
)



# Structured Output
from typing import TypedDict, List

from typing import TypedDict, List

class Analyzer(TypedDict):
    summary: str
    skills: List[str]
    education: List[str]
    experience: List[str]
    projects: List[str]
    missing_skills: List[str]
    suitable_roles: List[str]
    employability: str
    salary_prediction: List[str]
    recommendation: str


#structured_model = model.with_structured_output(Analyzer)


# Prompt
from langchain_core.prompts import PromptTemplate


# prompt = PromptTemplate(
#     template="""
# You are an experienced HR Manager and Technical Recruiter
# with more than 10 years of experience in recruitment,
# resume screening, candidate evaluation, and hiring.

# You have extensive knowledge of CV/resume analysis,
# Applicant Tracking Systems (ATS), technical skills,
# job requirements, candidate strengths and weaknesses,
# and recruitment best practices.

# Analyze the following candidate resume carefully.

# RESUME:
# {resume}

# Focus on:

# - Candidate's professional summary
# - Technical and soft skills
# - Educational qualifications
# - Work experience
# - Projects
# - Missing or outdated skills
# - Suitable job roles
# - Overall employability

# Do not invent information that is not present in the resume.

# If some information is missing, clearly mention:

# "Not provided in the resume."

# Provide an objective and professional assessment.
# """,
#     input_variables=["resume"]
# )
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    template="""
You are an experienced HR Manager and Technical Recruiter
with more than 10 years of experience in recruitment,
resume screening, salary benchmarking, candidate evaluation,
and hiring.

You have extensive knowledge of CV/resume analysis,
Applicant Tracking Systems (ATS), technical skills,
job requirements, candidate strengths and weaknesses,
Indian IT industry salary trends, and recruitment best practices.

Analyze the following candidate resume carefully.

RESUME:
{resume}

Focus on the following sections:

1. Professional Summary
   - Evaluate the candidate's profile and career objectives.
   - Identify key strengths and areas for improvement.

2. Technical and Soft Skills
   - Identify technical skills, programming languages,
     frameworks, tools, and soft skills.
   - Identify missing or outdated skills relevant to target roles.

3. Educational Qualifications
   - Evaluate degrees, specializations, academic achievements,
     and their relevance to target job roles.

4. Work Experience
   - Analyze previous employment, internships, responsibilities,
     and professional achievements.
   - Identify employment gaps only when supported by the resume.
   - Do not assume the reason for any career gap.

5. Projects
   - Evaluate projects based on technical complexity,
     practical implementation, tools used, and business value.
   - Identify missing performance metrics or measurable outcomes.
   - Do not invent project results or achievements.

6. Missing or Outdated Skills
   - Recommend skills that would improve the candidate's
     employability for suitable target roles.
   - Distinguish essential skills from optional skills.

7. Suitable Job Roles
   - Recommend relevant job titles based on the candidate's
     actual education, skills, experience, and projects.
   - Explain the reasons for each recommendation.
   - Identify additional skills required for each role.

8. Overall Employability
   - Assess technical readiness, project experience,
     professional experience, and portfolio quality.
   - Highlight strengths, limitations, and improvement priorities.
   - Provide a balanced assessment without making discriminatory
     assumptions about age, career gaps, or personal circumstances.

9. Salary Prediction (Indian Job Market)

   Estimate the candidate's potential annual salary in India
   using the following criteria:

   - Highest educational qualification and specialization.
   - Relevant technical skills and practical proficiency.
   - Quality and complexity of completed projects.
   - Professional work experience and relevant internships.
   - Career gaps, without assuming that a gap automatically
     reduces technical ability.
   - Target job roles and typical hiring requirements.
   - Evidence of deployment, measurable project outcomes,
     certifications, and portfolio quality, when provided.
   - Differences between service-based companies, product-based
     companies, startups, and other employers.

   Provide the following salary estimates in INR:

   A. Expected fresher or entry-level salary:
      Estimate a realistic annual CTC range for candidates
      applying to suitable entry-level positions.

   B. Expected salary by job role:
      Provide separate estimated annual CTC ranges for relevant
      roles such as AI Engineer, Agentic AI Engineer,
      Machine Learning Engineer, Data Scientist, and
      AI Application Developer, where appropriate.

   C. Conservative, realistic, and optimistic scenarios:
      - Conservative: Candidate meets basic job requirements.
      - Realistic: Candidate demonstrates the stated skills
        and can explain and defend their projects in interviews.
      - Optimistic: Candidate performs strongly in interviews
        and matches the requirements of a higher-paying employer.

   D. Monthly salary:
      Provide an approximate monthly gross salary based on
      the annual CTC. Clearly explain that CTC, gross salary,
      and take-home salary are different.

   E. Salary factors:
      Explain the main factors that could increase or decrease
      the candidate's offer, including interview performance,
      coding ability, practical skills, employer type,
      location, and relevant experience.

   F. Salary confidence:
      State whether the estimate has high, medium, or low
      confidence and explain why.

   Salary prediction rules:
   - Treat all salary figures as estimates, not guarantees.
   - Use the Indian job market as the default market.
   - Do not assume the candidate is a fresher solely because
     they lack recent professional experience.
   - Distinguish total professional experience from relevant
     AI/ML experience.
   - Do not treat academic projects as equivalent to paid
     professional experience.
   - Do not automatically assign a lower salary solely because
     of a career gap.
   - Do not invent the candidate's location, certifications,
     experience, or employer offers.
   - If current salary benchmark data is unavailable, clearly
     state that the ranges are approximate and not verified
     against live salary data.
   - Explain the assumptions behind each salary range.

10. Recommendations
    - Provide actionable steps to improve employability
      and salary prospects.
    - Prioritize technical skills, portfolio improvements,
      interview preparation, and practical experience.
    - Recommend a realistic next step for the candidate.

General Instructions:
- Do not invent information that is not present in the resume.
- If information is missing, clearly write:
  "Not provided in the resume."
- Separate facts from assumptions and estimates.
- Use clear headings, bullet points, and tables where appropriate.
- Maintain an objective, professional, and constructive tone.
- Base all conclusions on the supplied resume.
- Never guarantee employment or a particular salary.

OUTPUT FORMAT REQUIREMENTS:

Return a valid JSON object only.
Do not return Markdown tables, headings, or text outside JSON.

Include exactly these keys:
summary, skills, education, experience, projects,
missing_skills, suitable_roles, employability,
salary_prediction, recommendation.

Use strings for summary, employability, and recommendation.
Use arrays of strings for all other fields.

Salary predictions must be estimates, not guaranteed offers.
Do not invent candidate qualifications or achievements.

Return a detailed, structured resume analysis.
""",
    input_variables=["resume"]
)



# Create Chain
# from langchain_core.output_parsers import StrOutputParser
# output = StrOutputParser()

chain = prompt | json_llm


# File Upload
uploaded_file = st.sidebar.file_uploader(
    "📤 Upload your Resume",
    type=["pdf"],
    help="Upload your resume in PDF format."
)


# Analyze Button
if uploaded_file is not None:

    st.sidebar.success(f"Uploaded: {uploaded_file.name}")

    if st.sidebar.button(
        "🔍 Analyze Resume",
        use_container_width=True
    ):

        temp_pdf_path = None

        with st.sidebar.spinner("Analyzing your resume..."):

            try:
                # Save uploaded PDF temporarily
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as temp_file:

                    temp_file.write(uploaded_file.getbuffer())
                    temp_pdf_path = temp_file.name

                # Load PDF
                loader = PyPDFLoader(temp_pdf_path)
                documents = loader.load()

                # Extract text
                resume_text = "\n".join(
                    doc.page_content for doc in documents
                )

                if not resume_text.strip():
                    st.error(
                        "Could not extract text from this PDF. "
                        "Please upload a text-based PDF."
                    )
                    st.stop()

                # Invoke LangChain chain
                response = chain.invoke({
                    "resume": resume_text
                })

                # Convert JSON text into a Python dictionary
                result = json.loads(response.content)

                # Validate required fields
                required_keys = [
                    "summary",
                    "skills",
                    "education",
                    "experience",
                    "projects",
                    "missing_skills",
                    "suitable_roles",
                    "employability",
                    "salary_prediction",
                    "recommendation"
                ]

                missing_keys = [
                    key for key in required_keys
                    if key not in result
                ]

                if missing_keys:
                    st.error(
                        f"Model output is missing fields: {missing_keys}"
                    )
                    st.stop()

                # Store results for your existing display section
                st.session_state["analysis_result"] = result

                st.success("Resume analysis completed!")

            except json.JSONDecodeError:
                st.error(
                    "The model did not return valid JSON. "
                    "Please try again."
                )

            except Exception as e:
                st.error(f"Something went wrong: {e}")

            finally:
                # Remove temporary PDF
                if temp_pdf_path and os.path.exists(temp_pdf_path):
                    os.remove(temp_pdf_path)


# Display results outside the Analyze button block
if "analysis_result" in st.session_state:

    result = st.session_state["analysis_result"]



# Display Results outside the Analyze Button Block

if "analysis_result" in st.session_state:

    result = st.session_state["analysis_result"]

    # Candidate Summary
    st.markdown(
        '<div class="section-title">👤 Candidate Summary</div>',
        unsafe_allow_html=True
    )
    st.info(result["summary"])

    # Skills
    st.markdown(
        '<div class="section-title">🛠️ Technical & Soft Skills</div>',
        unsafe_allow_html=True
    )

    cols = st.columns(3)

    for i, skill in enumerate(result["skills"]):
        with cols[i % 3]:
            st.success(f"✓ {skill}")

    # Education
    st.markdown(
        '<div class="section-title">🎓 Education</div>',
        unsafe_allow_html=True
    )

    for education in result["education"]:
        st.write(f"• {education}")

    # Experience
    st.markdown(
        '<div class="section-title">💼 Work Experience</div>',
        unsafe_allow_html=True
    )

    for experience in result["experience"]:
        st.write(f"• {experience}")

    # Projects
    st.markdown(
        '<div class="section-title">🚀 Projects</div>',
        unsafe_allow_html=True
    )

    for project in result["projects"]:
        st.write(f"• {project}")

    # Missing Skills
    st.markdown(
        '<div class="section-title">⚠️ Missing / Weak Skills</div>',
        unsafe_allow_html=True
    )

    if result["missing_skills"]:
        for skill in result["missing_skills"]:
            st.warning(skill)
    else:
        st.success("No major missing skills identified.")

    # Suitable Job Roles
    st.markdown(
        '<div class="section-title">🎯 Suitable Job Roles</div>',
        unsafe_allow_html=True
    )

    for role in result["suitable_roles"]:
        st.write(f"• {role}")

    # Overall Employability
    st.markdown(
        '<div class="section-title">📊 Overall Employability</div>',
        unsafe_allow_html=True
    )

    st.info(result["employability"])

    # Salary Prediction
    st.markdown(
        '<div class="section-title">💰 Estimated Salary Prediction</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Estimated Indian job-market compensation based on the resume. "
        "These figures are estimates, not guaranteed offers."
    )

    for prediction in result["salary_prediction"]:
        st.write(f"• {prediction}")

    # HR Recommendation
    st.markdown(
        '<div class="section-title">💡 HR Recommendation</div>',
        unsafe_allow_html=True
    )

    st.info(result["recommendation"])

    # Success Message
    st.success("✅ Resume analysis completed successfully!")
