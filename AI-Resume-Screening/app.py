import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# PAGE SETUP
# =========================================================

st.set_page_config(
    page_title="AI Resume Screening",
    page_icon="🤖",
    layout="wide"
)


# =========================================================
# FUTURISTIC DESIGN
# =========================================================

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(circle at 50% 20%, #20205c 0%, transparent 30%),
        radial-gradient(circle at 10% 70%, #17134c 0%, transparent 28%),
        radial-gradient(circle at 90% 70%, #092c50 0%, transparent 28%),
        #03050f;
    color: white;
}

.block-container {
    max-width: 1250px;
    padding-top: 35px;
}

/* HEADER */

.header {
    text-align: center;
    padding: 20px 10px 5px;
}

.header-title {
    font-size: 52px;
    font-weight: 800;
    letter-spacing: 3px;
    background: linear-gradient(
        90deg,
        #ffffff,
        #a78bfa,
        #60a5fa,
        #ffffff
    );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.header-subtitle {
    color: #8993b5;
    font-size: 17px;
    letter-spacing: 1px;
}

/* ORB */

.orb-area {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 300px;
}

.orb {
    width: 170px;
    height: 170px;
    border-radius: 50%;

    background:
        radial-gradient(
            circle at 35% 30%,
            #ffffff 0%,
            #c4b5fd 8%,
            #8b5cf6 25%,
            #4c1d95 50%,
            #111044 75%,
            #050615 100%
        );

    box-shadow:
        0 0 25px #8b5cf6,
        0 0 65px #6366f1,
        0 0 120px rgba(59,130,246,0.55);

    position: relative;
}

.orb:before {
    content: "";
    position: absolute;
    inset: -22px;
    border-radius: 50%;
    border: 1px solid rgba(139,92,246,0.45);
    box-shadow:
        0 0 25px rgba(139,92,246,0.35),
        inset 0 0 25px rgba(96,165,250,0.25);
}

.orb:after {
    content: "";
    position: absolute;
    inset: -45px;
    border-radius: 50%;
    border: 1px solid rgba(96,165,250,0.18);
}

/* GLASS CARDS */

.card {
    background: rgba(13,17,38,0.72);
    border: 1px solid rgba(139,92,246,0.25);
    border-radius: 24px;
    padding: 25px;
    min-height: 260px;

    box-shadow:
        0 15px 50px rgba(0,0,0,0.35),
        inset 0 0 35px rgba(139,92,246,0.035);

    backdrop-filter: blur(18px);
}

.card-title {
    font-size: 21px;
    font-weight: 700;
    margin-bottom: 5px;
}

.card-description {
    color: #7f89aa;
    font-size: 14px;
    margin-bottom: 20px;
}

/* UPLOADER */

[data-testid="stFileUploaderDropzone"] {
    background: rgba(5,8,24,0.65) !important;
    border: 1px dashed rgba(139,92,246,0.65) !important;
    border-radius: 18px !important;
}

/* TEXT AREA */

textarea {
    background: rgba(4,7,20,0.85) !important;
    color: white !important;
    border-radius: 15px !important;
    border: 1px solid rgba(139,92,246,0.35) !important;
}

/* BUTTON */

.stButton > button {
    background: linear-gradient(
        90deg,
        #6d28d9,
        #4f46e5,
        #2563eb
    );

    color: white;
    border: none;
    border-radius: 16px;

    font-size: 18px;
    font-weight: 700;

    padding: 14px;

    box-shadow:
        0 0 20px rgba(99,102,241,0.35),
        0 0 50px rgba(124,58,237,0.15);

    transition: all 0.25s ease;
}

.stButton > button:hover {
    transform: translateY(-3px);

    box-shadow:
        0 0 30px rgba(139,92,246,0.6),
        0 0 70px rgba(59,130,246,0.25);
}

/* RESULT */

.score-box {
    text-align: center;
    padding: 35px;
    margin-top: 25px;

    background:
        radial-gradient(
            circle,
            rgba(99,102,241,0.20),
            rgba(8,11,30,0.75)
        );

    border: 1px solid rgba(139,92,246,0.35);
    border-radius: 28px;

    box-shadow:
        0 0 45px rgba(99,102,241,0.15);
}

.score {
    font-size: 65px;
    font-weight: 800;

    background: linear-gradient(
        90deg,
        #c4b5fd,
        #60a5fa
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.status {
    font-size: 22px;
    font-weight: 700;
}

/* METRIC */

[data-testid="stMetric"] {
    background: rgba(13,17,38,0.75);
    border: 1px solid rgba(139,92,246,0.2);
    border-radius: 18px;
    padding: 18px;
}

/* FOOTER */

.footer {
    text-align: center;
    color: #59627f;
    padding: 35px;
    font-size: 13px;
}
/* ==========================================================
   3D DECORATIVE VISUAL LAYER
   ========================================================== */

.three-d-layer {
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    overflow: hidden;
}

/* ---------- Floating Robot ---------- */

.robot-3d {
    position: fixed;
    left: 3%;
    top: 9%;
    width: 125px;
    height: 110px;
    transform: perspective(500px) rotateY(-12deg);
    filter: drop-shadow(0 0 25px rgba(75, 130, 255, 0.65));
    animation: robotFloat 4s ease-in-out infinite;
}

.robot-head {
    position: absolute;
    left: 23px;
    top: 10px;
    width: 78px;
    height: 65px;
    border-radius: 28px 28px 34px 34px;
    background:
        linear-gradient(145deg, #c7f4ff 0%, #6288ff 35%, #30218a 75%, #11134e 100%);
    border: 2px solid rgba(153, 232, 255, 0.75);
    box-shadow:
        inset -10px -10px 20px rgba(0,0,0,0.35),
        inset 8px 8px 15px rgba(255,255,255,0.35),
        0 0 30px rgba(66, 144, 255, 0.45);
}

.robot-face {
    position: absolute;
    left: 10px;
    top: 14px;
    width: 58px;
    height: 38px;
    border-radius: 20px;
    background:
        radial-gradient(circle at 27% 50%, #72eaff 0 5px, transparent 6px),
        radial-gradient(circle at 73% 50%, #72eaff 0 5px, transparent 6px),
        #071238;
    box-shadow:
        inset 0 0 12px rgba(0, 180, 255, 0.45),
        0 0 15px rgba(0, 204, 255, 0.25);
}

.robot-ear-left,
.robot-ear-right {
    position: absolute;
    top: 29px;
    width: 14px;
    height: 27px;
    border-radius: 10px;
    background: linear-gradient(#6b9eff, #3822a0);
    box-shadow: 0 0 15px rgba(90, 145, 255, 0.55);
}

.robot-ear-left {
    left: 9px;
}

.robot-ear-right {
    right: 9px;
}

.robot-body {
    position: absolute;
    left: 34px;
    top: 70px;
    width: 58px;
    height: 42px;
    border-radius: 22px 22px 14px 14px;
    background:
        linear-gradient(145deg, #769cff, #3e38a7 60%, #17165c);
    border: 1px solid rgba(129, 215, 255, 0.55);
    box-shadow: 0 0 25px rgba(78, 119, 255, 0.35);
}

.robot-glow {
    position: absolute;
    left: 57px;
    top: 83px;
    width: 13px;
    height: 13px;
    border-radius: 50%;
    background: #62ecff;
    box-shadow: 0 0 20px #2edbff;
}

/* ---------- Floating Resume ---------- */

.resume-3d {
    position: fixed;
    right: 4%;
    top: 8%;
    width: 125px;
    height: 155px;
    border-radius: 15px;
    transform: perspective(700px) rotateY(-15deg) rotateZ(6deg);
    background:
        linear-gradient(
            145deg,
            rgba(76, 121, 255, 0.88),
            rgba(117, 37, 224, 0.72)
        );
    border: 1px solid rgba(118, 226, 255, 0.65);
    box-shadow:
        inset 0 1px 2px rgba(255,255,255,0.35),
        0 0 35px rgba(96, 80, 255, 0.55);
    animation: resumeFloat 5s ease-in-out infinite;
}

.resume-3d::before {
    content: "";
    position: absolute;
    left: 15px;
    top: 15px;
    width: 94px;
    height: 122px;
    border-radius: 10px;
    background: rgba(7, 14, 52, 0.75);
    border: 1px solid rgba(146, 205, 255, 0.35);
}

.resume-person {
    position: absolute;
    left: 27px;
    top: 27px;
    width: 29px;
    height: 29px;
    border-radius: 50%;
    background: #d7f4ff;
    box-shadow: 0 0 15px rgba(99, 216, 255, 0.5);
}

.resume-line {
    position: absolute;
    height: 5px;
    border-radius: 10px;
    background: linear-gradient(90deg, #69e8ff, #b94cff);
    left: 27px;
}

.resume-line.one {
    top: 67px;
    width: 67px;
}

.resume-line.two {
    top: 81px;
    width: 55px;
}

.resume-line.three {
    top: 95px;
    width: 70px;
}

.resume-magnifier {
    position: absolute;
    right: -27px;
    bottom: 8px;
    width: 42px;
    height: 42px;
    border-radius: 50%;
    border: 8px solid #70dfff;
    box-shadow:
        0 0 20px rgba(67, 218, 255, 0.75);
}

.resume-magnifier::after {
    content: "";
    position: absolute;
    right: -22px;
    bottom: -15px;
    width: 25px;
    height: 8px;
    border-radius: 8px;
    background: #70dfff;
    transform: rotate(45deg);
}

/* ---------- PDF Floating Card ---------- */

.pdf-3d {
    position: fixed;
    left: 2%;
    top: 39%;
    width: 105px;
    height: 135px;
    border-radius: 15px;
    transform: rotate(-12deg) perspective(500px);
    background:
        linear-gradient(
            145deg,
            rgba(75, 97, 255, 0.9),
            rgba(101, 32, 194, 0.75)
        );
    border: 1px solid rgba(101, 224, 255, 0.7);
    box-shadow:
        0 0 30px rgba(89, 62, 255, 0.5);
    animation: pdfFloat 4.5s ease-in-out infinite;
}

.pdf-3d::before {
    content: "";
    position: absolute;
    left: 15px;
    top: 14px;
    width: 75px;
    height: 105px;
    border-radius: 9px;
    background: rgba(8, 17, 53, 0.72);
}

.pdf-label {
    position: absolute;
    left: 23px;
    top: 61px;
    padding: 6px 10px;
    border-radius: 7px;
    background: #ff3f8d;
    color: white;
    font-size: 12px;
    font-weight: 800;
    box-shadow: 0 0 18px rgba(255, 55, 150, 0.55);
}

.pdf-lines {
    position: absolute;
    left: 25px;
    top: 29px;
    width: 52px;
    height: 4px;
    border-radius: 10px;
    background: #8be9ff;
    box-shadow:
        0 11px 0 #687cff,
        0 65px 0 #687cff,
        0 76px 0 #687cff;
}

/* ---------- Target ---------- */

.target-3d {
    position: fixed;
    right: 1.5%;
    top: 35%;
    width: 82px;
    height: 82px;
    border-radius: 50%;
    background:
        radial-gradient(
            circle,
            #ff62e7 0 10%,
            #693aff 11% 27%,
            #1b7cff 28% 43%,
            #081748 44% 100%
        );
    border: 2px solid rgba(100, 220, 255, 0.6);
    box-shadow:
        0 0 35px rgba(87, 112, 255, 0.75);
    animation: targetPulse 2.5s ease-in-out infinite;
}

.target-arrow {
    position: absolute;
    left: -40px;
    top: 35px;
    width: 58px;
    height: 5px;
    border-radius: 10px;
    background: linear-gradient(90deg, #ff49ce, #67eaff);
    transform: rotate(-18deg);
    box-shadow: 0 0 15px #55dfff;
}

/* ---------- AI Brain ---------- */

.brain-3d {
    position: fixed;
    right: 2%;
    bottom: 11%;
    width: 170px;
    height: 170px;
    border-radius: 50%;
    background:
        radial-gradient(
            circle,
            rgba(85, 226, 255, 0.25),
            rgba(110, 39, 255, 0.18) 45%,
            transparent 70%
        );
    box-shadow:
        0 0 55px rgba(82, 100, 255, 0.30);
    animation: brainFloat 5s ease-in-out infinite;
}

.brain-3d::before {
    content: "🧠";
    position: absolute;
    left: 27px;
    top: 20px;
    font-size: 105px;
    filter:
        drop-shadow(0 0 10px #4ceaff)
        drop-shadow(0 0 25px #8d4dff);
}

.brain-platform {
    position: absolute;
    left: 18px;
    bottom: 12px;
    width: 135px;
    height: 16px;
    border-radius: 50%;
    background:
        linear-gradient(
            90deg,
            #5a22ff,
            #31dfff,
            #a52dff
        );
    box-shadow:
        0 0 25px rgba(64, 213, 255, 0.75);
}

/* ---------- Floating Cubes ---------- */

.cube-3d {
    position: fixed;
    width: 22px;
    height: 22px;
    transform: rotate(45deg);
    border: 1px solid rgba(94, 218, 255, 0.65);
    background:
        linear-gradient(
            135deg,
            rgba(68, 220, 255, 0.8),
            rgba(125, 39, 255, 0.65)
        );
    box-shadow:
        0 0 20px rgba(75, 161, 255, 0.55);
    animation: cubeFloat 5s ease-in-out infinite;
}

.cube-one {
    left: 26%;
    top: 7%;
}

.cube-two {
    right: 18%;
    top: 4%;
    width: 15px;
    height: 15px;
    animation-delay: 1s;
}

.cube-three {
    left: 1%;
    bottom: 8%;
    width: 35px;
    height: 35px;
    animation-delay: 2s;
}

.cube-four {
    right: 1%;
    bottom: 7%;
    width: 28px;
    height: 28px;
    animation-delay: 3s;
}

/* ---------- Neon Lines ---------- */

.neon-line {
    position: fixed;
    height: 1px;
    border-radius: 50%;
    background: linear-gradient(
        90deg,
        transparent,
        #42dfff,
        #a94cff,
        transparent
    );
    box-shadow:
        0 0 10px #4bcfff;
    opacity: 0.65;
}

.neon-line-one {
    width: 240px;
    left: 0;
    top: 54%;
}

.neon-line-two {
    width: 260px;
    right: 0;
    top: 56%;
}

.neon-line-three {
    width: 420px;
    left: 31%;
    bottom: 4%;
}

/* ---------- Animations ---------- */

@keyframes robotFloat {

    0%, 100% {
        transform:
            perspective(500px)
            rotateY(-12deg)
            translateY(0);
    }

    50% {
        transform:
            perspective(500px)
            rotateY(-12deg)
            translateY(-12px);
    }
}

@keyframes resumeFloat {

    0%, 100% {
        transform:
            perspective(700px)
            rotateY(-15deg)
            rotateZ(6deg)
            translateY(0);
    }

    50% {
        transform:
            perspective(700px)
            rotateY(-15deg)
            rotateZ(6deg)
            translateY(-15px);
    }
}

@keyframes pdfFloat {

    0%, 100% {
        transform:
            rotate(-12deg)
            translateY(0);
    }

    50% {
        transform:
            rotate(-8deg)
            translateY(-13px);
    }
}

@keyframes targetPulse {

    0%, 100% {
        transform: scale(1);
    }

    50% {
        transform: scale(1.08);
    }
}

@keyframes brainFloat {

    0%, 100% {
        transform: translateY(0);
    }

    50% {
        transform: translateY(-12px);
    }
}

@keyframes cubeFloat {

    0%, 100% {
        transform:
            rotate(45deg)
            translateY(0);
    }

    50% {
        transform:
            rotate(135deg)
            translateY(-15px);
    }
}

/* Keep Streamlit content above decorations */

.block-container {
    position: relative;
    z-index: 2;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="header">

<div class="header-title">
AI RESUME SCREENING
</div>

<div class="header-subtitle">
Smart Candidate Shortlisting System
</div>

</div>
""", unsafe_allow_html=True)


# =========================================================
# AI ORB
# =========================================================

st.markdown("""
<div class="orb-area">
    <div class="orb"></div>
</div>
""", unsafe_allow_html=True)

st.markdown(
    "<h3 style='text-align:center;'>AI Candidate Analyzer</h3>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;color:#7f89aa;'>"
    "Intelligent resume analysis powered by Natural Language Processing"
    "</p>",
    unsafe_allow_html=True
)

st.write("")


# =========================================================
# INPUT CARDS
# =========================================================

left, right = st.columns(2, gap="large")


with left:

    st.markdown("""
    <div class="card">

    <div class="card-title">
    📄 Resume Upload
    </div>

    <div class="card-description">
    Upload the candidate's resume in PDF format.
    </div>

    </div>
    """, unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Drop candidate resumes here",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        help="Upload up to 3 candidate resumes in PDF format."
    )

    if uploaded_files:
        if len(uploaded_files) > 3:
            st.warning("Please upload a maximum of 3 resumes at a time.")
        else:
            st.success(f"✓ {len(uploaded_files)} resume(s) uploaded")


with right:

    st.markdown("""
    <div class="card">

    <div class="card-title">
    💼 Job Requirements
    </div>

    <div class="card-description">
    Enter the skills and requirements for the job.
    </div>

    </div>
    """, unsafe_allow_html=True)

    job_description = st.text_area(
        "Job Description",
        height=160,
        placeholder=(
            "Example:\n"
            "Python developer with Machine Learning, SQL, "
            "Data Analysis and Git skills."
        ),
        label_visibility="collapsed"
    )


st.write("")
st.write("")


# =========================================================
# PDF READER
# =========================================================

def extract_resume_text(pdf_file):

    reader = PdfReader(pdf_file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# =========================================================
# AI MATCHING
# =========================================================

def calculate_match(resume_text, job_text):

    resume_lower = resume_text.lower()
    job_lower = job_text.lower()


    # =====================================================
    # 1. TF-IDF SIMILARITY
    # =====================================================

    documents = [
        resume_lower,
        job_lower
    ]

    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    vectors = vectorizer.fit_transform(documents)

    tfidf_similarity = cosine_similarity(
        vectors[0:1],
        vectors[1:2]
    )[0][0] * 100


    # =====================================================
    # 2. SKILL MATCHING
    # =====================================================

    skills = [
        "python",
        "sql",
        "machine learning",
        "data science",
        "data analysis",
        "artificial intelligence",
        "pandas",
        "numpy",
        "scikit-learn",
        "git",
        "github",
        "streamlit",
        "html",
        "css",
        "javascript",
        "java",
        "c++",
        "c",
        "matlab",
        "deep learning",
        "nlp",
        "natural language processing",
        "tensorflow",
        "pytorch",
        "excel",
        "mysql",
        "database",
        "power bi",
        "tableau"
    ]


    # Skills required by the job
    required_skills = []

    for skill in skills:

        if skill in job_lower:

            required_skills.append(skill)


    # Skills found in resume
    matched_skills = []

    for skill in required_skills:

        if skill in resume_lower:

            matched_skills.append(skill)


    # Skill match percentage
    if len(required_skills) > 0:

        skill_score = (
            len(matched_skills)
            / len(required_skills)
        ) * 100

    else:

        skill_score = 0


    # =====================================================
    # 3. FINAL SCORE
    # =====================================================

    if len(required_skills) > 0:

        final_score = (
            (skill_score * 0.70)
            +
            (tfidf_similarity * 0.30)
        )

    else:

        # If no recognized skills are entered,
        # use TF-IDF only.
        final_score = tfidf_similarity


    return (
        final_score,
        skill_score,
        tfidf_similarity,
        matched_skills,
        required_skills
    )


# =========================================================
# ANALYZE BUTTON
# =========================================================

button_col1, button_col2, button_col3 = st.columns(
    [1, 2, 1]
)

with button_col2:

    analyze = st.button(
        "✦  ANALYZE ALL RESUMES",
        use_container_width=True
    )


# =========================================================
# MULTI-CANDIDATE ANALYSIS
# =========================================================

if analyze:

    if not uploaded_files:

        st.warning(
            "Please upload at least one resume first."
        )

    elif len(uploaded_files) > 3:

        st.warning(
            "Please upload a maximum of 3 resumes at a time."
        )

    elif not job_description.strip():

        st.warning(
            "Please enter the job requirements first."
        )

    else:

        with st.spinner(
            "AI is analyzing all candidates..."
        ):

            candidate_results = []

            for uploaded_file in uploaded_files:

                try:

                    resume_text = extract_resume_text(
                        uploaded_file
                    )

                    if not resume_text.strip():
                        candidate_results.append({
                            "name": uploaded_file.name,
                            "score": 0.0,
                            "skill_score": 0.0,
                            "tfidf_score": 0.0,
                            "matched_skills": [],
                            "required_skills": [],
                            "resume_text": "",
                            "error": "No readable text was found in the PDF."
                        })
                        continue

                    (
                        score,
                        skill_score,
                        tfidf_score,
                        matched_skills,
                        required_skills
                    ) = calculate_match(
                        resume_text,
                        job_description
                    )

                    candidate_results.append({
                        "name": uploaded_file.name,
                        "score": score,
                        "skill_score": skill_score,
                        "tfidf_score": tfidf_score,
                        "matched_skills": matched_skills,
                        "required_skills": required_skills,
                        "resume_text": resume_text,
                        "error": None
                    })

                except Exception as error:

                    candidate_results.append({
                        "name": uploaded_file.name,
                        "score": 0.0,
                        "skill_score": 0.0,
                        "tfidf_score": 0.0,
                        "matched_skills": [],
                        "required_skills": [],
                        "resume_text": "",
                        "error": str(error)
                    })


            # Rank candidates from highest to lowest score.
            candidate_results.sort(
                key=lambda candidate: candidate["score"],
                reverse=True
            )


            st.divider()


            # =================================================
            # RESULT HEADING
            # =================================================

            st.markdown(
                "<h2 style='text-align:center;'>"
                "AI SCREENING RESULTS"
                "</h2>",
                unsafe_allow_html=True
            )

            st.markdown(
                "<p style='text-align:center;color:#7f89aa;'>"
                "Candidates ranked automatically by the AI matching score"
                "</p>",
                unsafe_allow_html=True
            )


            # =================================================
            # RANKING TABLE
            # =================================================

            st.subheader("🏆 Candidate Ranking")

            for rank, candidate in enumerate(candidate_results, start=1):

                score = candidate["score"]

                if score >= 60:
                    status = "SELECTED"
                    status_icon = "🟢"
                else:
                    status = "NOT SELECTED"
                    status_icon = "🔴"

                if rank == 1:
                    rank_icon = "🥇"
                elif rank == 2:
                    rank_icon = "🥈"
                elif rank == 3:
                    rank_icon = "🥉"
                else:
                    rank_icon = f"#{rank}"

                col1, col2, col3, col4 = st.columns(
                    [0.7, 2.5, 1.3, 1.5]
                )

                with col1:
                    st.markdown(f"### {rank_icon}")

                with col2:
                    st.markdown(
                        f"**{candidate['name']}**"
                    )

                with col3:
                    st.metric(
                        "Match",
                        f"{score:.2f}%"
                    )

                with col4:
                    st.markdown(
                        f"**{status_icon} {status}**"
                    )

                st.progress(
                    min(score / 100, 1.0)
                )


            # =================================================
            # CANDIDATE DETAILS
            # =================================================

            st.subheader("🔎 Candidate Details")

            for rank, candidate in enumerate(candidate_results, start=1):

                score = candidate["score"]
                skill_score = candidate["skill_score"]
                tfidf_score = candidate["tfidf_score"]

                if score >= 60:
                    status = "SELECTED"
                    status_icon = "🟢"
                else:
                    status = "NOT SELECTED"
                    status_icon = "🔴"

                with st.expander(
                    f"{rank}. {candidate['name']} — {score:.2f}% — {status}"
                ):

                    if candidate["error"]:

                        st.error(candidate["error"])
                        continue


                    # =================================================
                    # SCORE BOX
                    # =================================================

                    st.markdown(
                        f"""
                        <div class="score-box">

                        <div style="color:#8993b5;">
                        FINAL AI MATCH SCORE
                        </div>

                        <div class="score">
                        {score:.2f}%
                        </div>

                        <div class="status">
                        {status_icon} {status}
                        </div>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.write("")


                    # =================================================
                    # METRICS
                    # =================================================

                    c1, c2, c3 = st.columns(3)

                    with c1:
                        st.metric(
                            "Final Match Score",
                            f"{score:.2f}%"
                        )

                    with c2:
                        st.metric(
                            "Skill Match",
                            f"{skill_score:.2f}%"
                        )

                    with c3:
                        st.metric(
                            "TF-IDF Similarity",
                            f"{tfidf_score:.2f}%"
                        )


                    # =================================================
                    # SKILL ANALYSIS
                    # =================================================

                    st.write("### 🎯 Resume Compatibility")

                    st.progress(
                        min(score / 100, 1.0)
                    )

                    st.subheader(
                        "🛠️ Required Skill Analysis"
                    )

                    if candidate["required_skills"]:

                        st.write(
                            f"**Required skills detected:** "
                            f"{len(candidate['required_skills'])}"
                        )

                        st.write(
                            f"**Skills matched:** "
                            f"{len(candidate['matched_skills'])}"
                        )

                        for skill in candidate["required_skills"]:

                            if skill in candidate["matched_skills"]:
                                st.success(
                                    f"✓ {skill.title()} — Found in resume"
                                )
                            else:
                                st.error(
                                    f"✗ {skill.title()} — Not found in resume"
                                )

                    else:

                        st.info(
                            "No recognized technical skills were detected "
                            "in the job description. The system used "
                            "TF-IDF similarity."
                        )


                    # =================================================
                    # AI EXPLANATION
                    # =================================================

                    with st.container(border=True):

                        st.subheader(
                            "🧠 How the AI Evaluated This Candidate"
                        )

                        st.write(
                            "**1. Skill Matching — 70% weight**  \n"
                            "Checks whether the technical skills required "
                            "by the job are present in the resume."
                        )

                        st.write(
                            "**2. TF-IDF + Cosine Similarity — 30% weight**  \n"
                            "Measures the similarity between the resume text "
                            "and the job description."
                        )

                        st.info(
                            f"Skill Match: {skill_score:.2f}%  |  "
                            f"TF-IDF Similarity: {tfidf_score:.2f}%  |  "
                            f"Final Score: {score:.2f}%"
                        )


                    # =================================================
                    # EXTRACTED RESUME
                    # =================================================

                    with st.expander(
                        "📄 View Extracted Resume"
                    ):

                        st.text(
                            candidate["resume_text"][:5000]
                        )



# =========================================================
# FEATURES
# =========================================================

st.divider()

st.markdown(
    "<h2 style='text-align:center;'>Powerful AI Features</h2>",
    unsafe_allow_html=True
)

st.write("")


f1, f2, f3, f4 = st.columns(4)


with f1:

    st.markdown("### 📄")

    st.write(
        "**PDF Processing**"
    )

    st.caption(
        "Automatically extracts resume information."
    )


with f2:

    st.markdown("### 🧠")

    st.write(
        "**NLP Matching**"
    )

    st.caption(
        "Analyzes textual similarity intelligently."
    )


with f3:

    st.markdown("### 🎯")

    st.write(
        "**Match Scoring**"
    )

    st.caption(
        "Combines skill matching and TF-IDF similarity."
    )


with f4:

    st.markdown("### 🏆")

    st.write(
        "**Shortlisting**"
    )

    st.caption(
        "Helps identify suitable candidates."
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">

AI Resume Screening System
&nbsp; • &nbsp;
Python
&nbsp; • &nbsp;
Machine Learning
&nbsp; • &nbsp;
TF-IDF
&nbsp; • &nbsp;
Streamlit

</div>
""", unsafe_allow_html=True)
# =========================================================
# 3D FLOATING VISUALS
# =========================================================

three_d_html = """<div class="three-d-layer"><div class="robot-3d"><div class="robot-head"><div class="robot-face"></div></div><div class="robot-ear-left"></div><div class="robot-ear-right"></div><div class="robot-body"></div><div class="robot-glow"></div></div><div class="resume-3d"><div class="resume-person"></div><div class="resume-line one"></div><div class="resume-line two"></div><div class="resume-line three"></div><div class="resume-magnifier"></div></div><div class="pdf-3d"><div class="pdf-lines"></div><div class="pdf-label">PDF</div></div><div class="target-3d"><div class="target-arrow"></div></div><div class="brain-3d"><div class="brain-platform"></div></div><div class="cube-3d cube-one"></div><div class="cube-3d cube-two"></div><div class="cube-3d cube-three"></div><div class="cube-3d cube-four"></div><div class="neon-line neon-line-one"></div><div class="neon-line neon-line-two"></div><div class="neon-line neon-line-three"></div></div>"""

st.markdown(three_d_html, unsafe_allow_html=True)