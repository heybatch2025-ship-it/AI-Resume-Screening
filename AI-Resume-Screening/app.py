
import streamlit as st
from pypdf import PdfReader
import re
import io
import joblib
import csv
from datetime import datetime
from sentence_transformers import SentenceTransformer, util

# Optional multilingual utilities. The app still runs if these packages are absent.
try:
    from langdetect import detect as detect_language
except ImportError:
    detect_language = None

# Local translation is optional and loaded only when the user requests it.
# Google Translate / deep-translator is intentionally not used.

@st.cache_resource(show_spinner=False)
def get_translation_pipeline(model_name):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return tokenizer, model


def translate_text(text, model_name):
    import torch

    tokenizer, model = get_translation_pipeline(model_name)
    chunks = [
        text[i:i + 300]
        for i in range(0, len(text), 300)
        if text[i:i + 300].strip()
    ]

    translated_parts = []

    for chunk in chunks:
        inputs = tokenizer(
            chunk,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=256)

        translated_parts.append(
            tokenizer.decode(output[0], skip_special_tokens=True)
        )

    return "\n".join(translated_parts)

@st.cache_resource
def get_bert_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

bert_model = get_bert_model()

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

/* Hide Streamlit Cloud top header */
[data-testid="stHeader"] {
    display: none !important;
}

[data-testid="stToolbar"] {
    display: none !important;
}

/* Remove Streamlit decoration */
[data-testid="stDecoration"] {
    display: none !important;
}

.block-container {
    max-width: 1250px;
    padding-top: 35px;
}

/* HEADER */

.header {
    text-align: center;
    padding: 15px 10px 8px;

    position: fixed;
    top: 0;
    left: 0;
    width: 100%;

    z-index: 999999;

    background: rgba(5, 8, 25, 0.96);
    backdrop-filter: blur(15px);
    -webkit-backdrop-filter: blur(15px);

    box-sizing: border-box;
    
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
/* RESUME VISUAL */

.resume-visual {
    height: 155px;
    display: flex;
    justify-content: center;
    align-items: center;
    position: relative;
    margin-top: 5px;
}

.resume-card-3d {
    width: 145px;
    height: 105px;
    border-radius: 14px;
    background: linear-gradient(145deg, #263bff, #713cff, #151b61);
    border: 2px solid #55dfff;
    box-shadow:
        0 0 25px #514cff,
        0 0 50px rgba(90,70,255,0.35);
    position: relative;
}

.resume-card-3d::before {
    content: "RESUME";
    position: absolute;
    top: 25px;
    left: 20px;
    color: white;
    font-size: 18px;
    font-weight: 700;
}

.resume-card-3d::after {
    content: "PDF";
    position: absolute;
    right: -18px;
    bottom: -12px;
    background: linear-gradient(135deg, #ff3d91, #7d4cff);
    color: white;
    font-weight: 800;
    padding: 8px 12px;
    border-radius: 8px;
    box-shadow: 0 0 18px #ff3d91;
}

.upload-cloud {
    position: absolute;
    right: 24%;
    top: 12px;
    font-size: 42px;
    filter: drop-shadow(0 0 12px #54dfff);
}

/* JOB REQUIREMENTS VISUAL */

.job-visual {
    height: 155px;
    display: flex;
    justify-content: center;
    align-items: center;
    position: relative;
    margin-top: 5px;
}

.job-board {
    width: 150px;
    height: 110px;
    border-radius: 14px;
    background: linear-gradient(145deg, #182b70, #5130a8, #10183f);
    border: 2px solid #66eaff;
    box-shadow:
        0 0 25px #514cff,
        0 0 50px rgba(90,70,255,0.35);
    position: relative;
    padding: 18px;
    box-sizing: border-box;
}

.job-board::before {
    content: "JOB";
    position: absolute;
    top: 15px;
    left: 20px;
    color: white;
    font-size: 18px;
    font-weight: 800;
}

.job-board::after {
    content: "✓  Python     ✓  SQL     ✓  ML";
    white-space: pre;
    position: absolute;
    left: 20px;
    top: 48px;
    color: #8be9ff;
    font-size: 9px;
    line-height: 20px;
}

.job-icon {
    position: absolute;
    right: 18%;
    top: 8px;
    font-size: 43px;
    filter: drop-shadow(0 0 13px #9d5cff);
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
    left: 8%;
    top: 18%;
    width: 32px;
    height: 32px;
    border-radius: 10px;
}
.cube-two {
    right: 8%;
    top: 20%;
    width: 28px;
    height: 28px;
    border-radius: 8px;
    animation-delay: 1s;
}


.cube-three {
    left: 4%;
    bottom: 12%;
    width: 30px;
    height: 30px;
    border-radius: 9px;
    animation-delay: 2s;
}

.cube-four {
    right: 4%;
    bottom: 12%;
    width: 32px;
    height: 32px;
    border-radius: 10px;
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
    <div class="resume-visual">
        <div class="resume-card-3d"></div>
        <div class="upload-cloud">☁️</div>
    </div>

    <div style="
        text-align:center;
        color:#b8c8ff;
        font-size:15px;
        margin-bottom:12px;
    ">
        Upload your resume
        <br>
        <span style="font-size:12px;color:#7785ad;">
            PDF format supported
        </span>
    </div>
    """, unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Drop candidate resumes here",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        help="Upload up to 10 candidate resumes in PDF format."
    )

    if uploaded_files:
        if len(uploaded_files) > 10:
            st.warning("Please upload a maximum of 10 resumes at a time.")
        else:
            st.success(f"✓ {len(uploaded_files)} resume(s) uploaded")

with right:

    

    st.markdown("""
    <div class="job-visual">
        <div class="job-board"></div>
        <div class="job-icon">💼</div>
    </div>

    <div style="
        text-align:center;
        color:#b8c8ff;
        font-size:15px;
        margin-bottom:12px;
    ">
        Define job requirements
        <br>
        <span style="font-size:12px;color:#7785ad;">
            Skills • Experience • Technologies
        </span>
    </div>
    """, unsafe_allow_html=True)

    job_description = st.text_area(
        "Job Description",
        height=100,
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
# TRAINED RESUME CATEGORY MODEL
# =========================================================

from pathlib import Path

@st.cache_resource
def load_resume_category_model():
    model_path = Path(__file__).parent / "resume_category_model.pkl"

    if model_path.exists():
        return joblib.load(model_path)

    return None


resume_category_model = load_resume_category_model()
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

# A candidate is shortlisted at 50% or above.
# This avoids rejecting candidates who match most of the required skills
# but use different wording in their resume.
SELECTION_THRESHOLD = 65


def calculate_match(resume_text, job_text):

    resume_lower = resume_text.lower()
    job_lower = job_text.lower()

    # =====================================================
    # 1. BERT SEMANTIC SIMILARITY
    # =====================================================

    resume_embedding = bert_model.encode(
        resume_text,
        convert_to_tensor=True
    )

    job_embedding = bert_model.encode(
        job_text,
        convert_to_tensor=True
    )

    bert_similarity = util.cos_sim(
        resume_embedding,
        job_embedding
    ).item() * 100

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

    required_skills = []

    for skill in skills:

        pattern = r"\b" + re.escape(skill) + r"\b"

        if re.search(pattern, job_lower):

            required_skills.append(skill)

    matched_skills = []

    for skill in required_skills:

        pattern = r"\b" + re.escape(skill) + r"\b"

        if re.search(pattern, resume_lower):

            matched_skills.append(skill)

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
            (skill_score * 0.85)
            +
            (bert_similarity * 0.15)
        )

    else:

        final_score = bert_similarity

    return (
        final_score,
        skill_score,
        bert_similarity,
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

    elif len(uploaded_files) > 10:

        st.warning(
            "Please upload a maximum of 10 resumes at a time."
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
                                        # Predict resume category
                    predicted_category = "Not available"

                    if resume_category_model is not None:
                        try:
                            cleaned_resume = re.sub(
                                r"\s+",
                                " ",
                                re.sub(
                                    r"[^a-zA-Z\s]",
                                    " ",
                                    resume_text.lower()
                                )
                            ).strip()

                            if cleaned_resume:
                                predicted_category = (
                                    resume_category_model.predict(
                                        [cleaned_resume]
                                    )[0]
                                )

                        except Exception:
                            predicted_category = "Prediction unavailable"
                    
                    if not resume_text.strip():
                        candidate_results.append({
                            "name": uploaded_file.name,
                            "score": 0.0,
                            "skill_score": 0.0,
                            "bert_score": 0.0,
                            "matched_skills": [],
                            "required_skills": [],
                            "resume_text": "",
                            "error": "No readable text was found in the PDF."
                        })
                        continue

                    (
                        score,
                        skill_score,
                        bert_score,
                        matched_skills,
                        required_skills
                    ) = calculate_match(
                        resume_text,
                        job_description
                    )

                    candidate_results.append({
                        "name": uploaded_file.name,
                        "predicted_category": str(predicted_category),
                        "score": score,
                        "skill_score": skill_score,
                        "bert_score": bert_score,
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
                        "bert_score": 0.0,
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

            # Keep results available to the add-on tools after Streamlit reruns.
            st.session_state["candidate_results"] = candidate_results
            st.session_state["job_description"] = job_description

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

                if score >= SELECTION_THRESHOLD:
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
                bert_score = candidate["bert_score"]

                if score >= SELECTION_THRESHOLD:
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
                            "BERT Similarity",
                            f"{bert_score:.2f}%"
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
                            "BERT semantic similarity."

                        )


                    # =================================================
                    # AI EXPLANATION
                    # =================================================

                    with st.container(border=True):

                        st.subheader(
                            "🧠 How the AI Evaluated This Candidate"
                        )

                        st.write(
                            "**1. Skill Matching — 85% weight**  \n"
                            "Checks whether the technical skills required "
                            "by the job are present in the resume."
                        )

                        st.write(
                                "**2. BERT Semantic Similarity — 15% weight**  \n"
                            "Measures the similarity between the resume text "
                            "and the job description."
                        )

                        st.info(
                            f"Skill Match: {skill_score:.2f}%  |  "
                            f"BERT Similarity: {bert_score:.2f}%  |  "
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
        "Combines skill matching and BERT semantic similarity."
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
# ADVANCED AI FEATURES / ADD-ONS
# =========================================================

# Results are stored in session state so add-on tools remain available
# after Streamlit reruns.
candidate_results = st.session_state.get("candidate_results", [])
saved_job_description = st.session_state.get("job_description", job_description or "")

# =========================================================
# DASHBOARD + DOWNLOADABLE REPORTS
# =========================================================
st.markdown("<h2 style='text-align:center;'>Screening Dashboard</h2>", unsafe_allow_html=True)
if not candidate_results:
    st.info("Upload candidate PDF resumes and click ANALYZE ALL RESUMES to populate the dashboard and report downloads.")
else:
    valid_results = [c for c in candidate_results if not c.get("error")]
    total_candidates = len(candidate_results)
    analyzed_count = len(valid_results)
    selected_count = sum(1 for c in valid_results if float(c.get("score", 0) or 0) >= SELECTION_THRESHOLD)
    average_score = (sum(float(c.get("score", 0) or 0) for c in valid_results) / analyzed_count) if analyzed_count else 0.0
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Resumes uploaded", total_candidates)
    d2.metric("Successfully analyzed", analyzed_count)
    d3.metric("Above threshold", selected_count)
    d4.metric("Average match score", f"{average_score:.1f}%")

    st.subheader("Download screening report")
    st.caption("Exports contain candidate file names and automated match metrics. Review results manually before making hiring decisions.")
    report_rows = []
    for rank, candidate in enumerate(candidate_results, start=1):
        score = float(candidate.get("score", 0) or 0)
        report_rows.append({
            "Rank": rank,
            "Candidate": candidate.get("name", "Unknown"),
            "Predicted Category": candidate.get("predicted_category", "Not available"),
            "Match Score (%)": round(score, 2),
            "Skill Score (%)": round(float(candidate.get("skill_score", 0) or 0), 2),
            "Semantic Similarity (%)": round(float(candidate.get("bert_score", 0) or 0), 2),
            "Status": "Above threshold" if score >= SELECTION_THRESHOLD and not candidate.get("error") else "Review / below threshold",
            "Matched Skills": ", ".join(candidate.get("matched_skills", []) or []),
            "Missing Required Skills": ", ".join([x for x in (candidate.get("required_skills", []) or []) if x not in (candidate.get("matched_skills", []) or [])]),
            "Processing Note": candidate.get("error", "") or ""
        })
        
    st.subheader("🤖 ML-Predicted Resume Categories")

    category_rows = []

    for candidate in candidate_results:
        category_rows.append({
            "Candidate": candidate.get("name", "Unknown"),
            "Predicted Category": candidate.get(
                "predicted_category", "Unavailable"
            )
        })

    st.dataframe(
        category_rows,
        use_container_width=True,
        hide_index=True
    )        
    csv_buffer = io.StringIO()
    csv_writer = csv.DictWriter(csv_buffer, fieldnames=list(report_rows[0].keys()))
    csv_writer.writeheader()
    csv_writer.writerows(report_rows)
    report_stamp = datetime.now().strftime("%Y%m%d_%H%M")
    dl1, dl2, dl3 = st.columns(3)
    with dl1:
        st.download_button("Download CSV", data=csv_buffer.getvalue().encode("utf-8-sig"), file_name=f"resume_screening_{report_stamp}.csv", mime="text/csv", use_container_width=True)
    with dl2:
        text_report = ["AI RESUME SCREENING REPORT", f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", f"Threshold: {SELECTION_THRESHOLD}%", "", "Note: Automated scores are decision-support signals, not hiring decisions.", ""]
        for row in report_rows:
            text_report.extend([f"{row['Rank']}. {row['Candidate']}", f"   Match: {row['Match Score (%)']}% | Skills: {row['Skill Score (%)']}% | Similarity: {row['Semantic Similarity (%)']}%", f"   Status: {row['Status']}", f"   Matched skills: {row['Matched Skills']}", f"   Missing skills: {row['Missing Required Skills']}", f"   Note: {row['Processing Note']}", ""])
        st.download_button("Download TXT report", data="\n".join(text_report).encode("utf-8"), file_name=f"resume_screening_{report_stamp}.txt", mime="text/plain", use_container_width=True)
    with dl3:
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            from reportlab.lib import colors
            from reportlab.lib.styles import getSampleStyleSheet
            pdf_buffer = io.BytesIO()
            doc = SimpleDocTemplate(pdf_buffer, pagesize=A4, title="AI Resume Screening Report")
            styles = getSampleStyleSheet()
            story = [Paragraph("AI Resume Screening Report", styles["Title"]), Spacer(1, 8), Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Threshold: {SELECTION_THRESHOLD}%", styles["Normal"]), Spacer(1, 12), Paragraph("Automated scores are decision-support signals. Review candidates manually before hiring decisions.", styles["Normal"]), Spacer(1, 12)]
            table_data = [["Rank", "Candidate", "Match", "Skills", "Status"]]
            for row in report_rows:
                safe_name = str(row["Candidate"]).encode("latin-1", "replace").decode("latin-1")
                table_data.append([str(row["Rank"]), safe_name[:35], f"{row['Match Score (%)']}%", f"{row['Skill Score (%)']}%", row["Status"]])
            table = Table(table_data, repeatRows=1, colWidths=[35, 190, 60, 60, 110])
            table.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#29245c")), ("TEXTCOLOR", (0,0), (-1,0), colors.white), ("GRID", (0,0), (-1,-1), 0.4, colors.grey), ("VALIGN", (0,0), (-1,-1), "TOP"), ("FONTSIZE", (0,0), (-1,-1), 8), ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f1f2f8")])]))
            story.append(table)
            doc.build(story)
            st.download_button("Download PDF report", data=pdf_buffer.getvalue(), file_name=f"resume_screening_{report_stamp}.pdf", mime="application/pdf", use_container_width=True)
        except Exception as exc:
            st.warning("PDF export is unavailable in this environment. Use CSV or TXT report instead.")

st.divider()
st.markdown(
    "<h2 style='text-align:center;'>Advanced Screening Toolkit</h2>",
    unsafe_allow_html=True
)
st.caption(
    "Prototype tools for recruiter assistance. Scores and flags are decision-support "
    "signals, not automatic proof of candidate ability, honesty, or suitability."
)

addon_tabs = st.tabs([
    "⚖️ Custom Rubric",
    "🌐 Multilingual",
    "💬 AI Interview",
    "📝 Resume Coach",
    "🛡️ Integrity Checks",
    "📨 Candidate Messages",
    "🤝 Communication Skills",
])

with addon_tabs[0]:
    st.subheader("Custom, role-specific scoring")
    st.write(
        "Adjust the weights for each candidate. Experience, education, and coding "
        "scores must be entered by a recruiter from verified, job-related evidence."
    )
    if not candidate_results:
        st.info("Upload resumes and click ANALYZE ALL RESUMES first.")
    else:
        names = [c["name"] for c in candidate_results]
        selected_name = st.selectbox(
            "Choose a candidate for rubric scoring",
            names,
            key="rubric_candidate"
        )
        selected_candidate = next(c for c in candidate_results if c["name"] == selected_name)

        rw1, rw2 = st.columns(2)
        with rw1:
            w_skills = st.slider("Skills match weight (%)", 0, 100, 50, key="rubric_w_skills")
            w_experience = st.slider("Relevant experience weight (%)", 0, 100, 20, key="rubric_w_experience")
        with rw2:
            w_education = st.slider("Required education/certification weight (%)", 0, 100, 10, key="rubric_w_education")
            w_coding = st.slider("Coding/work-sample test weight (%)", 0, 100, 20, key="rubric_w_coding")

        weight_sum = w_skills + w_experience + w_education + w_coding
        st.write(f"**Weight total:** {weight_sum}% (weights are normalized automatically)")
        if weight_sum == 0:
            st.warning("At least one weight must be greater than zero.")
        else:
            st.caption("Enter scores from evidence; leave unassessed categories at 0 weight if not applicable.")
            score_key = re.sub(r"[^a-zA-Z0-9]+", "_", selected_name)
            e1, e2, e3 = st.columns(3)
            with e1:
                experience_score = st.slider(
                    "Experience evidence score", 0, 100, 50,
                    key=f"exp_score_{score_key}"
                )
            with e2:
                education_score = st.slider(
                    "Education/certification score", 0, 100, 50,
                    key=f"edu_score_{score_key}"
                )
            with e3:
                coding_score = st.slider(
                    "Coding/work-sample score", 0, 100, 50,
                    key=f"coding_score_{score_key}"
                )

            components = [
                (selected_candidate.get("skill_score", 0), w_skills),
                (experience_score, w_experience),
                (education_score, w_education),
                (coding_score, w_coding),
            ]
            custom_score = sum(float(value) * weight for value, weight in components) / weight_sum
            st.metric("Custom rubric score", f"{custom_score:.2f}/100")
            st.progress(max(0.0, min(custom_score / 100, 1.0)))
            st.info(
                "This is a separate rubric score and does not overwrite the original AI match score. "
                "Review the evidence and apply the same rubric consistently to all candidates."
            )


with addon_tabs[1]:
    st.subheader("🌐 Multilingual Resume Screening")

    if not candidate_results:
        st.info("Analyze resumes first to use multilingual tools.")
    else:
        names = [c["name"] for c in candidate_results]
        selected_name = st.selectbox(
            "Choose resume", names, key="language_candidate"
        )
        candidate = next(
            c for c in candidate_results if c["name"] == selected_name
        )
        resume_text = candidate.get("resume_text", "")

        if not resume_text.strip():
            st.warning("No readable text was extracted from this PDF.")
        else:
            st.text_area(
                "Original resume preview",
                resume_text[:4000],
                height=180,
                key="multilingual_original"
            )

            if detect_language is not None:
                try:
                    detected = detect_language(resume_text[:3000])
                    st.success(f"Detected language: {detected}")
                except Exception:
                    st.info("Language detection was inconclusive.")

            languages = {
                "English": "en",
                "Hindi": "hi",
                "Spanish": "es",
                "French": "fr"
            }

            translation_models = {
                ("en", "hi"): "Helsinki-NLP/opus-mt-en-hi",
                ("hi", "en"): "Helsinki-NLP/opus-mt-hi-en",
                ("en", "es"): "Helsinki-NLP/opus-mt-en-es",
                ("es", "en"): "Helsinki-NLP/opus-mt-es-en",
                ("en", "fr"): "Helsinki-NLP/opus-mt-en-fr",
                ("fr", "en"): "Helsinki-NLP/opus-mt-fr-en",
            }

            col1, col2 = st.columns(2)

            with col1:
                source = st.selectbox(
                    "Resume language",
                    list(languages.keys()),
                    key="translation_source"
                )

            with col2:
                target = st.selectbox(
                    "Translate into",
                    [lang for lang in languages if lang != source],
                    key="translation_target"
                )

            model_name = translation_models.get(
                (languages[source], languages[target])
            )

            st.caption(
                "Choose the language the resume is actually written in. "
                "The translation model downloads on first use."
            )

            if st.button("Translate full resume", key="translate_resume"):
                try:
                    with st.spinner("Loading model and translating resume..."):
                        translated = translate_text(resume_text, model_name)

                    if translated.strip():
                        st.text_area(
                            f"Translation — {target}",
                            translated,
                            height=300,
                            key="translated_resume_output"
                        )
                        st.download_button(
                            "Download translated resume",
                            data=translated.encode("utf-8"),
                            file_name="translated_resume.txt",
                            mime="text/plain",
                            key="download_translation"
                        )
                    else:
                        st.warning("The model returned empty text.")

                except Exception as exc:
                    st.error(
                        "Translation failed. Check internet access, "
                        "installed dependencies, and model availability."
                    )
                    st.caption(str(exc))

            st.caption(
                "Translation may not be perfect. Verify names, dates, "
                "qualifications, and technical terms manually."
            )



with addon_tabs[2]:
    st.subheader("AI Interview / Voice Prototype")

    st.write(
        "Choose a suggested question or write your own question. "
        "Enter the candidate's answer to get a preliminary coverage score."
    )

    interview_questions = [
        "Describe a relevant project you worked on and your individual contribution.",
        "How would you approach a difficult problem related to this role?",
        "Which tools or technologies from the job description have you used, and how?",
    ]

    st.markdown("**Suggested questions**")
    for i, question in enumerate(interview_questions, 1):
        st.write(f"{i}. {question}")

    if candidate_results:
        interview_name = st.selectbox(
            "Candidate for interview",
            [c["name"] for c in candidate_results],
            key="interview_candidate"
        )

        question_mode = st.radio(
            "Choose question source",
            ["Suggested question", "Write my own question"],
            horizontal=True,
            key="interview_question_mode"
        )

        if question_mode == "Suggested question":
            interview_question = st.selectbox(
                "Select interview question",
                interview_questions,
                key="interview_question"
            )
        else:
            interview_question = st.text_area(
                "Type your own interview question",
                key="custom_interview_question",
                placeholder="Example: Explain how you used Python in your project.",
                height=100
            )

        st.markdown("**Selected interview question**")
        if interview_question.strip():
            st.info(interview_question)
        else:
            st.warning("Please enter an interview question.")

        interview_answer = st.text_area(
            "Candidate answer",
            height=160,
            key="interview_answer",
            placeholder="Type or paste the candidate's answer here..."
        )

        if st.button("Evaluate answer", key="evaluate_interview"):
            if not interview_question.strip():
                st.warning("Please select or type an interview question.")
            elif not interview_answer.strip():
                st.warning("Please enter the candidate's answer.")
            else:
                answer_words = re.findall(
                    r"\b\w+\b", interview_answer.lower()
                )

                job_terms = set(re.findall(
                    r"\b[a-zA-Z][a-zA-Z+#.-]{2,}\b",
                    saved_job_description.lower()
                ))

                answer_terms = set(answer_words)
                relevant_terms = answer_terms.intersection(job_terms)

                length_score = min(len(answer_words) / 80, 1) * 40
                relevance_score = min(
                    len(relevant_terms) / max(len(job_terms), 1) * 8,
                    40
                )

                structure_score = (
                    20 if any(token in interview_answer.lower() for token in [
                        "because", "result", "built", "developed",
                        "improved", "learned", "implemented", "created"
                    ]) else 8
                )

                indicator = min(
                    100, length_score + relevance_score + structure_score
                )

                st.metric(
                    "Preliminary answer coverage indicator",
                    f"{indicator:.0f}/100"
                )

                st.markdown("**Question asked**")
                st.write(interview_question)

                st.markdown("**Candidate answer**")
                st.write(interview_answer)

                st.markdown("**Potential job-description terms mentioned**")
                st.write(
                    ", ".join(sorted(relevant_terms))
                    if relevant_terms else "No matching terms detected."
                )

                st.caption(
                    "This is a rule-based indicator, not an AI judgment of "
                    "answer quality. Recruiters should assess correctness, "
                    "reasoning, examples, and context themselves."
                )

    else:
        st.info(
            "Analyze resumes first to associate interview questions with a candidate."
        )

    st.markdown("---")
    st.markdown("**Optional voice capture**")

    if hasattr(st, "audio_input"):
        audio_recording = st.audio_input(
            "Record a short answer (if supported by your browser)"
        )

        if audio_recording is not None:
            st.audio(audio_recording)
            st.info(
                "Audio is captured for this session. Automatic speech "
                "transcription is not configured yet. You can type the "
                "transcribed answer in the Candidate answer box."
            )
    else:
        st.caption(
            "Audio recording is not supported by this Streamlit version. "
            "Use typed answers instead."
        )

with addon_tabs[3]:
    st.subheader("Resume improvement coach")
    st.write(
        "Finds job-description terms missing from the resume and suggests clearer wording. "
        "Suggestions must not add achievements, skills, or experience that the candidate has not confirmed."
    )
    if not candidate_results:
        st.info("Analyze resumes first to use the resume coach.")
    else:
        coach_name = st.selectbox(
            "Choose candidate resume", [c["name"] for c in candidate_results], key="coach_candidate"
        )
        coach_candidate = next(c for c in candidate_results if c["name"] == coach_name)
        coach_text = coach_candidate.get("resume_text", "")
        required = coach_candidate.get("required_skills", [])
        matched = coach_candidate.get("matched_skills", [])
        missing = [skill for skill in required if skill not in matched]

        st.write("**Potential keyword gaps**")
        if missing:
            st.write(", ".join(skill.title() for skill in missing))
            st.caption("Only add a keyword if it truthfully describes the candidate's experience.")
        else:
            st.success("No gaps found among the skills recognized by the current keyword list.")

        st.write("**Bullet-point improvement template**")
        st.code(
            "Action verb + task/project + tool or method + measurable result\n"
            "Example template: Developed [what] using [tool/method], resulting in [verified outcome].",
            language="text"
        )
        if st.button("Create safe rewrite suggestions", key="rewrite_suggestions"):
            lines = [line.strip(" •\t-") for line in coach_text.splitlines() if len(line.strip()) > 25]
            lines = lines[:5]
            if lines:
                for line in lines:
                    st.write(f"• Consider rewriting this verified detail more clearly: “{line[:220]}”")
                    st.caption("Template: Action + task + method/tool + result (add only verified facts).")
            else:
                st.warning("No clear bullet points were extracted. Review the PDF text manually.")
        st.text_area(
            "Candidate-edited bullet point (optional)",
            height=100,
            key="candidate_bullet_edit",
            placeholder="Write a truthful bullet point using your actual experience."
        )

with addon_tabs[4]:
    st.subheader("Resume integrity review flags")
    st.write(
        "These checks highlight items for verification; they do not determine whether a resume is fraudulent "
        "or AI-generated."
    )
    if not candidate_results:
        st.info("Analyze resumes first to run integrity checks.")
    else:
        integrity_name = st.selectbox(
            "Choose resume", [c["name"] for c in candidate_results], key="integrity_candidate"
        )
        integrity_candidate = next(c for c in candidate_results if c["name"] == integrity_name)
        integrity_text = integrity_candidate.get("resume_text", "")
        words = re.findall(r"\b[a-zA-Z0-9+#.-]+\b", integrity_text.lower())
        normalized_lines = [
            re.sub(r"\s+", " ", line.strip().lower())
            for line in integrity_text.splitlines() if len(line.strip()) > 35
        ]
        duplicate_lines = sorted({line for line in normalized_lines if normalized_lines.count(line) > 1})
        date_patterns = re.findall(
            r"\b(?:19|20)\d{2}\b", integrity_text
        )
        flags = []
        if duplicate_lines:
            flags.append(f"{len(duplicate_lines)} repeated long line(s) detected in the extracted text.")
        if len(words) < 80:
            flags.append("Very little text was extracted; this may be a scanned/image-based or short resume.")
        if len(date_patterns) >= 2 and len(date_patterns) != len(set(date_patterns)):
            flags.append("Some years repeat. Repeated years can be normal; verify chronology only if relevant.")
        if flags:
            for flag in flags:
                st.warning(flag)
        else:
            st.success("No basic text-integrity flags were found by these limited checks.")
        st.caption(
            "Do not label a candidate dishonest based on these flags. Verify qualifications, employment, "
            "and work samples through consistent, consented procedures. AI-writing detectors are unreliable."
        )

with addon_tabs[5]:
    st.subheader("Candidate engagement message drafts")
    st.write("Prepare a personalized message. This tool drafts text only; it does not send emails or messages.")
    if candidate_results:
        message_name = st.selectbox(
            "Candidate", [c["name"] for c in candidate_results], key="message_candidate"
        )
        message_status = st.selectbox(
            "Status", ["Shortlisted", "Further review", "Not shortlisted", "Interview invitation"],
            key="message_status"
        )
        recruiter_name = st.text_input("Recruiter/team name", value="Recruitment Team", key="recruiter_name")
        role_name = st.text_input("Role title", value="the advertised role", key="role_title")
        if st.button("Generate message draft", key="generate_candidate_message"):
            safe_candidate_name = message_name.rsplit(".", 1)[0].replace("_", " ").replace("-", " ")
            message_templates = {
                "Shortlisted": (
                    f"Hello {safe_candidate_name},\n\nThank you for applying for {role_name}. "
                    "Your application has been shortlisted for the next stage. Our team will share "
                    "the next steps after a recruiter review.\n\nRegards,\n{team}"
                ),
                "Further review": (
                    f"Hello {safe_candidate_name},\n\nThank you for applying for {role_name}. "
                    "Your application is under review. We will update you when the review is complete.\n\n"
                    "Regards,\n{team}"
                ),
                "Not shortlisted": (
                    f"Hello {safe_candidate_name},\n\nThank you for your interest in {role_name} and for "
                    "taking the time to apply. We will not be progressing your application at this stage. "
                    "We appreciate your interest and wish you well.\n\nRegards,\n{team}"
                ),
                "Interview invitation": (
                    f"Hello {safe_candidate_name},\n\nWe would like to invite you to an initial interview "
                    f"for {role_name}. Please reply to the recruiter to coordinate a suitable time.\n\n"
                    "Regards,\n{team}"
                ),
            }
            draft = message_templates[message_status].replace("{team}", recruiter_name)
            st.text_area("Editable message draft", draft, height=200, key="candidate_message_draft")
            st.caption("Review names, role details, and status before sending.")

with addon_tabs[6]:
    st.subheader("Job-related communication assessment")
    st.write(
        "Use a written, job-relevant sample to review clarity and structure. This does not infer personality, "
        "emotion, honesty, or culture fit from voice, accent, face, or micro-expressions."
    )
    sample_response = st.text_area(
        "Paste a candidate's job-related written response",
        height=150,
        key="soft_skills_response",
        placeholder="Paste a response to a role-related question..."
    )
    if st.button("Review communication structure", key="review_communication"):
        if not sample_response.strip():
            st.warning("Paste a response first.")
        else:
            response_words = re.findall(r"\b\w+\b", sample_response)
            sentences = [s for s in re.split(r"[.!?]+", sample_response) if s.strip()]
            word_count = len(response_words)
            sentence_count = len(sentences)
            average_sentence = word_count / max(sentence_count, 1)
            clarity_indicator = 100
            if word_count < 25:
                clarity_indicator -= 25
            if sentence_count == 0:
                clarity_indicator -= 25
            if average_sentence > 35:
                clarity_indicator -= 15
            if not any(word in sample_response.lower() for word in
                       ["because", "for example", "therefore", "result", "first", "then"]):
                clarity_indicator -= 10
            st.metric("Writing structure indicator", f"{max(0, clarity_indicator)}/100")
            st.write(f"Word count: {word_count} · Sentence count: {sentence_count}")
            st.write("**Human-review prompts**")
            st.write("- Is the answer relevant to the question?")
            st.write("- Are claims specific and supported with examples?")
            st.write("- Is the explanation understandable for the role?")
            st.caption(
                "This is a simple writing-structure heuristic, not a validated soft-skills or personality test. "
                "Provide reasonable accommodations and assess candidates consistently."
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
BERT Semantic Similarity
&nbsp; • &nbsp;
Streamlit

</div>
""", unsafe_allow_html=True)
# =========================================================
# 3D FLOATING VISUALS
# =========================================================

three_d_html = """<div class="three-d-layer"><div class="robot-3d"><div class="robot-head"><div class="robot-face"></div></div><div class="robot-ear-left"></div><div class="robot-ear-right"></div><div class="robot-body"></div><div class="robot-glow"></div></div><div class="resume-3d"><div class="resume-person"></div><div class="resume-line one"></div><div class="resume-line two"></div><div class="resume-line three"></div><div class="resume-magnifier"></div></div><div class="pdf-3d"><div class="pdf-lines"></div><div class="pdf-label">PDF</div></div><div class="target-3d"><div class="target-arrow"></div></div><div class="brain-3d"><div class="brain-platform"></div></div><div class="cube-3d cube-one"></div><div class="cube-3d cube-two"></div><div class="cube-3d cube-three"></div><div class="cube-3d cube-four"></div><div class="neon-line neon-line-one"></div><div class="neon-line neon-line-two"></div><div class="neon-line neon-line-three"></div></div>"""

st.markdown(three_d_html, unsafe_allow_html=True)
