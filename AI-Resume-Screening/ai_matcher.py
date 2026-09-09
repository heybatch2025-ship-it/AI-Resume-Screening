from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Resume PDF
pdf_path = "resumes/sample_resume.pdf"

# Read resume
reader = PdfReader(pdf_path)

resume_text = ""

for page in reader.pages:
    page_text = page.extract_text()
    if page_text:
        resume_text += page_text + " "

# Job description
job_description = """
We are looking for a Computer Science candidate with skills in
Python, SQL, Machine Learning, HTML, CSS, JavaScript and Git.
The candidate should have knowledge of software development
and artificial intelligence.
"""

# Convert text into TF-IDF vectors
documents = [job_description, resume_text]

vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(documents)

# Calculate similarity
similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

score = similarity[0][0] * 100

print("----- AI RESUME MATCHING -----")
print("AI Match Score:", round(score, 2), "%")

if score >= 60:
    print("AI Recommendation: SELECTED")
else:
    print("AI Recommendation: NOT SELECTED")