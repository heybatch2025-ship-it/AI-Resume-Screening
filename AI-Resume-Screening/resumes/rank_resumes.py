import os
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Job description
job_description = """
We are looking for a Computer Science candidate with skills in
Python, SQL, Machine Learning, HTML, CSS, JavaScript and Git.
The candidate should have knowledge of software development
and artificial intelligence.
"""

resume_folder = "resumes"

results = []

# Read every PDF in the resumes folder
for filename in os.listdir(resume_folder):

    if filename.lower().endswith(".pdf"):

        pdf_path = os.path.join(resume_folder, filename)

        reader = PdfReader(pdf_path)

        resume_text = ""

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                resume_text += page_text + " "

        # TF-IDF
        documents = [job_description, resume_text]

        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform(documents)

        similarity = cosine_similarity(matrix[0:1], matrix[1:2])

        score = similarity[0][0] * 100

        results.append((filename, score))


# Sort from highest to lowest
results.sort(key=lambda x: x[1], reverse=True)

print("\n===== AI RESUME RANKING =====")

for rank, (filename, score) in enumerate(results, start=1):

    print(
        f"{rank}. {filename} - "
        f"{round(score, 2)}%"
    )