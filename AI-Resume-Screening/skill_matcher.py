from pypdf import PdfReader

# Resume PDF path
pdf_path = "resumes/sample_resume.pdf"

# Read PDF
reader = PdfReader(pdf_path)

text = ""

for page in reader.pages:
    page_text = page.extract_text()
    if page_text:
        text += page_text + "\n"

# Convert text to lowercase
text = text.lower()

# Skills database
skills = [
    "python",
    "sql",
    "machine learning",
    "html",
    "css",
    "javascript",
    "java",
    "c++",
    "git",
    "data science"
]

# Find skills in resume
found_skills = []

for skill in skills:
    if skill in text:
        found_skills.append(skill)

print("----- RESUME SKILL ANALYSIS -----")
print("Skills found in resume:")
print(found_skills)
# Required skills for the job
job_skills = [
    "python",
    "sql",
    "machine learning",
    "html",
    "javascript"
]

# Find matching skills
matched_skills = []

for skill in job_skills:
    if skill in found_skills:
        matched_skills.append(skill)

# Calculate match percentage
match_percentage = (len(matched_skills) / len(job_skills)) * 100

print("\n----- JOB MATCHING -----")
print("Required Skills:", job_skills)
print("Matched Skills:", matched_skills)
print("Match Score:", round(match_percentage, 2), "%")
if match_percentage >= 70:
    print("Recommendation: SELECTED")
else:
    print("Recommendation: NOT SELECTED")