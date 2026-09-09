from pypdf import PdfReader

pdf_path = "resumes/sample_resume.pdf"

reader = PdfReader(pdf_path)

text = ""

for page in reader.pages:
    page_text = page.extract_text()
    if page_text:
        text += page_text + "\n"

print("----- RESUME TEXT -----")
print(text)