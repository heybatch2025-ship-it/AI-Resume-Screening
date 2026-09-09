job_skills = input("Enter required job skills: ").lower().split(",")

resume_skills = input("Enter candidate skills: ").lower().split(",")

# Remove extra spaces
job_skills = [skill.strip() for skill in job_skills]
resume_skills = [skill.strip() for skill in resume_skills]

matched_skills = []

for skill in job_skills:
    if skill in resume_skills:
        matched_skills.append(skill)

match_percentage = (len(matched_skills) / len(job_skills)) * 100

print("\nJob Skills:", job_skills)
print("Resume Skills:", resume_skills)
print("Matched Skills:", matched_skills)
print("Match Score:", round(match_percentage, 2), "%")