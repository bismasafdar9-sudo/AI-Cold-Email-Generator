from chains.email_generator import generate_email

job_description = """
Need Python Developer

Skills:
- Python
- Machine Learning
- Flask
- SQL
"""

email = generate_email(job_description)

print("\nGenerated Email:\n")
print(email)