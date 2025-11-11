import pandas as pd

# Create a sample DataFrame with requirements
data = {
    'requirement_id': [1, 2, 3],
    'requirement_text': [
        "The system shall provide user authentication via username and password.",
        "The system shall encrypt all sensitive data at rest using AES-256.",
        "The system shall respond to user requests within 500ms under normal load."
    ]
}

# Create DataFrame and save to Excel
df = pd.DataFrame(data)
df.to_excel('./output/sample_requirements.xlsx', index=False)
print("Sample requirements file created: sample_requirements.xlsx")
