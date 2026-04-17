import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Take input
folder = input("Enter resume folder path: ")
job_file = input("Enter job description file path: ")

resumes = []
names = []

# Read resumes
if os.path.exists(folder):
    for file in os.listdir(folder):
        if file.endswith(".txt"):
            f = open(folder + "/" + file, "r", encoding="utf-8")
            text = f.read()
            f.close()

            if len(text.split()) >= 50:   # minimum 50 words
                text = text.lower()
                resumes.append(text)
                names.append(file)
            else:
                print(file, "skipped (less than 50 words)")
else:
    print("Folder not found")

# Read job description
if os.path.exists(job_file):
    f = open(job_file, "r", encoding="utf-8")
    job_desc = f.read().lower()
    f.close()
else:
    print("Job file not found")
    job_desc = ""

# Combine all text
all_text = resumes + [job_desc]

# Convert to numbers (TF-IDF)
vectorizer = TfidfVectorizer()
matrix = vectorizer.fit_transform(all_text)

# Compare resumes with job description
scores = cosine_similarity(matrix[:-1], matrix[-1])

# Store results
result = []
for i in range(len(names)):
    percent = scores[i][0] * 100
    result.append((names[i], percent))

# Sort results
result.sort(key=lambda x: x[1], reverse=True)

# Print top 5
print("\nTop Candidates:\n")
for i in range(min(5, len(result))):
    print(i+1, ".", result[i][0], "->", round(result[i][1], 2), "%")