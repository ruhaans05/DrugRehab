import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import json


print("For all questions answer with yes, no, or sometimes")
print("For definition of drug, view: https://www.fda.gov/drugs/drug-approvals-and-databases/drugsfda-glossary-terms")

avg = 0.0

q1 = input("Do you use drugs every day?")
while q1 not in ('yes', 'no', 'sometimes'):
    print('Please answer with yes, no, or sometimes. Verify your spelling.')
    q1 = input("Do you use drugs every day?")
    
q2 = input("Do you use drugs by yourself?")
while q2 not in ('yes', 'no', 'sometimes'):
    print('Please answer with yes, no, or sometimes. Verify your spelling.')
    q2 = input("Do you use drugs by yourself?")
    
q3 = input("Were you peer pressured into trying drugs?")
while q3 not in ('yes', 'no', 'sometimes'):
    print('Please answer with yes, no, or sometimes. Verify your spelling.')
    q3 = input("Were you peer pressured into trying drugs?")
    
q4 = input("Do you spend a lot of money on drugs?")
while q4 not in ('yes', 'no', 'sometimes'):
    print('Please answer with yes, no, or sometimes. Verify your spelling.')
    q4 = input("Do you spend a lot of money on drugs?")

q5 = input("Do you fake sobriety?")
while q5 not in ('yes', 'no', 'sometimes'):
    print('Please answer with yes, no, or sometimes. Verify your spelling.')
    q5 = input("Do you fake sobriety?")

q6 = input("Have you been caught using drugs by a guardian, authority figure, or someone you don't like?")
while q6 not in ('yes', 'no', 'sometimes'):
    print('Please answer with yes, no, or sometimes. Verify your spelling.')
    q6 = input("Have you been caught using drugs by a guardian, authority figure, or someone you don't like?")

q7 = input("Do drugs negatively affect your social life?")
while q7 not in ('yes', 'no', 'sometimes'):
    print('Please answer with yes, no, or sometimes. Verify your spelling.')
    q7 = input("Do drugs negatively affect your social life?")

q8 = input("Do drugs negatively affect your work/education?")
while q8 not in ('yes', 'no', 'sometimes'):
    print('Please answer with yes, no, or sometimes. Verify your spelling.')
    q8 = input("Do drugs negatively affect your work/education?")

q9 = input("Do you feel an attachment to drugs?")
while q9 not in ('yes', 'no', 'sometimes'):
    print('Please answer with yes, no, or sometimes. Verify your spelling.')
    q9 = input("Do you feel an attachment to drugs?")

q10 = input("Do you prefer being intoxicated than being sober in social settings?")
while q10 not in ('yes', 'no', 'sometimes'):
    print('Please answer with yes, no, or sometimes. Verify your spelling.')
    q10 = input("Do you prefer being intoxicated than being sober in social settings?")


answers = [q1,q2,q3,q4,q5,q6,q7,q8,q9,q10] #stores all the answers to the questions asked


for i in range(len(answers)):
    if answers[i].lower() == 'yes':
        avg += 1.0
    elif answers[i].lower() == 'sometimes':
        avg += 0.5
    
avg /= len(questions)



encoded = [1 if a=='yes' else 0.5 if a=='sometimes' else 0 for a in answers]

RESPONSES_FILE = 'responses.csv'
THRESHOLDS_FILE = 'thresholds.json'
ALPHA = 0.05 #threshold adjustment

if os.path.exists(RESPONSES_FILE):
    df = pd.read_csv(RESPONSES_FILE)
else:
    df = pd.DataFrame(columns=[f"q{i+1}" for i in range(len(questions))])

# Step 4: Append new response and save
new_row = pd.DataFrame([encoded], columns=df.columns)
df = pd.concat([df, new_row], ignore_index=True)
df.to_csv(RESPONSES_FILE, index=False)

# Step 5: Run KMeans if enough samples
if len(df) >= 3:
    kmeans = KMeans(n_clusters=3, random_state=42).fit(df)
    centroid_means = kmeans.cluster_centers_.mean(axis=1)
    order = sorted(range(3), key=lambda i: centroid_means[i])
    mapped_means = {
        "light": centroid_means[order[0]],
        "moderate": centroid_means[order[1]],
        "heavy": centroid_means[order[2]]
    }

    # Step 6: Load or initialize thresholds
    if os.path.exists(THRESHOLDS_FILE):
        with open(THRESHOLDS_FILE, "r") as f:
            thresholds = json.load(f)
    else:
        thresholds = {"light": 0.25, "moderate": 0.5, "heavy": 0.675}

    # Step 7: Slowly update thresholds
    for key in thresholds:
        thresholds[key] = (1 - ALPHA) * thresholds[key] + ALPHA * mapped_means[key]

    with open(THRESHOLDS_FILE, "w") as f:
        json.dump(thresholds, f)

    # Step 8: Classify based on closest evolving threshold
    dist = {key: abs(avg - thresholds[key]) for key in thresholds}
    classification = min(dist, key=dist.get)
    print("\nYour drug use intensity is classified as:", classification.capitalize(), "use")

else:
    # Fallback if too few samples
    if avg < thresholds['light']:
        print("\nYour drug use intensity is classified as: Light use")
        category = "light"
    elif avg < thresholds['moderate']:
        print("\nYour drug use intensity is classified as: Moderate use")
        category = "moderate"
    else:
        print("\nYour drug use intensity is classified as: Heavy use")
        category = "heavy"

print('Detailed stats:\n\n')
print('Score from survey is', avg, '\nOriginal DrugRehab model defines anything around 0.3 to 0.675 to be moderate usage.\n>0.675 as heavy\n<0.3 as light')
