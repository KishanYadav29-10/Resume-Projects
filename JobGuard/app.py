import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model
model = joblib.load('model.pkl')

st.set_page_config(page_title="🕵️ Fake Job Posting Detection", layout="centered")

st.title("🕵️ Fake Job Posting Detection")
st.markdown("Enter job details to detect if it's **fraudulent** or **legitimate**.")

# Sidebar for threshold adjustment
st.sidebar.header("⚙️ Settings")
threshold = st.sidebar.slider("Detection Threshold", 0.0, 1.0, 0.45, 0.01)

# Input fields
job_desc = st.text_area("Job Description", placeholder="Enter detailed job description...")
skills = st.text_input("Skills (optional)")
employment_type = st.selectbox("Employment Type", ["Full-time", "Part-time", "Contract", "Temporary", "Other"])
industry = st.text_input("Industry (optional)", "unknown")
location = st.text_input("Location (e.g., New York, USA)", "unknown")
salary_range = st.text_input("Salary Range (e.g., 40000-60000)", "unknown")
department = st.text_input("Department (optional)", "unknown")
benefits = st.text_input("Benefits (optional)", "unknown")

# Helper functions for feature engineering (should match those used in training)
import re
import string

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

def extract_avg_salary(salary):
    try:
        numbers = [float(x.replace(',', '')) for x in re.findall(r'\d+(?:\.\d+)?', salary)]
        if len(numbers) == 2:
            return (numbers[0] + numbers[1]) / 2
        elif len(numbers) == 1:
            return numbers[0]
        else:
            return np.nan
    except:
        return np.nan

def create_features(job_desc, skills, employment_type, industry, location, salary_range, department, benefits):
    job_text = clean_text(job_desc + ' ' + skills)
    has_link = int(bool(re.search(r"http|www", job_text)))
    has_whatsapp = int(bool(re.search(r"whatsapp", job_text)))
    has_fee = int(bool(re.search(r"fee|charge|deposit|registration", job_text)))
    avg_salary = extract_avg_salary(salary_range)
    
    data = {
        'job_text': [job_text],
        'employment_type': [employment_type],
        'industry': [industry],
        'location': [location],
        'salary_range': [salary_range],
        'department': [department],
        'benefits': [benefits],
        'avg_salary': [avg_salary],
        'has_link': [has_link],
        'has_whatsapp': [has_whatsapp],
        'has_fee': [has_fee]
    }
    
    return pd.DataFrame(data)

if st.button("🔍 Detect"):
    input_df = create_features(job_desc, skills, employment_type, industry, location, salary_range, department, benefits)
    pred_proba = model.predict_proba(input_df)[0]
    fake_proba = pred_proba[1] * 100

    if pred_proba[1] >= threshold:
        result = "❌ Fake"
    else:
        result = "✅ Legitimate"

    st.markdown(f"## Prediction: {result}")
    st.write(f"**Confidence Score (Fake):** {fake_proba:.2f}%")
    st.write(f"⚙️ **Detection Threshold:** {threshold}")

    with st.expander("🔎 Show Prediction Probabilities"):
        st.json({
            "Legitimate (0)": round(pred_proba[0], 4),
            "Fake (1)": round(pred_proba[1], 4)
        })

st.markdown("---")
st.caption("Built with ❤️ kishan using Streamlit and Machine Learning")
