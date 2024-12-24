import streamlit as st

import fitz  # PyMuPDF

import re
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

from PIL import Image

import plotly.graph_objects as go


# Sample data for training
data = {
    "Resume Text": [
        "Experienced in Python and data analysis. Looking for a Data Scientist role. Contact: john.doe@example.com, Phone: 123-456-7890",
        "Skilled in Java and software development. Seeking a Java Developer position. Email: jane.smith@example.com, Phone: +1-987-654-3210",
        "Proficient in HTML, CSS, and JavaScript. Available for Web Developer jobs. You can reach me at contact@webdev.com.",
    ],
    "Job Role": ["Data Scientist", "Java Developer", "Web Developer"],
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Train a simple classifier
X_train, X_test, y_train, y_test = train_test_split(df['Resume Text'], df['Job Role'], test_size=0.2, random_state=42)
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X_train, y_train)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Function to extract details
def extract_details(text):
    name = re.search(r'\b([A-Z][a-z]+(?: [A-Z][a-z]+)*|[A-Z\s]+)', text)
    skills = re.findall(r'\b(?:Python|Java|JavaScript|C\+\+|HTML|CSS|Machine Learning|Data Science|Data Analyst|Software Engineer|Web Developer|Deep Learning)\b', text)
    return name.group(0) if name else "Name not found", list(set(skills))

# Function to extract contact information
def extract_contact_info(text):
    email = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    phone = re.findall(r'\+?\d[\d -]{8,}\d', text)
    return email[0] if email else "Email not found", phone[0] if phone else "Phone not found"

# Skill improvement suggestions dictionary
skill_improvement_links = {
    "Python": ("Advanced Python Concepts", "https://realpython.com/"),
    "Java": ("Java Programming Basics", "https://www.codecademy.com/learn/learn-java"),
    "JavaScript": ("JavaScript Fundamentals", "https://developer.mozilla.org/en-US/docs/Learn/JavaScript"),
    "C++": ("C++ Programming Guide", "https://www.learncpp.com/"),
    "HTML": ("HTML & CSS Basics", "https://www.freecodecamp.org/learn/responsive-web-design/basic-html-and-html5/"),
    "CSS": ("CSS Flexbox and Grid", "https://css-tricks.com/snippets/css/complete-guide-grid/"),
    "Machine Learning": ("Introduction to Machine Learning", "https://www.coursera.org/learn/machine-learning"),
    "Data Science": ("Data Science Specialization", "https://www.coursera.org/specializations/jhu-data-science"),
    "Data Analyst": ("Data Analysis with Python", "https://www.freecodecamp.org/learn/data-analysis-with-python/"),
    "Software Engineer": ("Software Engineering Principles", "https://www.udacity.com/course/software-development-process--nd9990"),
    "Web Developer": ("Full-Stack Web Development", "https://www.freecodecamp.org/learn/front-end-development-libraries/"),
    "Deep Learning": ("Deep Learning Specialization", "https://www.coursera.org/specializations/deep-learning"),
}

# Dictionary to hold skill logos
skill_logos = {
    "Python": "images/python_logo.png",  # Update with your image paths
    "Java": "images/java_logo.png",
    "JavaScript": "images/javascript_logo.png",
    "C++": "images/cplusplus_logo.png",
    "HTML": "images/html_logo.png",
    "CSS": "images/css_logo.png",
    "Machine Learning": "images/machine_learning_logo.png",
    "Data Science": "images/data_science_logo.png",
    "Data Analyst": "images/data_analyst_logo.png",
    "Software Engineer": "images/software_engineer_logo.png",
    "Web Developer": "images/web_developer_logo.png",
    "Deep Learning": "images/deep_learning_logo.png",
}

# Keywords for ATS score calculation
ats_keywords = {
    "Data Scientist": ["Python", "Data Analysis", "Machine Learning", "Statistics"],
    "Java Developer": ["Java", "Spring", "Hibernate", "Software Development"],
    "Web Developer": ["HTML", "CSS", "JavaScript", "React", "Web Development"],
}

# Function to calculate ATS score
def calculate_ats_score(skills, predicted_role):
    if predicted_role in ats_keywords:
        required_keywords = ats_keywords[predicted_role]
        matched_keywords = [keyword for keyword in required_keywords if keyword in skills]
        score = (len(matched_keywords) / len(required_keywords)) * 100  # Score as a percentage
        return round(score, 2), matched_keywords
    return 0, []

# Function to create a gauge chart for ATS score
def display_ats_gauge(ats_score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=ats_score,
        title={"text": "ATS Score"},
        gauge={
            "axis": {"range": [0, 100]},  # Set the range from 0 to 100
            "bar": {"color": "lightblue"},
            "steps": [
                {"range": [0, 50], "color": "red"},
                {"range": [50, 75], "color": "yellow"},
                {"range": [75, 100], "color": "green"},
            ],
        },
    ))

    # Update layout for better visibility
    fig.update_layout(
        height=250,
        width=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    st.plotly_chart(fig)

# Streamlit interface
st.set_page_config(page_title="Resume Parser", layout="wide")  # Set page layout

# Welcome message
st.title("Welcome to the Resume Parser App")
st.write("This application helps you parse resumes and provides insights into job roles, ATS scores, and skill improvement suggestions.")
st.write("Please upload your resume in PDF format to get started.")

# Sidebar for file upload
st.sidebar.title("Upload Your Resume")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Extract text from PDF
    text = extract_text_from_pdf(uploaded_file)

    # Display the extracted text immediately after the welcome note
    st.subheader("Extracted Text:")
    st.write(text)

    # Extract details
    name, skills = extract_details(text)
    
    # Extract contact information
    email, phone = extract_contact_info(text)
    
    # Display extracted details
    st.subheader("Extracted Details:")
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"*** ***")
        st.write(f"**Name:** {name}")
        st.write(f"*** ***")
        st.write(f"**Email:** {email}")
        st.write(f"*** ***")
        st.write(f"**Phone:** {phone}")
        st.write(f"*** ***")


    # Predict job role using the trained model
    predicted_role = model.predict([text])[0]
    st.subheader("Predicted Job Role:")
    st.write(f"**{predicted_role}**")
    st.write(f"*** ***")

    # Calculate ATS score
    ats_score, matched_keywords = calculate_ats_score(skills, predicted_role)
    # Display ATS score as a subheader
    st.subheader("ATS Score:")
    # Display ATS gauge chart
    display_ats_gauge(ats_score)
    st.write(f"*** ***")

    # Provide skills section with logos and names side by side
    if skills:
        st.subheader("Skills:")
        for skill in skills:
            if skill in skill_logos:
                # Create columns for logo and skill name
                col1, col2 = st.columns([1, 5])  # Adjust column width ratios as needed
                with col1:
                    st.image(skill_logos[skill], width=50)  # Display the logo
                with col2:
                    st.write(f"**{skill}**")  # Display the skill name
    st.write(f"*** ***")
    
    # Separate suggestions for skill improvement
    if skills:
        st.subheader("Suggestions for Skill Improvement:")
        for skill in skills:
            if skill in skill_improvement_links:
                suggestion, link = skill_improvement_links[skill]
                st.write(f"- **{suggestion}**: [Learn more]({link})")

    # Show matched keywords for ATS
    if matched_keywords:
        st.write("Matched Keywords for ATS Score:")
        st.write(", ".join(matched_keywords))

# Footer
st.sidebar.markdown("---")
st.sidebar.write("© 2024 Resume Parser App")
