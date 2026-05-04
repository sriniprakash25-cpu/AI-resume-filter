📘 AI Resume Screening System (Deep Learning)
📌 Project Overview

This project is an intelligent AI-based Resume Screening System that automatically analyzes and ranks resumes based on their relevance to a given job description. It uses deep learning and Natural Language Processing (NLP) to understand the semantic meaning of text instead of relying only on keyword matching.

The system helps recruiters quickly identify the best candidates by evaluating skills, experience, and education.

🎯 Features
📄 Supports multiple file formats (PDF, DOCX, TXT)
🧠 Uses deep learning model for semantic understanding
🔍 Skill matching (keyword + semantic)
⏳ Experience detection (supports ranges & decimals)
🎓 Education matching using contextual similarity
⚖️ Weighted scoring system
🏆 Automatic ranking of candidates
🖥️ User-friendly GUI using Tkinter
🧠 Technologies Used
Python
Natural Language Processing
Deep Learning
Sentence-BERT
Cosine Similarity
Tkinter
⚙️ How It Works
Enter Job Description
Upload multiple resumes
System extracts and preprocesses text
Converts text into embeddings using Sentence-BERT
Computes similarity between job description and resumes
Evaluates:
Skills
Experience
Education
Calculates final score using weighted formula
Displays ranked results in GUI
🧮 Scoring Formula
Final Score = (Skills × 0.45) + (Experience × 0.35) + (Education × 0.20)
📂 Project Structure
├── main.py              # Main application (GUI + logic)
├── requirements.txt    # Required libraries
├── resumes/            # Sample resumes
└── README.md           # Project documentation
▶️ Installation & Setup
1. Clone the Repository
git clone https://github.com/your-username/ai-resume-screening.git
cd ai-resume-screening
2. Install Dependencies
pip install -r requirements.txt
3. Run the Application
python main.py
📦 Required Libraries
numpy
pdfplumber
python-docx
scikit-learn
sentence-transformers
tkinter
📊 Output
Ranked list of candidates
Overall score
Skills, experience, education breakdown
Matched & missing skills
🚀 Future Enhancements
Web-based interface (Flask / Django)
Database integration
Resume parsing using advanced models
Multi-language support
Real-time job portal integration
