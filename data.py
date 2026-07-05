"""
Generates mock resumes and job descriptions for the recruitment system.
Run this once before using the app.
"""

import os
import json

# Create directories
os.makedirs("data/resumes", exist_ok=True)
os.makedirs("data/jds", exist_ok=True)

# 15 Mock Resumes
RESUMES = {
    "john_doe.txt": """
JOHN DOE
Email: john.doe@email.com | Phone: +1-234-567-8900

SUMMARY
Senior AI Engineer with 5 years of experience in building production ML systems using Python, TensorFlow, and LangChain.

EXPERIENCE
Senior AI Engineer | TechCorp AI | 2021 - Present
- Led development of 3 LLM-based chatbots using LangChain and Gemini API
- Optimized vector database queries, reducing latency by 40%
- Built RAG pipelines with ChromaDB and LangGraph

ML Engineer | DataFlow Inc | 2018 - 2021
- Developed ML models for NLP tasks using Python and TensorFlow
- Deployed models on AWS Lambda with 99.9% uptime

EDUCATION
B.S. in Computer Science | MIT | 2018

SKILLS
Python, TensorFlow, PyTorch, LangChain, LangGraph, ChromaDB, FastAPI, AWS, GCP, RAG, NLP
""",
    "jane_smith.txt": """
JANE SMITH
Email: jane.smith@email.com | Phone: +1-345-678-9012

SUMMARY
Full-stack AI/ML engineer with 4 years in building end-to-end ML systems and APIs.

EXPERIENCE
AI/ML Engineer | StartupXYZ | 2022 - Present
- Built LLM integrations using LangChain and OpenAI APIs
- Implemented vector search with Chroma for semantic retrieval
- Developed FastAPI microservices for ML inference

Junior ML Engineer | BigData Co | 2020 - 2022
- Created data pipelines with Python and Pandas
- Trained and evaluated classification models

EDUCATION
M.S. in Data Science | Stanford | 2020
B.S. in Mathematics | UC Berkeley | 2018

SKILLS
Python, LangChain, OpenAI API, ChromaDB, FastAPI, Docker, PostgreSQL, Pandas, Scikit-learn
""",
    "karen_white.txt": """
KAREN WHITE
Email: karen.white@email.com | Phone: +1-456-789-0123

SUMMARY
Principal ML Engineer with 7 years at FAANG companies. Expert in production systems and team leadership.

EXPERIENCE
Principal ML Engineer | GoogleAI | 2019 - Present
- Led ML platform team of 6 engineers
- Implemented large-scale RAG systems with 1M+ documents
- Mentored junior engineers on LLM best practices

Senior ML Engineer | Facebook AI | 2016 - 2019
- Built recommendation systems serving 1B+ users
- Optimized TensorFlow models for mobile deployment

EDUCATION
Ph.D. in Machine Learning | CMU | 2016
B.S. in Computer Science | Carnegie Mellon University | 2012

SKILLS
Python, PyTorch, TensorFlow, LLMs, RAG, ChromaDB, Kubernetes, GCP, Leadership, System Design
""",
    "michael_johnson.txt": """
MICHAEL JOHNSON
Email: michael.johnson@email.com | Phone: +1-567-890-1234

SUMMARY
Backend engineer transitioning to AI with 6 years of software engineering experience.

EXPERIENCE
Senior Backend Engineer | CloudSys | 2018 - Present
- Built microservices architecture handling 10K req/sec
- Implemented caching and optimization strategies
- Recently started exploring LangChain and LLM integrations

Backend Engineer | WebScale Inc | 2017 - 2018
- Developed REST APIs in Python and Node.js
- Managed PostgreSQL and Redis databases

EDUCATION
B.S. in Computer Science | University of Washington | 2017

SKILLS
Python, Node.js, FastAPI, PostgreSQL, Redis, Docker, Kubernetes, System Design, (learning) LangChain
""",
    "sarah_lee.txt": """
SARAH LEE
Email: sarah.lee@email.com | Phone: +1-678-901-2345

SUMMARY
Data Scientist with 3 years in machine learning and statistical analysis.

EXPERIENCE
Data Scientist | AnalyticsPro | 2021 - Present
- Built predictive models for customer churn (RandomForest, XGBoost)
- Created data visualizations and dashboards with Tableau
- Basic Python and SQL, exploring LangChain

Junior Data Scientist | DataInsight Co | 2021
- Analyzed A/B test results and reported findings
- Built regression models for trend analysis

EDUCATION
M.S. in Statistics | University of Michigan | 2021
B.S. in Mathematics | University of Illinois | 2019

SKILLS
Python, Pandas, Scikit-learn, SQL, Tableau, Statistics, (beginner) LangChain
""",
    "alex_kumar.txt": """
ALEX KUMAR
Email: alex.kumar@email.com | Phone: +1-789-012-3456

SUMMARY
DevOps & MLOps engineer with 5 years in deployment and infrastructure.

EXPERIENCE
MLOps Engineer | AI Ops Inc | 2022 - Present
- Deployed ML models using Docker, Kubernetes, and CI/CD pipelines
- Integrated LangChain models into production systems
- Monitored and optimized model serving infrastructure

DevOps Engineer | CloudScale Corp | 2019 - 2022
- Managed AWS infrastructure and automation
- Implemented monitoring and logging solutions

EDUCATION
B.S. in Computer Science | IIT Delhi | 2019

SKILLS
Docker, Kubernetes, AWS, CI/CD, Python, Bash, Terraform, (learning) MLOps, LangChain basics
""",
    "david_brown.txt": """
DAVID BROWN
Email: david.brown@email.com | Phone: +1-890-123-4567

SUMMARY
Research scientist transitioning to industry with 2 years of research and 1 year of applied ML.

EXPERIENCE
ML Researcher | ResearchLab | 2024 - Present
- Published 2 papers on LLM fine-tuning and RAG optimization
- Implemented state-of-the-art NLP models

Research Intern | University Lab | 2022 - 2024
- Conducted NLP research using TensorFlow and PyTorch
- Contributed to open-source NLP projects

EDUCATION
B.S. in Computer Science | UC Berkeley | 2022

SKILLS
Python, PyTorch, TensorFlow, Research, NLP, (learning) LangChain, LLMs, Limited production experience
""",
    "emma_wilson.txt": """
EMMA WILSON
Email: emma.wilson@email.com | Phone: +1-901-234-5678

SUMMARY
Full-stack engineer with 8 years of experience in web and recently AI systems.

EXPERIENCE
AI Full-Stack Engineer | InnovateTech | 2021 - Present
- Built end-to-end AI applications with React frontend and Python backend
- Integrated Gemini API with LangChain for chatbot applications
- Deployed to AWS and optimized for scale

Senior Full-Stack Engineer | WebDev Co | 2016 - 2021
- Developed web applications with Node.js and React
- Managed AWS infrastructure and databases

EDUCATION
B.S. in Computer Science | University of Texas | 2016

SKILLS
Python, JavaScript, React, Node.js, FastAPI, AWS, Docker, LangChain, Gemini API, Full-stack development
""",
    "robert_martinez.txt": """
ROBERT MARTINEZ
Email: robert.martinez@email.com | Phone: +1-012-345-6789

SUMMARY
Senior AI Engineer with 6 years in production AI systems and team leadership.

EXPERIENCE
Head of AI | TechVentures | 2021 - Present
- Led AI team of 4 engineers building LLM applications
- Implemented RAG systems with LangChain and ChromaDB
- Mentored engineers on prompt engineering and LLM best practices

Senior AI Engineer | AIFirst Corp | 2018 - 2021
- Built recommendation systems and NLP pipelines
- Optimized model serving with FastAPI

EDUCATION
M.S. in Computer Science | Carnegie Mellon University | 2018
B.S. in Computer Science | UC San Diego | 2016

SKILLS
Python, LangChain, LangGraph, ChromaDB, FastAPI, Leadership, NLP, RAG, Prompt Engineering
""",
    "lisa_anderson.txt": """
LISA ANDERSON
Email: lisa.anderson@email.com | Phone: +1-123-456-7890

SUMMARY
ML Engineer with 4 years of experience in computer vision and recent pivot to LLM systems.

EXPERIENCE
ML Engineer | VisionAI Inc | 2021 - Present
- Worked on computer vision models initially
- Recently transitioned to LLM/RAG work with LangChain
- Built document understanding systems with ChromaDB

Computer Vision Engineer | ImageTech | 2020 - 2021
- Developed CNN models for image classification
- Deployed models using TensorFlow Lite

EDUCATION
M.S. in Computer Science | MIT | 2020
B.S. in Computer Science | Georgia Tech | 2018

SKILLS
Python, PyTorch, TensorFlow, Computer Vision, (recent) LangChain, ChromaDB, RAG, FastAPI
""",
    "thomas_henry.txt": """
THOMAS HENRY
Email: thomas.henry@email.com | Phone: +1-234-567-8901

SUMMARY
Principal Software Architect with 12 years of experience. Limited ML background but strong system design.

EXPERIENCE
Principal Architect | EnterpriseSoft | 2018 - Present
- Designed large-scale distributed systems
- Recently exploring LLM integrations for enterprise applications
- Strong in architecture and system design

Senior Engineer | BigCorp | 2015 - 2018
- Led engineering team building core systems
- Optimized database performance and system design

EDUCATION
B.S. in Computer Science | Stanford | 2012

SKILLS
System Design, Java, Python, Microservices, Database Design, Leadership, (limited) ML, (learning) LangChain
""",
    "rachel_green.txt": """
RACHEL GREEN
Email: rachel.green@email.com | Phone: +1-345-678-9012

SUMMARY
ML Engineer with 3 years in building conversational AI systems.

EXPERIENCE
ML Engineer | ChatbotCo | 2022 - Present
- Built multi-turn conversational systems with LangChain
- Integrated Gemini and Claude LLMs into production
- Optimized conversation flows for better UX

Junior ML Engineer | AIStart | 2021 - 2022
- Built simple chatbots using rule-based systems
- Transitioned to LLM-based approaches

EDUCATION
M.S. in AI | University of Southern California | 2021
B.S. in Computer Science | UCLA | 2019

SKILLS
Python, LangChain, Conversational AI, Prompt Engineering, ChromaDB, FastAPI, Gemini API, Claude API
""",
    "christopher_lee.txt": """
CHRISTOPHER LEE
Email: christopher.lee@email.com | Phone: +1-456-789-0123

SUMMARY
Backend engineer with 5 years experience, exploring AI and ML integration.

EXPERIENCE
Senior Backend Engineer | ScaleApp | 2020 - Present
- Built scalable backend systems in Python and Java
- Recently integrated LLM APIs for new features
- Implemented caching and optimization techniques

Backend Engineer | AppDev Co | 2019 - 2020
- Developed REST APIs and microservices
- Worked with PostgreSQL and Redis

EDUCATION
B.S. in Computer Science | University of California | 2019

SKILLS
Python, Java, FastAPI, PostgreSQL, Redis, Microservices, Docker, (beginner) LLM integration, (learning) LangChain
""",
    "victoria_brown.txt": """
VICTORIA BROWN
Email: victoria.brown@email.com | Phone: +1-567-890-1234

SUMMARY
Data Engineer with 4 years in building data pipelines. New to ML/AI space.

EXPERIENCE
Senior Data Engineer | DataFlow Corp | 2022 - Present
- Built ETL pipelines processing 10GB+ daily data
- Recently exploring data pipelines for ML systems
- Learning about embeddings and vector databases

Data Engineer | DataWorks | 2020 - 2022
- Created data infrastructure for analytics
- Optimized database queries and performance

EDUCATION
B.S. in Computer Science | University of Washington | 2020

SKILLS
Python, SQL, Spark, Airflow, PostgreSQL, ETL, (beginner) ML concepts, (learning) Vector databases, ChromaDB
""",
    "nicholas_taylor.txt": """
NICHOLAS TAYLOR
Email: nicholas.taylor@email.com | Phone: +1-678-901-2345

SUMMARY
Experienced ML Engineer with 5 years in computer vision and recent expansion into LLM/RAG systems.

EXPERIENCE
ML Engineer | VisionPlus | 2021 - Present
- Developed computer vision models with PyTorch
- Recently built RAG systems with LangChain and ChromaDB
- Implemented retrieval pipelines for document understanding

ML Engineer | ImageAI | 2020 - 2021
- Built CNN models for various vision tasks
- Deployed models with TensorFlow Serving

EDUCATION
M.S. in Machine Learning | Carnegie Mellon University | 2020
B.S. in Computer Science | Cornell University | 2018

SKILLS
Python, PyTorch, TensorFlow, Computer Vision, LangChain, ChromaDB, RAG, FastAPI, System Design
"""
}

# Default JD
JD_TEMPLATE = """
Role: Senior AI Engineer

Required Skills:
- Python
- LangChain
- ChromaDB
- FastAPI
- LangGraph
- Machine Learning fundamentals
- RAG systems

Responsibilities:
- Design and build production LLM applications
- Optimize vector database queries and retrieval
- Mentor junior engineers on best practices
- Build end-to-end AI systems
- Deploy and monitor AI models in production

Required Experience: 4 years
"""

# Create resume files
print("Creating mock resumes...")
for filename, content in RESUMES.items():
    filepath = os.path.join("data/resumes", filename)
    with open(filepath, "w") as f:
        f.write(content)
    print(f"  ✓ {filename}")

# Create JD file
print("\nCreating default job description...")
jd_path = "data/jds/ai_engineer_jd.txt"
with open(jd_path, "w") as f:
    f.write(JD_TEMPLATE)
print(f"  ✓ {jd_path}")

# Create directories for outputs
os.makedirs("data/parsed_resumes", exist_ok=True)
os.makedirs("data/chroma_db", exist_ok=True)

print("\n✅ Data generation complete!")
print(f"Created {len(RESUMES)} mock resumes in data/resumes/")
print("Ready to run: python pregenerate_cache.py && python app.py")
