from pydantic import BaseModel, Field
from typing import List, Dict, Optional
class CandidateProfile(BaseModel):
    name: str = Field(description="Full name of the candidate")
    email: str = Field(description="Email address of the candidate")
    phone: str = Field(default="N/A", description="Phone number of the candidate")
    education: str = Field(description="Highest degree and university, e.g., M.S. in Computer Science, Stanford")
    experience_years: int = Field(description="Total years of work experience as an integer")
    skills: List[str] = Field(description="List of technical skills extracted from the resume, normalized (e.g. lowercase or standard terms)")
    key_projects: List[str] = Field(description="Summary of key projects mentioned in the resume")
class JobDescription(BaseModel):
    role_name: str = Field(description="Clean role title, e.g., Senior AI Engineer")
    required_skills: List[str] = Field(description="List of core technical skills required, normalized and lowercase")
    required_experience_years: int = Field(description="Minimum years of experience required for the role")
    responsibilities: List[str] = Field(description="List of main responsibilities or duties of the role")
class EmailDraft(BaseModel):
    recipient_email: str = Field(description="Candidate email address")
    subject: str = Field(description="Email subject line")
    body: str = Field(description="Professional email body text")
class InterviewQuestions(BaseModel):
    candidate_name: str = Field(description="Name of the candidate")
    questions: List[str] = Field(description="List of exactly 5 specific technical and behavioral questions grounded in their resume and JD gaps")
class QueryIntent(BaseModel):
    intent: str = Field(description="Classified intent: 'load', 'count', 'screen', 'rewrite', 'interview', 'salary', 'compare', 'schedule', 'trend', 'batch_report', 'confirm', 'chat'")
    target_candidate: Optional[str] = Field(default=None, description="Extracted candidate name if relevant (e.g. for interview questions or scheduling)")
    target_role: Optional[str] = Field(default=None, description="Extracted role/skills if relevant (e.g. for salary or JD rewrite)")
    additional_context: Optional[str] = Field(default=None, description="Any other specific instructions, e.g. 'for a startup' or tone instructions")
