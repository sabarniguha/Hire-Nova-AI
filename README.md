# 🧠 InterviewVerse AI Pro

A premium, single-file AI-powered mock interview platform built with Streamlit. Practice
technical, HR, behavioral, and coding interviews against 12 AI interviewer personas and
18 company-specific interview styles — then get a full multi-agent evaluation report with
a downloadable PDF.

## ✨ Features

- **Live AI Interviews** — Groq conducts a real, streamed, conversational interview (not canned questions).
- **18 Company Personas** — Google, Amazon, Microsoft, Meta, Netflix, Apple, IBM, Cisco, Adobe,
  Oracle, Atlassian, Flipkart, Walmart, Goldman Sachs, JP Morgan, Morgan Stanley, Uber, Airbnb —
  each with a distinct interviewer style, focus area, and hiring bar baked into the prompts.
- **12 Interviewer Personas** — HR Manager, Senior Software Engineer, Engineering Manager, Tech
  Lead, Professor, AI Researcher, Startup Founder, Security Engineer, Cloud Architect, Product
  Manager, Recruiter, Behavioral Coach.
- **20 Interview Tracks** — HR, Behavioral, Technical, Coding, System Design, AI/ML, Data Science,
  Python, Java, C++, JavaScript, SQL, DBMS, OS, Computer Networks, Cloud, Cybersecurity, DevOps,
  React, Node.js.
- **8 Interview Modes** — Standard, Stress, Friendly Mentor, Rapid Fire, Mock Assessment, and
  Beginner/Intermediate/Advanced/Expert difficulty.
- **Coding Interviews** — In-app code editor, AI-generated questions, and AI code review
  (correctness, complexity, optimizations, edge cases).
- **Multi-Agent Evaluation** — Mistral powers 7 specialized evaluator agents (Technical,
  Communication, Behavior, Confidence, Problem Solving, Leadership, Grammar) aggregated into a
  final hiring recommendation (Strong Hire / Hire / Hold / Reject).
- **Resume Analyzer** — Upload a PDF, get an ATS score, detected skills, missing-skill
  suggestions, and improvement tips.
- **Analytics Dashboard** — Radar charts, score trends, skill heatmaps, weekly/monthly/yearly
  views, all in Plotly with a consistent dark theme.
- **Gamification** — XP, levels, streaks, and achievement badges.
- **Report Export** — PDF (ReportLab), Markdown, and HTML.
- **Auth** — Signup/login/forgot-password with bcrypt password hashing and JWT sessions.

## 🧩 AI Model Architecture

| Model | Role | Required? |
|---|---|---|
| **Groq** | Conducts interviews, generates questions & follow-ups, streams responses, powers coding-question review | Yes, for live mode |
| **Mistral** | Evaluates answers, computes scores, writes reports & learning roadmaps | Yes, for live mode |
| **Gemini** | Optional enhancement slot, surfaced in Settings/AI selector | **No** — never required |

**The app runs fully in demo mode with zero API keys.** Every screen — including interviews and
evaluation reports — has a graceful, clearly-labeled offline fallback so nothing ever breaks.

## 🚀 Getting Started

```bash
pip install -r requirements.txt
cp .env.example .env   # optional — add your Groq/Mistral keys, or skip and use demo mode
streamlit run app.py
```

The SQLite database (`database.db`) and its tables are created automatically on first run —
no manual SQL required.

You can also add your API keys later from inside the app: **Settings → AI Provider Keys**.

## 🗂️ Project Structure

```
app.py              # The entire application (UI, backend, AI, DB — single file, per spec)
requirements.txt
README.md
.env.example
database.db          # auto-created SQLite DB
uploads/              # uploaded resumes
reports/              # generated report artifacts
assets/                # static assets
```

## 🛠️ Tech Stack

Python · Streamlit · SQLite · Groq API · Mistral API · Plotly · Pandas · NumPy ·
python-dotenv · PyPDF2 · pdfplumber · ReportLab · bcrypt · PyJWT · streamlit-option-menu ·
Pillow · (optional: SpeechRecognition, pyttsx3, OpenCV)

## ⚠️ Notes on Optional Dependencies

Every optional package (Gemini SDK, streamlit-option-menu, bcrypt, PyJWT, ReportLab,
pdfplumber/PyPDF2, SpeechRecognition, pyttsx3, OpenCV) is imported defensively. If a package
isn't installed, the related feature degrades gracefully (e.g. a plain radio nav instead of
the animated option-menu, a SHA-256 fallback instead of bcrypt) instead of crashing the app.
Check **Settings → System Status** at any time to see what's active.

## 📄 License

Built as a portfolio-grade demonstration project.
