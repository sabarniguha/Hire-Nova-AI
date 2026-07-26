"""
HireNova AI
======================
The Ultimate Enterprise AI Interview & Career Intelligence Platform.
A premium, single-file AI-powered mock interview platform.

Run with:
    streamlit run app.py

Author: Built for Sabarni
"""

# ============================================================================
# SECTION 0: IMPORTS
# ============================================================================
from __future__ import annotations

import base64
import contextlib
import hashlib
import io
import json
import os
import random
import re
import sqlite3
import time
import uuid
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ---- Optional dependency shims -------------------------------------------
# Every optional dependency is imported defensively. If it's missing, the
# related feature degrades gracefully instead of crashing the app.

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import bcrypt
    HAS_BCRYPT = True
except ImportError:
    HAS_BCRYPT = False

try:
    import jwt as pyjwt
    HAS_JWT = True
except ImportError:
    HAS_JWT = False

try:
    from groq import Groq
    HAS_GROQ_SDK = True
except ImportError:
    HAS_GROQ_SDK = False

try:
    from mistralai import Mistral
    HAS_MISTRAL_SDK = True
except ImportError:
    HAS_MISTRAL_SDK = False

try:
    import google.generativeai as genai
    HAS_GEMINI_SDK = True
except ImportError:
    HAS_GEMINI_SDK = False

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

try:
    from PyPDF2 import PdfReader
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

try:
    from streamlit_option_menu import option_menu
    HAS_OPTION_MENU = True
except ImportError:
    HAS_OPTION_MENU = False

try:
    from PIL import Image
    HAS_PILLOW = True
except ImportError:
    HAS_PILLOW = False

try:
    import speech_recognition as sr
    HAS_SPEECH_RECOGNITION = True
except ImportError:
    HAS_SPEECH_RECOGNITION = False

try:
    import pyttsx3
    HAS_PYTTSX3 = True
except ImportError:
    HAS_PYTTSX3 = False

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

# ============================================================================
# SECTION 1: GLOBAL CONFIG / CONSTANTS
# ============================================================================

APP_NAME = "HireNova AI"
APP_TAGLINE = "The Ultimate Enterprise AI Interview & Career Intelligence Platform"
APP_EDITION = "Enterprise Edition"
ALT_TAGLINES = [
    "Master Interviews. Build Skills. Get Hired.",
    "Your AI-Powered Career Success Platform.",
    "Prepare Smarter. Interview Better. Get Hired.",
    "Enterprise AI Interview Intelligence Platform.",
    "Practice Like It's the Real Interview.",
]
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "database.db")
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

for _dir in (UPLOAD_DIR, REPORTS_DIR, ASSETS_DIR):
    os.makedirs(_dir, exist_ok=True)

JWT_SECRET = os.getenv("JWT_SECRET", "interviewverse-dev-secret-change-me")
JWT_ALGO = "HS256"
SESSION_HOURS = 12

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-large-latest")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("interviewverse")

XP_PER_INTERVIEW = 50
XP_PER_HIGH_SCORE = 25  # bonus if score >= 80
LEVEL_XP_STEP = 200

DIFFICULTY_LEVELS = ["Beginner", "Intermediate", "Advanced", "Expert"]
INTERVIEW_MODES = [
    "Standard", "Stress Interview", "Friendly Mentor", "Rapid Fire", "Mock Assessment",
]
INTERVIEW_TRACKS = [
    "HR", "Behavioral", "Technical", "Coding", "System Design", "AI/ML",
    "Data Science", "Python", "Java", "C++", "JavaScript", "SQL", "DBMS",
    "Operating Systems", "Computer Networks", "Cloud Computing",
    "Cybersecurity", "DevOps", "React", "Node.js",
]
CODE_LANGUAGES = ["Python", "Java", "C++", "JavaScript", "SQL"]

# ============================================================================
# SECTION 2: DOMAIN DATA — COMPANIES, PERSONAS, AGENTS
# ============================================================================
# These data structures drive prompt construction so that every company /
# persona genuinely produces different interviewer behavior, rather than
# being cosmetic labels.

@dataclass(frozen=True)
class CompanyProfile:
    name: str
    focus: str            # what this company's interviews emphasize
    style: str             # tone / pacing of the interviewer
    bar: str                # what "good" looks like at this company
    signature_topics: list

COMPANIES: dict[str, CompanyProfile] = {
    "Google": CompanyProfile(
        "Google", "algorithmic depth, scalability, and clean code",
        "analytical, probing, asks 'why' repeatedly, expects you to reason from first principles",
        "elegant, optimal solutions with clear complexity analysis",
        ["data structures", "distributed systems", "Googleyness", "ambiguity handling"],
    ),
    "Amazon": CompanyProfile(
        "Amazon", "leadership principles paired with practical system design",
        "direct, story-driven, repeatedly asks for concrete past examples (STAR format)",
        "ownership, bias for action, and customer obsession backed by specific metrics",
        ["Leadership Principles", "scalable architecture", "operational excellence"],
    ),
    "Microsoft": CompanyProfile(
        "Microsoft", "collaborative problem solving and practical engineering trade-offs",
        "warm but rigorous, encourages thinking out loud, values growth mindset",
        "clear communication and pragmatic, maintainable solutions",
        ["growth mindset", "cross-team collaboration", "Azure/cloud fundamentals"],
    ),
    "Meta": CompanyProfile(
        "Meta", "speed of execution, impact, and product sense alongside coding",
        "fast-paced, terse, pushes hard on optimizing under tight time pressure",
        "moving fast without breaking things, measurable impact",
        ["product sense", "A/B testing intuition", "high-throughput systems"],
    ),
    "Netflix": CompanyProfile(
        "Netflix", "high autonomy, judgment, and freedom-and-responsibility culture fit",
        "conversational but blunt, tests candor and independent decision-making",
        "senior-level judgment with minimal oversight",
        ["freedom & responsibility", "microservices", "chaos engineering"],
    ),
    "Apple": CompanyProfile(
        "Apple", "attention to detail, craftsmanship, and product polish",
        "measured, precise, dislikes hand-waving, wants exact specifics",
        "obsessive quality and elegant user-facing detail",
        ["design sensibility", "performance optimization", "secrecy/confidentiality"],
    ),
    "IBM": CompanyProfile(
        "IBM", "enterprise-scale reliability and structured problem solving",
        "formal, methodical, walks through requirements step by step",
        "robust, well-documented enterprise-grade solutions",
        ["enterprise architecture", "hybrid cloud", "consulting mindset"],
    ),
    "Cisco": CompanyProfile(
        "Cisco", "networking fundamentals and infrastructure reliability",
        "technical, protocol-focused, drills into networking internals",
        "deep understanding of networking and infrastructure trade-offs",
        ["networking protocols", "security", "infrastructure at scale"],
    ),
    "Adobe": CompanyProfile(
        "Adobe", "creative-technical balance and product craftsmanship",
        "encouraging, curious about your creative process and technical rigor",
        "solutions that are both technically sound and user-delightful",
        ["creative tooling", "performance in rich UIs", "cross-platform design"],
    ),
    "Oracle": CompanyProfile(
        "Oracle", "database internals and enterprise software depth",
        "rigorous, exam-like, tests depth over breadth",
        "deep mastery of a narrow, well-defined technical domain",
        ["database internals", "SQL optimization", "enterprise licensing systems"],
    ),
    "Atlassian": CompanyProfile(
        "Atlassian", "teamwork values and pragmatic collaborative engineering",
        "friendly, values-driven, references 'open company, no bullshit'",
        "collaborative, transparent problem solving",
        ["teamwork values", "agile practices", "developer tooling"],
    ),
    "Flipkart": CompanyProfile(
        "Flipkart", "e-commerce scale problems and Indian market context",
        "fast, competitive, expects strong DSA fundamentals",
        "efficient solutions under India-scale traffic assumptions",
        ["e-commerce systems", "high concurrency", "DSA fundamentals"],
    ),
    "Walmart": CompanyProfile(
        "Walmart", "retail-scale logistics and cost-efficient engineering",
        "practical, business-outcome oriented, ties tech to cost savings",
        "reliable systems that operate efficiently at massive retail scale",
        ["supply chain systems", "cost optimization", "omni-channel retail"],
    ),
    "Goldman Sachs": CompanyProfile(
        "Goldman Sachs", "quantitative rigor and financial-systems correctness",
        "formal, precise, expects mathematical and risk-aware reasoning",
        "provably correct, low-latency, risk-conscious solutions",
        ["quantitative reasoning", "low-latency systems", "risk management"],
    ),
    "JP Morgan": CompanyProfile(
        "JP Morgan", "financial domain knowledge and regulatory-grade reliability",
        "structured, compliance-aware, methodical",
        "secure, auditable, highly reliable financial systems",
        ["fintech systems", "compliance", "transaction integrity"],
    ),
    "Morgan Stanley": CompanyProfile(
        "Morgan Stanley", "analytical depth and resilient trading-adjacent systems",
        "sharp, fast follow-ups, tests composure under pressure",
        "resilient, well-reasoned solutions under time pressure",
        ["capital markets systems", "resilience", "analytical thinking"],
    ),
    "Uber": CompanyProfile(
        "Uber", "real-time, geo-distributed systems and marketplace dynamics",
        "energetic, scenario-based, loves 'what if this fails' follow-ups",
        "systems that stay correct under real-time, high-failure conditions",
        ["real-time systems", "marketplace design", "geo-distributed infra"],
    ),
    "Airbnb": CompanyProfile(
        "Airbnb", "trust, belonging, and community-oriented product design",
        "warm, story-driven, tests empathy and design thinking",
        "trustworthy, human-centered product and technical solutions",
        ["trust & safety", "community design", "marketplace economics"],
    ),
}

PERSONAS = {
    "HR Manager": "a warm but evaluative HR Manager assessing culture fit, motivation, and communication",
    "Senior Software Engineer": "a technically sharp Senior Software Engineer who digs into implementation details",
    "Engineering Manager": "an Engineering Manager balancing technical depth with team/leadership signals",
    "Tech Lead": "a Tech Lead who cares about architecture decisions and trade-off reasoning",
    "Professor": "an academic Professor who tests fundamentals and conceptual clarity rigorously",
    "AI Researcher": "an AI Researcher probing deep understanding of ML theory and reasoning",
    "Startup Founder": "a fast-moving Startup Founder who values scrappiness, ownership, and speed",
    "Security Engineer": "a Security Engineer who stress-tests answers for vulnerabilities and edge cases",
    "Cloud Architect": "a Cloud Architect focused on scalability, reliability, and cost trade-offs",
    "Product Manager": "a Product Manager evaluating product thinking and user-centered reasoning",
    "Recruiter": "a friendly Recruiter screening for communication clarity and role fit",
    "Behavioral Coach": "a Behavioral Coach focused on STAR-method storytelling and self-awareness",
}

MODE_INSTRUCTIONS = {
    "Standard": "Conduct a balanced, professional interview at a normal, measured pace.",
    "Stress Interview": "Apply deliberate pressure: interrupt gently, ask rapid follow-ups, challenge assumptions, and remain slightly skeptical to test composure.",
    "Friendly Mentor": "Be warm, encouraging, and supportive. Give gentle hints if the candidate struggles, like a mentor would.",
    "Rapid Fire": "Ask short, quick-fire questions. Keep each question to one or two sentences and move fast between topics.",
    "Mock Assessment": "Behave like a formal graded assessment: minimal chit-chat, clear structured questions, strictly evaluative tone.",
}

# Multi-agent evaluators used by the Mistral-powered evaluation pipeline.
EVALUATION_AGENTS = {
    "Technical Evaluator": "Assess technical correctness, depth of knowledge, and problem-solving ability. Score 0-100.",
    "Communication Evaluator": "Assess clarity, structure, and articulateness of the candidate's answers. Score 0-100.",
    "Behavior Evaluator": "Assess behavioral competency: teamwork, adaptability, conflict handling. Score 0-100.",
    "Confidence Evaluator": "Assess confidence and composure conveyed through the answers (tone, hedging, certainty). Score 0-100.",
    "Problem Solving Evaluator": "Assess structured problem-solving approach and logical reasoning. Score 0-100.",
    "Leadership Evaluator": "Assess ownership, initiative, and leadership signals in the answers. Score 0-100.",
    "Grammar Evaluator": "Assess grammar, vocabulary, and professional language quality. Score 0-100.",
}

# ============================================================================
# SECTION 3: DATABASE LAYER (SQLite, auto-created, no manual SQL needed)
# ============================================================================

@contextlib.contextmanager
def get_conn():
    """Context-managed SQLite connection with row factory set to dict-like rows."""
    conn = sqlite3.connect(DB_PATH, timeout=10, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Create all tables if they do not already exist. Safe to call every run."""
    with get_conn() as conn:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                college TEXT DEFAULT '',
                github TEXT DEFAULT '',
                linkedin TEXT DEFAULT '',
                photo_b64 TEXT DEFAULT '',
                xp INTEGER DEFAULT 0,
                level INTEGER DEFAULT 1,
                streak INTEGER DEFAULT 0,
                last_active_date TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                groq_key TEXT DEFAULT '',
                mistral_key TEXT DEFAULT '',
                gemini_key TEXT DEFAULT '',
                theme TEXT DEFAULT 'dark',
                font_size TEXT DEFAULT 'Medium',
                language TEXT DEFAULT 'English',
                notifications INTEGER DEFAULT 1
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS interviews (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                track TEXT,
                company TEXT,
                persona TEXT,
                mode TEXT,
                difficulty TEXT,
                language TEXT,
                transcript_json TEXT,
                started_at TEXT,
                ended_at TEXT,
                duration_seconds INTEGER DEFAULT 0,
                status TEXT DEFAULT 'in_progress',
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS reports (
                id TEXT PRIMARY KEY,
                interview_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                overall_score REAL,
                technical_score REAL,
                communication_score REAL,
                confidence_score REAL,
                behavior_score REAL,
                leadership_score REAL,
                problem_solving_score REAL,
                grammar_score REAL,
                hiring_recommendation TEXT,
                strengths_json TEXT,
                weaknesses_json TEXT,
                roadmap_json TEXT,
                raw_json TEXT,
                created_at TEXT,
                FOREIGN KEY (interview_id) REFERENCES interviews(id),
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS resume_analyses (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                filename TEXT,
                ats_score REAL,
                skills_json TEXT,
                missing_skills_json TEXT,
                suggestions_json TEXT,
                raw_text TEXT,
                created_at TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS achievements (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                title TEXT,
                description TEXT,
                earned_at TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        conn.commit()


# ---- User CRUD -------------------------------------------------------------

def db_create_user(name: str, email: str, password: str) -> Optional[str]:
    uid = str(uuid.uuid4())
    pwd_hash = hash_password(password)
    try:
        with get_conn() as conn:
            conn.execute(
                "INSERT INTO users (id, name, email, password_hash, created_at) VALUES (?,?,?,?,?)",
                (uid, name, email.lower().strip(), pwd_hash, datetime.utcnow().isoformat()),
            )
        return uid
    except sqlite3.IntegrityError:
        return None


def db_get_user_by_email(email: str) -> Optional[sqlite3.Row]:
    with get_conn() as conn:
        cur = conn.execute("SELECT * FROM users WHERE email = ?", (email.lower().strip(),))
        return cur.fetchone()


def db_get_user_by_id(user_id: str) -> Optional[sqlite3.Row]:
    with get_conn() as conn:
        cur = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        return cur.fetchone()


def db_update_user(user_id: str, **fields) -> None:
    if not fields:
        return
    keys = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values()) + [user_id]
    with get_conn() as conn:
        conn.execute(f"UPDATE users SET {keys} WHERE id = ?", values)


def db_add_xp(user_id: str, amount: int) -> None:
    user = db_get_user_by_id(user_id)
    if not user:
        return
    new_xp = user["xp"] + amount
    new_level = 1 + new_xp // LEVEL_XP_STEP
    today = datetime.utcnow().date().isoformat()
    last_active = user["last_active_date"]
    streak = user["streak"]
    if last_active:
        last_date = datetime.fromisoformat(last_active).date()
        delta = (datetime.utcnow().date() - last_date).days
        if delta == 1:
            streak += 1
        elif delta > 1:
            streak = 1
        # delta == 0 -> same day, streak unchanged
    else:
        streak = 1
    db_update_user(user_id, xp=new_xp, level=new_level, last_active_date=today, streak=streak)


# ---- Interview CRUD ---------------------------------------------------------

def db_create_interview(user_id: str, track: str, company: str, persona: str,
                          mode: str, difficulty: str, language: str) -> str:
    iid = str(uuid.uuid4())
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO interviews
               (id, user_id, track, company, persona, mode, difficulty, language,
                transcript_json, started_at, status)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (iid, user_id, track, company, persona, mode, difficulty, language,
             json.dumps([]), datetime.utcnow().isoformat(), "in_progress"),
        )
    return iid


def db_update_interview_transcript(interview_id: str, transcript: list) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE interviews SET transcript_json = ? WHERE id = ?",
            (json.dumps(transcript), interview_id),
        )


def db_end_interview(interview_id: str, duration_seconds: int) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE interviews SET ended_at = ?, duration_seconds = ?, status = 'completed' WHERE id = ?",
            (datetime.utcnow().isoformat(), duration_seconds, interview_id),
        )


def db_get_interview(interview_id: str) -> Optional[sqlite3.Row]:
    with get_conn() as conn:
        cur = conn.execute("SELECT * FROM interviews WHERE id = ?", (interview_id,))
        return cur.fetchone()


def db_get_user_interviews(user_id: str, limit: int = 200) -> list[sqlite3.Row]:
    with get_conn() as conn:
        cur = conn.execute(
            "SELECT * FROM interviews WHERE user_id = ? ORDER BY started_at DESC LIMIT ?",
            (user_id, limit),
        )
        return cur.fetchall()


# ---- Report CRUD --------------------------------------------------------------

def db_save_report(interview_id: str, user_id: str, evaluation: dict) -> str:
    rid = str(uuid.uuid4())
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO reports
               (id, interview_id, user_id, overall_score, technical_score, communication_score,
                confidence_score, behavior_score, leadership_score, problem_solving_score,
                grammar_score, hiring_recommendation, strengths_json, weaknesses_json,
                roadmap_json, raw_json, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                rid, interview_id, user_id,
                evaluation.get("overall_score", 0),
                evaluation.get("technical_score", 0),
                evaluation.get("communication_score", 0),
                evaluation.get("confidence_score", 0),
                evaluation.get("behavior_score", 0),
                evaluation.get("leadership_score", 0),
                evaluation.get("problem_solving_score", 0),
                evaluation.get("grammar_score", 0),
                evaluation.get("hiring_recommendation", "Hold"),
                json.dumps(evaluation.get("strengths", [])),
                json.dumps(evaluation.get("weaknesses", [])),
                json.dumps(evaluation.get("roadmap", {})),
                json.dumps(evaluation),
                datetime.utcnow().isoformat(),
            ),
        )
    return rid


def db_get_report_for_interview(interview_id: str) -> Optional[sqlite3.Row]:
    with get_conn() as conn:
        cur = conn.execute(
            "SELECT * FROM reports WHERE interview_id = ? ORDER BY created_at DESC LIMIT 1",
            (interview_id,),
        )
        return cur.fetchone()


def db_get_user_reports(user_id: str) -> list[sqlite3.Row]:
    with get_conn() as conn:
        cur = conn.execute(
            "SELECT * FROM reports WHERE user_id = ? ORDER BY created_at DESC", (user_id,)
        )
        return cur.fetchall()


# ---- Resume analysis CRUD -----------------------------------------------------

def db_save_resume_analysis(user_id: str, filename: str, analysis: dict, raw_text: str) -> str:
    aid = str(uuid.uuid4())
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO resume_analyses
               (id, user_id, filename, ats_score, skills_json, missing_skills_json,
                suggestions_json, raw_text, created_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                aid, user_id, filename, analysis.get("ats_score", 0),
                json.dumps(analysis.get("skills", [])),
                json.dumps(analysis.get("missing_skills", [])),
                json.dumps(analysis.get("suggestions", [])),
                raw_text[:20000],
                datetime.utcnow().isoformat(),
            ),
        )
    return aid


def db_get_user_resume_analyses(user_id: str) -> list[sqlite3.Row]:
    with get_conn() as conn:
        cur = conn.execute(
            "SELECT * FROM resume_analyses WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        )
        return cur.fetchall()


# ---- Achievements ---------------------------------------------------------------

def db_grant_achievement(user_id: str, title: str, description: str) -> None:
    with get_conn() as conn:
        exists = conn.execute(
            "SELECT 1 FROM achievements WHERE user_id = ? AND title = ?", (user_id, title)
        ).fetchone()
        if exists:
            return
        conn.execute(
            "INSERT INTO achievements (id, user_id, title, description, earned_at) VALUES (?,?,?,?,?)",
            (str(uuid.uuid4()), user_id, title, description, datetime.utcnow().isoformat()),
        )


def db_get_user_achievements(user_id: str) -> list[sqlite3.Row]:
    with get_conn() as conn:
        cur = conn.execute(
            "SELECT * FROM achievements WHERE user_id = ? ORDER BY earned_at DESC", (user_id,)
        )
        return cur.fetchall()

# ============================================================================
# SECTION 4: AUTH & SECURITY
# ============================================================================

def hash_password(password: str) -> str:
    """Hash a password with bcrypt if available, else fall back to a salted
    SHA-256 scheme (clearly weaker — used only if bcrypt isn't installed)."""
    if HAS_BCRYPT:
        return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    salt = "iv-fallback-salt"
    return "sha256$" + hashlib.sha256((salt + password).encode("utf-8")).hexdigest()


def verify_password(password: str, password_hash: str) -> bool:
    if password_hash.startswith("sha256$"):
        salt = "iv-fallback-salt"
        return password_hash == "sha256$" + hashlib.sha256((salt + password).encode("utf-8")).hexdigest()
    if HAS_BCRYPT:
        try:
            return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
        except ValueError:
            return False
    return False


def create_session_token(user_id: str) -> str:
    payload = {
        "user_id": user_id,
        "exp": datetime.utcnow() + timedelta(hours=SESSION_HOURS),
        "iat": datetime.utcnow(),
    }
    if HAS_JWT:
        return pyjwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)
    # Fallback: opaque token stored only in-memory via session_state (still works
    # for a single browser session, just not verifiable cross-process).
    return f"local${user_id}${uuid.uuid4()}"


def decode_session_token(token: str) -> Optional[str]:
    if not token:
        return None
    if token.startswith("local$"):
        parts = token.split("$")
        return parts[1] if len(parts) > 1 else None
    if HAS_JWT:
        try:
            payload = pyjwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
            return payload.get("user_id")
        except Exception:
            return None
    return None


def is_valid_email(email: str) -> bool:
    return re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email or "") is not None


def is_strong_password(password: str) -> tuple[bool, str]:
    if len(password or "") < 8:
        return False, "Password must be at least 8 characters."
    if not re.search(r"[A-Z]", password):
        return False, "Include at least one uppercase letter."
    if not re.search(r"[0-9]", password):
        return False, "Include at least one number."
    return True, ""

# ============================================================================
# SECTION 5: AI CLIENT LAYER (Groq / Mistral / Gemini) with retry & fallback
# ============================================================================

def _get_api_key(provider: str) -> str:
    """Resolve an API key: session-stored user key takes priority over env var."""
    user = st.session_state.get("user")
    key_field = f"{provider}_key"
    if user and user.get(key_field):
        return user[key_field]
    env_map = {"groq": "GROQ_API_KEY", "mistral": "MISTRAL_API_KEY", "gemini": "GEMINI_API_KEY"}
    return os.getenv(env_map[provider], "")


def groq_available() -> bool:
    return HAS_GROQ_SDK and bool(_get_api_key("groq"))


def mistral_available() -> bool:
    return HAS_MISTRAL_SDK and bool(_get_api_key("mistral"))


def gemini_available() -> bool:
    return HAS_GEMINI_SDK and bool(_get_api_key("gemini"))


def with_retry(fn, *args, retries: int = 2, backoff: float = 1.5, **kwargs):
    """Call fn with simple exponential backoff retry. Raises the last error."""
    last_err = None
    for attempt in range(retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 — intentional broad catch for API robustness
            last_err = e
            logger.warning("Call failed (attempt %d/%d): %s", attempt + 1, retries + 1, e)
            if attempt < retries:
                time.sleep(backoff ** attempt)
    raise last_err


def groq_chat_stream(messages: list[dict], temperature: float = 0.7, max_tokens: int = 1024):
    """Stream a chat completion from Groq. Yields text chunks.
    Falls back to a single offline notice chunk if Groq is unavailable."""
    if not groq_available():
        yield ("⚠️ Groq is not configured. Add your GROQ_API_KEY in Settings to enable "
               "live AI interviews. Showing a placeholder question so you can preview the flow.")
        return
    try:
        client = Groq(api_key=_get_api_key("groq"))

        def _call():
            return client.chat.completions.create(
                model=GROQ_MODEL, messages=messages, temperature=temperature,
                max_tokens=max_tokens, stream=True,
            )
        stream = with_retry(_call, retries=1)
        for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta
    except Exception as e:
        logger.error("Groq streaming failed: %s", e)
        yield f"\n\n⚠️ *AI temporarily unavailable ({type(e).__name__}). Please try again.*"


def groq_chat(messages: list[dict], temperature: float = 0.5, max_tokens: int = 1200) -> str:
    """Non-streaming Groq call — used for structured/quick generations."""
    if not groq_available():
        return ""
    try:
        client = Groq(api_key=_get_api_key("groq"))

        def _call():
            return client.chat.completions.create(
                model=GROQ_MODEL, messages=messages, temperature=temperature, max_tokens=max_tokens,
            )
        resp = with_retry(_call, retries=2)
        return resp.choices[0].message.content or ""
    except Exception as e:
        logger.error("Groq call failed: %s", e)
        return ""


def mistral_chat(messages: list[dict], temperature: float = 0.3, max_tokens: int = 1600) -> str:
    """Non-streaming Mistral call — used for evaluation, scoring, and reports."""
    if not mistral_available():
        return ""
    try:
        client = Mistral(api_key=_get_api_key("mistral"))

        def _call():
            return client.chat.complete(
                model=MISTRAL_MODEL, messages=messages, temperature=temperature, max_tokens=max_tokens,
            )
        resp = with_retry(_call, retries=2)
        return resp.choices[0].message.content or ""
    except Exception as e:
        logger.error("Mistral call failed: %s", e)
        return ""


def gemini_chat(prompt: str, temperature: float = 0.5) -> str:
    """Optional Gemini call. Never used for mandatory features."""
    if not gemini_available():
        return ""
    try:
        genai.configure(api_key=_get_api_key("gemini"))

        def _call():
            model = genai.GenerativeModel(GEMINI_MODEL)
            return model.generate_content(
                prompt, generation_config={"temperature": temperature}
            )
        resp = with_retry(_call, retries=1)
        return resp.text or ""
    except Exception as e:
        logger.error("Gemini call failed: %s", e)
        return ""


def extract_json(text: str) -> Optional[dict]:
    """Extract the first valid JSON object from a model response, tolerating
    markdown code fences and surrounding prose."""
    if not text:
        return None
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    candidate = match.group(0) if match else text
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # Attempt light repair: trailing commas
        repaired = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            return None

# ============================================================================
# SECTION 6: INTERVIEW ENGINE — prompt construction (Groq conducts interview)
# ============================================================================

def build_system_prompt(track: str, company: str, persona: str, mode: str,
                          difficulty: str) -> str:
    """Construct a system prompt that fuses persona + company + mode + difficulty
    so every combination genuinely changes interviewer behavior."""
    persona_desc = PERSONAS.get(persona, "a professional interviewer")
    mode_instr = MODE_INSTRUCTIONS.get(mode, MODE_INSTRUCTIONS["Standard"])

    company_block = ""
    if company and company != "General / No Company":
        cp = COMPANIES.get(company)
        if cp:
            company_block = f"""
You are interviewing on behalf of {cp.name}. At {cp.name}, interviews emphasize
{cp.focus}. Your interviewing style is {cp.style}. You are evaluating whether the
candidate meets this bar: {cp.bar}. Where natural, weave in signature topics such as
{', '.join(cp.signature_topics)}."""

    return f"""You are {persona_desc}, conducting a live {track} interview.
Difficulty level: {difficulty}.
{mode_instr}
{company_block}

RULES:
- Ask ONE question at a time, then wait for the candidate's answer.
- Keep questions focused and realistic for a {difficulty.lower()}-level candidate.
- After the candidate answers, briefly acknowledge (1 sentence max), optionally ask a
  short natural follow-up, then move to the next question when appropriate.
- Do not reveal scores or evaluations during the interview — that happens afterward.
- Stay fully in character as the persona and company described above.
- Keep your responses concise (2-5 sentences) like a real spoken interview, not an essay.
- If this is the very first message, greet the candidate briefly and ask your first question.
"""


def get_opening_message(track: str, company: str, persona: str, mode: str,
                          difficulty: str) -> str:
    """Generate the interviewer's opening line + first question via Groq."""
    system_prompt = build_system_prompt(track, company, persona, mode, difficulty)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Please begin the interview now."},
    ]
    if not groq_available():
        return (
            f"👋 Hi, I'm your {persona} for this {track} interview"
            f"{' at ' + company if company and company != 'General / No Company' else ''}. "
            f"*(Demo mode — connect a Groq API key in Settings for a live AI-driven interview.)* "
            f"To start: Can you walk me through a recent project you're proud of and your specific role in it?"
        )
    return groq_chat(messages, temperature=0.8, max_tokens=300) or (
        "Hi! Let's get started — tell me about yourself and why you're interested in this role."
    )


def stream_interviewer_reply(system_prompt: str, transcript: list[dict]):
    """Yield streamed interviewer reply chunks given the running transcript."""
    messages = [{"role": "system", "content": system_prompt}]
    for turn in transcript:
        role = "assistant" if turn["role"] == "interviewer" else "user"
        messages.append({"role": role, "content": turn["content"]})
    yield from groq_chat_stream(messages, temperature=0.8, max_tokens=400)


def generate_coding_question(track: str, difficulty: str, language: str) -> dict:
    """Ask Groq for a structured coding question. Falls back to a static bank."""
    fallback_bank = {
        "Beginner": {"title": "Two Sum", "prompt": "Given an array of integers and a target, return indices of the two numbers that add up to the target."},
        "Intermediate": {"title": "Longest Substring Without Repeating Characters", "prompt": "Given a string, find the length of the longest substring without repeating characters."},
        "Advanced": {"title": "Merge K Sorted Lists", "prompt": "Merge k sorted linked lists into one sorted linked list."},
        "Expert": {"title": "Word Ladder II", "prompt": "Find all shortest transformation sequences from beginWord to endWord, changing one letter at a time using a given dictionary."},
    }
    if not groq_available():
        base = fallback_bank.get(difficulty, fallback_bank["Intermediate"])
        return {"title": base["title"], "prompt": base["prompt"], "language": language, "starter_code": ""}

    sys_prompt = (
        f"You are a coding interview question generator for the '{track}' track at "
        f"{difficulty} difficulty, language {language}. Respond ONLY with strict JSON: "
        '{"title": "...", "prompt": "...", "starter_code": "..."}. No markdown fences.'
    )
    raw = groq_chat(
        [{"role": "system", "content": sys_prompt}, {"role": "user", "content": "Generate one question."}],
        temperature=0.9, max_tokens=500,
    )
    parsed = extract_json(raw)
    if parsed and "title" in parsed and "prompt" in parsed:
        parsed["language"] = language
        parsed.setdefault("starter_code", "")
        return parsed
    base = fallback_bank.get(difficulty, fallback_bank["Intermediate"])
    return {"title": base["title"], "prompt": base["prompt"], "language": language, "starter_code": ""}


def review_code(question: dict, code: str, language: str) -> str:
    """AI code review: correctness, complexity, optimizations, edge cases."""
    if not code.strip():
        return "Please write some code before requesting a review."
    if not groq_available():
        return (
            "⚠️ Groq is not configured, so a live code review isn't available. "
            "In demo mode: check your solution handles empty input, duplicate values, "
            "and large inputs efficiently before submitting."
        )
    sys_prompt = (
        "You are a rigorous but constructive code reviewer for a technical interview. "
        "Review the candidate's code for: correctness, time complexity, space complexity, "
        "possible optimizations, and edge cases missed. Be specific and concise. Use short "
        "markdown sections with headers: Correctness, Complexity, Optimizations, Edge Cases."
    )
    user_prompt = (
        f"Question: {question.get('title')}\n{question.get('prompt')}\n\n"
        f"Candidate's {language} code:\n```{language.lower()}\n{code}\n```"
    )
    result = groq_chat(
        [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
        temperature=0.4, max_tokens=700,
    )
    return result or "⚠️ Code review failed — please try again."

# ============================================================================
# SECTION 7: MULTI-AGENT EVALUATION PIPELINE (Mistral evaluates & scores)
# ============================================================================

def _transcript_to_text(transcript: list[dict]) -> str:
    lines = []
    for turn in transcript:
        speaker = "Interviewer" if turn["role"] == "interviewer" else "Candidate"
        lines.append(f"{speaker}: {turn['content']}")
    return "\n".join(lines)


def run_evaluation_agents(transcript: list[dict], track: str, company: str,
                            difficulty: str) -> dict:
    """Run the multi-agent evaluation pipeline via Mistral and aggregate into
    a single structured report. Falls back to a heuristic estimate if Mistral
    is unavailable, so the app never breaks."""
    transcript_text = _transcript_to_text(transcript)

    if not mistral_available():
        return _heuristic_evaluation(transcript, track, difficulty)

    agent_list = "\n".join(f"- {name}: {desc}" for name, desc in EVALUATION_AGENTS.items())
    sys_prompt = f"""You are a panel of specialized interview evaluation agents, followed by a
Hiring Manager and a Final Aggregator. The agents are:
{agent_list}

Analyze the following {track} interview transcript (company context: {company}, difficulty:
{difficulty}) and respond with STRICT JSON only, no markdown fences, matching exactly this
schema:
{{
  "technical_score": <0-100 int>,
  "communication_score": <0-100 int>,
  "confidence_score": <0-100 int>,
  "behavior_score": <0-100 int>,
  "leadership_score": <0-100 int>,
  "problem_solving_score": <0-100 int>,
  "grammar_score": <0-100 int>,
  "overall_score": <0-100 int, weighted holistic average>,
  "hiring_recommendation": "<one of: Strong Hire, Hire, Hold, Reject>",
  "strengths": ["...", "..."],
  "weaknesses": ["...", "..."],
  "improvement_areas": ["...", "..."],
  "roadmap": {{
     "courses": ["...", "..."],
     "books": ["...", "..."],
     "leetcode": ["...", "..."],
     "projects": ["...", "..."]
  }}
}}"""

    raw = mistral_chat(
        [{"role": "system", "content": sys_prompt},
         {"role": "user", "content": f"Transcript:\n{transcript_text}"}],
        temperature=0.2, max_tokens=1800,
    )
    parsed = extract_json(raw)
    if parsed:
        return _normalize_evaluation(parsed)
    logger.warning("Mistral evaluation JSON parse failed — using heuristic fallback.")
    return _heuristic_evaluation(transcript, track, difficulty)


def _normalize_evaluation(parsed: dict) -> dict:
    """Clamp scores to 0-100 and ensure all expected keys exist."""
    score_keys = ["technical_score", "communication_score", "confidence_score",
                  "behavior_score", "leadership_score", "problem_solving_score",
                  "grammar_score", "overall_score"]
    for k in score_keys:
        try:
            parsed[k] = max(0, min(100, float(parsed.get(k, 0))))
        except (TypeError, ValueError):
            parsed[k] = 0
    parsed.setdefault("hiring_recommendation", "Hold")
    parsed.setdefault("strengths", [])
    parsed.setdefault("weaknesses", [])
    parsed.setdefault("improvement_areas", [])
    parsed.setdefault("roadmap", {"courses": [], "books": [], "leetcode": [], "projects": []})
    return parsed


def _heuristic_evaluation(transcript: list[dict], track: str, difficulty: str) -> dict:
    """A transparent, deterministic fallback scorer used only when no AI evaluator
    is configured, so the platform remains fully usable offline."""
    candidate_turns = [t["content"] for t in transcript if t["role"] == "candidate"]
    if not candidate_turns:
        base = 0
    else:
        avg_len = np.mean([len(t.split()) for t in candidate_turns])
        base = int(min(100, max(20, avg_len * 2)))
    rng = random.Random(len(transcript))
    jitter = lambda: max(0, min(100, base + rng.randint(-10, 10)))
    scores = {
        "technical_score": jitter(), "communication_score": jitter(),
        "confidence_score": jitter(), "behavior_score": jitter(),
        "leadership_score": jitter(), "problem_solving_score": jitter(),
        "grammar_score": jitter(),
    }
    overall = round(np.mean(list(scores.values())), 1)
    scores["overall_score"] = overall
    rec = "Strong Hire" if overall >= 85 else "Hire" if overall >= 70 else "Hold" if overall >= 50 else "Reject"
    scores["hiring_recommendation"] = rec
    scores["strengths"] = ["Engaged with each question", "Attempted structured answers"]
    scores["weaknesses"] = ["Connect an AI provider in Settings for a detailed, accurate evaluation"]
    scores["improvement_areas"] = [f"Deepen fundamentals in {track}"]
    scores["roadmap"] = {
        "courses": [f"Intro to {track} — recommended once AI evaluation is enabled"],
        "books": ["Cracking the Coding Interview"],
        "leetcode": ["Two Sum", "Valid Parentheses", "Binary Tree Level Order Traversal"],
        "projects": [f"Build a small {track}-focused portfolio project"],
    }
    scores["_offline_estimate"] = True
    return scores

# ============================================================================
# SECTION 8: RESUME ANALYZER
# ============================================================================

COMMON_SKILLS = [
    "Python", "Java", "C++", "JavaScript", "TypeScript", "SQL", "React", "Node.js",
    "Django", "Flask", "FastAPI", "Streamlit", "AWS", "Azure", "GCP", "Docker",
    "Kubernetes", "Git", "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch",
    "Pandas", "NumPy", "Data Structures", "Algorithms", "System Design", "REST API",
    "GraphQL", "MongoDB", "PostgreSQL", "MySQL", "Redis", "CI/CD", "Linux", "Agile",
    "HTML", "CSS", "NLP", "Computer Vision", "Spring Boot", "Microservices",
]


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF using pdfplumber, falling back to PyPDF2."""
    text = ""
    if HAS_PDFPLUMBER:
        try:
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                text = "\n".join(page.extract_text() or "" for page in pdf.pages)
            if text.strip():
                return text
        except Exception as e:
            logger.warning("pdfplumber extraction failed: %s", e)
    if HAS_PYPDF2:
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as e:
            logger.warning("PyPDF2 extraction failed: %s", e)
    return text


def analyze_resume(text: str, target_role: str = "") -> dict:
    """Extract skills / sections and compute an ATS-style score.
    Uses Groq for richer extraction when available, else regex heuristics."""
    text_lower = text.lower()
    found_skills = [s for s in COMMON_SKILLS if s.lower() in text_lower]

    sections = {
        "has_education": bool(re.search(r"\beducation\b", text_lower)),
        "has_experience": bool(re.search(r"\bexperience\b|\binternship\b", text_lower)),
        "has_projects": bool(re.search(r"\bprojects?\b", text_lower)),
        "has_skills": bool(re.search(r"\bskills?\b", text_lower)),
        "has_certifications": bool(re.search(r"\bcertificat", text_lower)),
        "has_contact": bool(re.search(r"@[\w.-]+\.\w+", text)),
    }
    section_score = sum(sections.values()) / len(sections) * 40
    skill_score = min(40, len(found_skills) * 3)
    length_score = 20 if 200 <= len(text.split()) <= 1200 else 10
    ats_score = round(section_score + skill_score + length_score, 1)

    missing_skills, suggestions = [], []
    if groq_available():
        sys_prompt = (
            "You are an ATS resume analyzer and career coach. Given resume text"
            + (f" and target role '{target_role}'" if target_role else "")
            + ", respond with STRICT JSON only: "
              '{"missing_skills": ["..."], "suggestions": ["..."], "summary": "1-2 sentence summary"}'
        )
        raw = groq_chat(
            [{"role": "system", "content": sys_prompt},
             {"role": "user", "content": text[:6000]}],
            temperature=0.3, max_tokens=600,
        )
        parsed = extract_json(raw)
        if parsed:
            missing_skills = parsed.get("missing_skills", [])
            suggestions = parsed.get("suggestions", [])

    if not suggestions:
        for key, present in sections.items():
            if not present:
                label = key.replace("has_", "").replace("_", " ")
                suggestions.append(f"Add a clear '{label}' section — it's missing or hard to detect.")
        if len(found_skills) < 5:
            suggestions.append("List more relevant technical skills explicitly (a dedicated Skills section helps ATS parsing).")
        if not suggestions:
            suggestions.append("Resume structure looks solid — focus on quantifying achievements with metrics.")

    return {
        "ats_score": ats_score,
        "skills": found_skills,
        "missing_skills": missing_skills,
        "suggestions": suggestions,
        "sections": sections,
    }


# ============================================================================
# SECTION 9: PDF REPORT GENERATION (ReportLab)
# ============================================================================

def generate_report_pdf(user_name: str, interview_row: sqlite3.Row, evaluation: dict) -> Optional[bytes]:
    """Build a polished PDF interview report. Returns None if reportlab is
    unavailable (caller should fall back to Markdown/HTML export)."""
    if not HAS_REPORTLAB:
        return None
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.6 * inch, bottomMargin=0.6 * inch)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleX", parent=styles["Title"], textColor=colors.HexColor("#6C5CE7"))
    h2 = ParagraphStyle("H2X", parent=styles["Heading2"], textColor=colors.HexColor("#2D3436"))
    body = styles["BodyText"]

    story = [
        Paragraph(f"{APP_NAME} — Interview Report", title_style),
        Spacer(1, 6),
        Paragraph(f"Candidate: {user_name}", body),
        Paragraph(f"Track: {interview_row['track']} &nbsp;|&nbsp; Company: {interview_row['company'] or 'General'}", body),
        Paragraph(f"Persona: {interview_row['persona']} &nbsp;|&nbsp; Difficulty: {interview_row['difficulty']} &nbsp;|&nbsp; Mode: {interview_row['mode']}", body),
        Paragraph(f"Date: {interview_row['started_at'][:19].replace('T', ' ')}", body),
        Spacer(1, 14),
        Paragraph("Scores", h2),
    ]

    score_rows = [["Metric", "Score / 100"]]
    for label, key in [
        ("Overall", "overall_score"), ("Technical", "technical_score"),
        ("Communication", "communication_score"), ("Confidence", "confidence_score"),
        ("Behavior", "behavior_score"), ("Leadership", "leadership_score"),
        ("Problem Solving", "problem_solving_score"), ("Grammar", "grammar_score"),
    ]:
        score_rows.append([label, str(round(evaluation.get(key, 0), 1))])

    table = Table(score_rows, colWidths=[3 * inch, 2 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#6C5CE7")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#F0EEFF")),
    ]))
    story.append(table)
    story.append(Spacer(1, 14))

    story.append(Paragraph(f"Hiring Recommendation: <b>{evaluation.get('hiring_recommendation', 'Hold')}</b>", h2))
    story.append(Spacer(1, 10))

    for label, key in [("Strengths", "strengths"), ("Weaknesses", "weaknesses"),
                          ("Improvement Areas", "improvement_areas")]:
        items = evaluation.get(key, [])
        if items:
            story.append(Paragraph(label, h2))
            for item in items:
                story.append(Paragraph(f"• {item}", body))
            story.append(Spacer(1, 8))

    roadmap = evaluation.get("roadmap", {})
    if roadmap:
        story.append(Paragraph("Learning Roadmap", h2))
        for label, key in [("Courses", "courses"), ("Books", "books"),
                              ("LeetCode Practice", "leetcode"), ("Projects", "projects")]:
            items = roadmap.get(key, [])
            if items:
                story.append(Paragraph(f"<b>{label}:</b> {', '.join(items)}", body))
        story.append(Spacer(1, 10))

    story.append(Spacer(1, 20))
    story.append(Paragraph(f"Generated by {APP_NAME} on {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", styles["Italic"]))

    doc.build(story)
    return buffer.getvalue()


def generate_report_markdown(user_name: str, interview_row: sqlite3.Row, evaluation: dict) -> str:
    lines = [
        f"# {APP_NAME} — Interview Report", "",
        f"**Candidate:** {user_name}  ",
        f"**Track:** {interview_row['track']}  ",
        f"**Company:** {interview_row['company'] or 'General'}  ",
        f"**Persona:** {interview_row['persona']}  ",
        f"**Difficulty:** {interview_row['difficulty']} | **Mode:** {interview_row['mode']}  ",
        f"**Date:** {interview_row['started_at'][:19].replace('T', ' ')}", "",
        "## Scores", "",
        "| Metric | Score |", "|---|---|",
    ]
    for label, key in [
        ("Overall", "overall_score"), ("Technical", "technical_score"),
        ("Communication", "communication_score"), ("Confidence", "confidence_score"),
        ("Behavior", "behavior_score"), ("Leadership", "leadership_score"),
        ("Problem Solving", "problem_solving_score"), ("Grammar", "grammar_score"),
    ]:
        lines.append(f"| {label} | {round(evaluation.get(key, 0), 1)} |")
    lines += ["", f"**Hiring Recommendation:** {evaluation.get('hiring_recommendation', 'Hold')}", ""]
    for label, key in [("Strengths", "strengths"), ("Weaknesses", "weaknesses"),
                          ("Improvement Areas", "improvement_areas")]:
        items = evaluation.get(key, [])
        if items:
            lines.append(f"## {label}")
            lines += [f"- {i}" for i in items]
            lines.append("")
    roadmap = evaluation.get("roadmap", {})
    if roadmap:
        lines.append("## Learning Roadmap")
        for label, key in [("Courses", "courses"), ("Books", "books"),
                              ("LeetCode Practice", "leetcode"), ("Projects", "projects")]:
            items = roadmap.get(key, [])
            if items:
                lines.append(f"- **{label}:** {', '.join(items)}")
    return "\n".join(lines)


def generate_report_html(user_name: str, interview_row: sqlite3.Row, evaluation: dict) -> str:
    md = generate_report_markdown(user_name, interview_row, evaluation)
    body = md.replace("\n", "<br>")
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>{APP_NAME} Report</title>
<style>body{{font-family:Arial,sans-serif;background:#0f0f1a;color:#eee;padding:2rem;}}
h1,h2{{color:#a78bfa;}}</style></head><body>{body}</body></html>"""

# ============================================================================
# SECTION 10: THEME / CSS (Glassmorphism Dark Theme)
# ============================================================================

def inject_css() -> None:
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    :root {
        --bg-primary: #0a0a12;
        --bg-secondary: #12121f;
        --glass-bg: rgba(255, 255, 255, 0.04);
        --glass-border: rgba(255, 255, 255, 0.09);
        --accent-1: #6C5CE7;
        --accent-2: #00CEC9;
        --accent-3: #FD79A8;
        --text-primary: #F5F5F7;
        --text-secondary: #9A9AB0;
        --success: #00E396;
        --warning: #FDCB6E;
        --danger: #FF6B6B;
    }

    .stApp {
        background: radial-gradient(circle at 15% 20%, #1a1a2e 0%, #0a0a12 45%),
                    radial-gradient(circle at 85% 80%, #16162a 0%, #0a0a12 55%);
        color: var(--text-primary);
    }

    #MainMenu, footer, header {visibility: hidden;}

    /* ---- Glass Card ---- */
    .glass-card {
        background: var(--glass-bg);
        border: 1px solid var(--glass-border);
        border-radius: 18px;
        padding: 1.4rem 1.6rem;
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.35);
        transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
        margin-bottom: 1rem;
    }
    .glass-card:hover {
        transform: translateY(-3px);
        border-color: rgba(108, 92, 231, 0.5);
        box-shadow: 0 12px 40px rgba(108, 92, 231, 0.18);
    }

    .gradient-text {
        background: linear-gradient(90deg, var(--accent-1), var(--accent-3) 60%, var(--accent-2));
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; font-weight: 800;
    }

    .stat-card {
        background: linear-gradient(135deg, rgba(108,92,231,0.15), rgba(0,206,201,0.08));
        border: 1px solid var(--glass-border);
        border-radius: 16px; padding: 1.1rem 1.3rem;
        backdrop-filter: blur(14px);
        transition: transform 0.2s ease;
    }
    .stat-card:hover { transform: translateY(-2px) scale(1.01); }
    .stat-value { font-size: 1.8rem; font-weight: 800; color: var(--text-primary); }
    .stat-label { font-size: 0.8rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.06em; }

    .badge {
        display: inline-block; padding: 0.25rem 0.7rem; border-radius: 999px;
        font-size: 0.75rem; font-weight: 600; letter-spacing: 0.02em;
    }
    .badge-success { background: rgba(0,227,150,0.15); color: var(--success); border: 1px solid rgba(0,227,150,0.35); }
    .badge-warning { background: rgba(253,203,110,0.15); color: var(--warning); border: 1px solid rgba(253,203,110,0.35); }
    .badge-danger { background: rgba(255,107,107,0.15); color: var(--danger); border: 1px solid rgba(255,107,107,0.35); }
    .badge-accent { background: rgba(108,92,231,0.18); color: #b3a6ff; border: 1px solid rgba(108,92,231,0.4); }

    /* Buttons */
    .stButton > button {
        border-radius: 12px !important;
        border: 1px solid var(--glass-border) !important;
        background: linear-gradient(135deg, var(--accent-1), #8b7bf0) !important;
        color: white !important; font-weight: 600 !important;
        transition: all 0.2s ease !important; box-shadow: 0 4px 14px rgba(108,92,231,0.25);
    }
    .stButton > button:hover {
        transform: translateY(-2px); box-shadow: 0 8px 22px rgba(108,92,231,0.4);
        border-color: rgba(255,255,255,0.3) !important;
    }

    /* Chat bubbles */
    .chat-bubble-interviewer {
        background: var(--glass-bg); border: 1px solid var(--glass-border);
        border-radius: 16px 16px 16px 4px; padding: 0.9rem 1.1rem; margin: 0.5rem 0;
        max-width: 85%; backdrop-filter: blur(12px); animation: fadeIn 0.35s ease;
    }
    .chat-bubble-candidate {
        background: linear-gradient(135deg, rgba(108,92,231,0.25), rgba(108,92,231,0.12));
        border: 1px solid rgba(108,92,231,0.35);
        border-radius: 16px 16px 4px 16px; padding: 0.9rem 1.1rem; margin: 0.5rem 0 0.5rem auto;
        max-width: 85%; animation: fadeIn 0.35s ease;
    }
    .chat-label { font-size: 0.72rem; color: var(--text-secondary); margin-bottom: 0.2rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;}

    @keyframes fadeIn { from {opacity:0; transform: translateY(6px);} to {opacity:1; transform: translateY(0);} }
    @keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.4;} }
    .typing-dot { animation: pulse 1.2s infinite; }

    .hero-title { font-size: 3.2rem; font-weight: 800; line-height: 1.1; }
    .hero-sub { font-size: 1.15rem; color: var(--text-secondary); margin-top: 0.6rem; }

    .feature-card {
        background: var(--glass-bg); border: 1px solid var(--glass-border);
        border-radius: 18px; padding: 1.5rem; height: 100%;
        transition: all 0.25s ease;
    }
    .feature-card:hover { border-color: var(--accent-1); transform: translateY(-4px); }
    .feature-icon { font-size: 2rem; margin-bottom: 0.6rem; }

    .company-chip {
        display: inline-block; padding: 0.4rem 0.9rem; border-radius: 10px;
        background: var(--glass-bg); border: 1px solid var(--glass-border);
        margin: 0.2rem; font-size: 0.85rem; color: var(--text-secondary);
    }

    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: var(--bg-secondary); }
    ::-webkit-scrollbar-thumb { background: #3a3a55; border-radius: 8px; }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0e0e1a 0%, #0a0a12 100%);
        border-right: 1px solid var(--glass-border);
    }

    div[data-testid="stMetric"] {
        background: var(--glass-bg); border: 1px solid var(--glass-border);
        border-radius: 14px; padding: 0.8rem;
    }
    </style>
    """, unsafe_allow_html=True)


def glass_card_open(extra_class: str = "") -> None:
    st.markdown(f'<div class="glass-card {extra_class}">', unsafe_allow_html=True)


def glass_card_close() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def badge(text: str, kind: str = "accent") -> str:
    return f'<span class="badge badge-{kind}">{text}</span>'


def stat_card(label: str, value: str, icon: str = "") -> str:
    return f"""<div class="stat-card"><div class="stat-label">{icon} {label}</div>
    <div class="stat-value">{value}</div></div>"""

# ============================================================================
# SECTION 11: SESSION STATE MANAGEMENT
# ============================================================================

def init_session_state() -> None:
    defaults = {
        "auth_token": None, "user": None, "page": "landing",
        "auth_view": "login", "active_interview_id": None,
        "interview_transcript": [], "interview_system_prompt": "",
        "interview_config": {}, "interview_started_at": None,
        "coding_question": None, "coding_review": None,
        "viewing_report_interview_id": None, "resume_last_analysis": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def refresh_user() -> None:
    if st.session_state.get("user"):
        row = db_get_user_by_id(st.session_state["user"]["id"])
        if row:
            st.session_state["user"] = dict(row)


def do_login(user_row: sqlite3.Row) -> None:
    token = create_session_token(user_row["id"])
    st.session_state["auth_token"] = token
    st.session_state["user"] = dict(user_row)
    st.session_state["page"] = "dashboard"


def do_logout() -> None:
    for k in ["auth_token", "user", "active_interview_id", "interview_transcript",
              "interview_system_prompt", "interview_config"]:
        st.session_state[k] = None if k in ("auth_token", "user", "active_interview_id") else ([] if "transcript" in k else ("" if "prompt" in k else {}))
    st.session_state["page"] = "landing"


def require_login() -> bool:
    if not st.session_state.get("user"):
        st.session_state["page"] = "landing"
        st.rerun()
        return False
    return True


# ============================================================================
# SECTION 12: LANDING PAGE
# ============================================================================

TESTIMONIALS = [
    ("Ananya R.", "Placed at a top product company", "HireNova AI's stress-mode interview was tougher than my real one — I walked in ready."),
    ("Rohit K.", "Final-year CS student", "The company-specific personas actually feel different. Google's interviewer really does keep asking 'why'."),
    ("Priya S.", "Career switcher into Data Science", "The resume analyzer caught three missing keywords that got me past the first screen."),
]

FAQS = [
    ("Do I need an API key to try it?", "No — the platform runs in a fully functional demo mode without any keys. Add a Groq and/or Mistral key in Settings for live AI-driven interviews and evaluations."),
    ("Which AI models power the platform?", "Groq conducts your interview and generates questions in real time. Mistral evaluates your answers and builds your report. Google Gemini AI is optional and never required."),
    ("Is my data private?", "Your interviews, reports, and resume data are stored locally in this app's SQLite database, tied to your account."),
    ("Can I practice company-specific interviews?", "Yes — pick from 18 top companies, each with a distinct interviewer style, focus area, and evaluation bar."),
]

ABOUT_TEXT = (
    f"{APP_NAME} is a premium enterprise AI interview and career intelligence platform designed to help "
    "students and professionals prepare for technical interviews, coding assessments, behavioral rounds, "
    "system design discussions, resume screening, and company-specific hiring processes.\n\n"
    "The platform combines multiple AI models to create realistic interview simulations, evaluate candidate "
    "performance, generate personalized learning roadmaps, and provide actionable career insights."
)


@st.dialog(f"🎬 {APP_NAME} — Live Demo Preview")
def _render_demo_dialog() -> None:
    st.caption("A preview of what your post-interview report looks like — no signup required.")
    demo_categories = ["Technical", "Communication", "Confidence", "Behavior", "Leadership", "Problem Solving", "Grammar"]
    demo_values = [82, 76, 71, 80, 68, 85, 79]
    fig = go.Figure(data=go.Scatterpolar(
        r=demo_values + [demo_values[0]], theta=demo_categories + [demo_categories[0]], fill="toself",
        line_color="#6C5CE7", fillcolor="rgba(108,92,231,0.35)",
    ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100], gridcolor="#333"),
                                   bgcolor="rgba(0,0,0,0)"), showlegend=False, height=320)
    fig = _plotly_dark_layout(fig)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f"**Overall Score:** 77.3/100 &nbsp;&nbsp; {badge('Hire', 'success')}", unsafe_allow_html=True)
    st.markdown("**Sample strengths:** Clear structured answers · Strong ownership language · Good technical depth")
    st.markdown("**Sample roadmap:** *System Design Interview* course · *Cracking the Coding Interview* · 3 curated LeetCode problems")
    st.markdown("---")
    if st.button("🚀 Start Free — Get My Real Report", use_container_width=True):
        st.session_state["page"] = "auth"; st.session_state["auth_view"] = "signup"; st.rerun()


def render_landing_page() -> None:
    st.markdown(f"""
    <div style="text-align:center; padding: 2.2rem 1rem 0.4rem;">
        <div class="badge badge-accent">✨ {APP_EDITION} · AI-Powered Interview Intelligence</div>
        <div class="hero-title gradient-text" style="margin-top:0.8rem;">{APP_NAME}</div>
        <div class="hero-sub">{APP_TAGLINE}</div>
        <div style="color:var(--text-secondary); font-size:0.95rem; max-width:680px; margin:0.7rem auto 0;">
            Prepare for technical interviews, coding rounds, behavioral interviews, resume screening, ATS
            optimization, and company-specific mock interviews using advanced AI models.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="text-align:center; color:var(--text-secondary); font-size:0.8rem; margin-bottom:1rem;">
        ⚡ Powered by <b>Groq AI</b> · <b>Mistral AI</b> · <b>Google Gemini AI</b> <span style="opacity:0.7;">(optional)</span>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns([0.8, 0.8, 0.8, 0.8])
    with col1:
        if st.button("🚀 Start Free", use_container_width=True):
            st.session_state["page"] = "auth"; st.session_state["auth_view"] = "signup"; st.rerun()
    with col2:
        if st.button("🧭 Explore Features", use_container_width=True):
            st.session_state["landing_scroll_target"] = "features-section"; st.rerun()
    with col3:
        if st.button("🎬 View Demo", use_container_width=True):
            _render_demo_dialog()
    with col4:
        if st.button("🔑 Log In", use_container_width=True):
            st.session_state["page"] = "auth"; st.session_state["auth_view"] = "login"; st.rerun()

    scroll_target = st.session_state.pop("landing_scroll_target", None)
    if scroll_target:
        st.components.v1.html(f"""
        <script>
        setTimeout(function() {{
            const el = window.parent.document.getElementById('{scroll_target}');
            if (el) {{ el.scrollIntoView({{behavior: 'smooth', block: 'start'}}); }}
        }}, 150);
        </script>
        """, height=0)

    st.markdown("<br>", unsafe_allow_html=True)
    stat_cols = st.columns(4)
    stats = [("18", "Company Personas", "🏢"), ("20", "Interview Tracks", "🎯"),
             ("12", "AI Interviewer Personas", "🧑‍💼"), ("100%", "Free to Practice", "⚡")]
    for col, (val, label, icon) in zip(stat_cols, stats):
        with col:
            st.markdown(stat_card(label, val, icon), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    glass_card_open()
    st.markdown("#### 🏢 About HireNova AI")
    st.write(ABOUT_TEXT)
    glass_card_close()

    st.markdown('<div id="features-section"></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f'<h3 class="gradient-text">Why {APP_NAME}</h3>', unsafe_allow_html=True)
    features = [
        ("🎙️", "Live AI Interviews", "Streamed, conversational interviews powered by Groq — not canned questions."),
        ("🏢", "18 Company Personas", "Google, Amazon, Meta, Netflix and more — each with a distinct bar and style."),
        ("🧠", "Multi-Agent Evaluation", "Seven specialized AI agents score technical skill, communication, confidence, and more."),
        ("💻", "Coding Interviews", "In-browser code editor with AI review of complexity, correctness, and edge cases."),
        ("📄", "Resume Analyzer", "ATS scoring, skill-gap detection, and tailored suggestions from your PDF resume."),
        ("📊", "Deep Analytics", "Radar charts, skill heatmaps, and progress tracking across every session."),
    ]
    fcols = st.columns(3)
    for i, (icon, title, desc) in enumerate(features):
        with fcols[i % 3]:
            st.markdown(f"""<div class="feature-card"><div class="feature-icon">{icon}</div>
            <b>{title}</b><p style="color:var(--text-secondary); font-size:0.9rem;">{desc}</p></div>""",
            unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<h3 class="gradient-text">Practice for Any Company</h3>', unsafe_allow_html=True)
    chips = "".join(f'<span class="company-chip">{c}</span>' for c in COMPANIES.keys())
    st.markdown(f'<div>{chips}</div>', unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown('<h3 class="gradient-text">What Candidates Say</h3>', unsafe_allow_html=True)
    tcols = st.columns(3)
    for col, (name, role, quote) in zip(tcols, TESTIMONIALS):
        with col:
            glass_card_open()
            st.markdown(f'"{quote}"')
            st.markdown(f"**{name}** · <span style='color:var(--text-secondary)'>{role}</span>", unsafe_allow_html=True)
            glass_card_close()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<h3 class="gradient-text">Simple Pricing</h3>', unsafe_allow_html=True)
    pcols = st.columns(3)
    plans = [
        ("Free", "₹0", ["Unlimited demo interviews", "Basic analytics", "Community support"], False),
        ("Pro", "₹499/mo", ["Live AI interviews (BYO API key)", "Full multi-agent reports", "PDF export", "Resume analyzer"], True),
        ("Enterprise", "Contact us", ["Team dashboards", "Custom company personas", "Priority support"], False),
    ]
    for col, (name, price, feats, highlighted) in zip(pcols, plans):
        with col:
            glass_card_open("gradient-text" if highlighted else "")
            st.markdown(f"### {name}\n#### {price}")
            for f in feats:
                st.markdown(f"✅ {f}")
            glass_card_close()
    st.caption("Demo pricing shown for illustration — this build runs fully free with your own API keys.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<h3 class="gradient-text">FAQ</h3>', unsafe_allow_html=True)
    for q, a in FAQS:
        with st.expander(q):
            st.write(a)

    st.markdown("<br><hr style='border-color:var(--glass-border);'>", unsafe_allow_html=True)
    st.markdown(f"""<div style="text-align:center; color:var(--text-secondary); padding:1rem;">
    {APP_NAME} · Built with Streamlit, Groq &amp; Mistral · © {datetime.utcnow().year}
    </div>""", unsafe_allow_html=True)

# ============================================================================
# SECTION 13: AUTH PAGE (Signup / Login / Forgot Password)
# ============================================================================

def render_auth_page() -> None:
    _, mid, _ = st.columns([1, 1.1, 1])
    with mid:
        st.markdown(f'<h2 class="gradient-text" style="text-align:center;">{APP_NAME}</h2>', unsafe_allow_html=True)
        glass_card_open()
        tabs = st.tabs(["Log In", "Sign Up", "Forgot Password"])

        with tabs[0]:
            with st.form("login_form"):
                email = st.text_input("Email", key="login_email")
                password = st.text_input("Password", type="password", key="login_password")
                remember = st.checkbox("Remember me", value=True)
                submitted = st.form_submit_button("Log In", use_container_width=True)
            if submitted:
                if not is_valid_email(email):
                    st.error("Enter a valid email address.")
                else:
                    user = db_get_user_by_email(email)
                    if user and verify_password(password, user["password_hash"]):
                        do_login(user)
                        st.success("Welcome back!")
                        st.rerun()
                    else:
                        st.error("Invalid email or password.")

        with tabs[1]:
            with st.form("signup_form"):
                name = st.text_input("Full Name")
                email_s = st.text_input("Email", key="signup_email")
                password_s = st.text_input("Password", type="password", key="signup_password")
                confirm = st.text_input("Confirm Password", type="password")
                submitted_s = st.form_submit_button("Create Account", use_container_width=True)
            if submitted_s:
                if not name.strip():
                    st.error("Please enter your name.")
                elif not is_valid_email(email_s):
                    st.error("Enter a valid email address.")
                elif password_s != confirm:
                    st.error("Passwords do not match.")
                else:
                    ok, msg = is_strong_password(password_s)
                    if not ok:
                        st.error(msg)
                    else:
                        uid = db_create_user(name.strip(), email_s, password_s)
                        if uid is None:
                            st.error("An account with this email already exists.")
                        else:
                            user = db_get_user_by_id(uid)
                            do_login(user)
                            db_grant_achievement(uid, "Welcome Aboard", "Created your InterviewVerse AI Pro account.")
                            st.success("Account created! Redirecting to your dashboard...")
                            st.rerun()

        with tabs[2]:
            st.info("Demo build: password reset emails aren't wired to a mail server. "
                    "Enter your email and a new password below to reset it directly.")
            with st.form("forgot_form"):
                email_f = st.text_input("Account Email")
                new_pw = st.text_input("New Password", type="password")
                submitted_f = st.form_submit_button("Reset Password", use_container_width=True)
            if submitted_f:
                user = db_get_user_by_email(email_f)
                if not user:
                    st.error("No account found with that email.")
                else:
                    ok, msg = is_strong_password(new_pw)
                    if not ok:
                        st.error(msg)
                    else:
                        db_update_user(user["id"], password_hash=hash_password(new_pw))
                        st.success("Password reset! You can now log in.")

        glass_card_close()
        if st.button("← Back to Home", use_container_width=True):
            st.session_state["page"] = "landing"; st.rerun()

# ============================================================================
# SECTION 14: SIDEBAR NAVIGATION
# ============================================================================

NAV_ITEMS = [
    ("Dashboard", "speedometer2"), ("New Interview", "chat-dots"),
    ("Resume Analyzer", "file-earmark-text"), ("History", "clock-history"),
    ("Analytics", "bar-chart-line"), ("Profile", "person-circle"),
    ("Settings", "gear"),
]


def render_sidebar() -> str:
    user = st.session_state["user"]
    with st.sidebar:
        st.markdown(f'<h3 class="gradient-text">{APP_NAME}</h3>', unsafe_allow_html=True)
        photo = user.get("photo_b64") or ""
        avatar_html = (
            f'<img src="data:image/png;base64,{photo}" style="width:52px;height:52px;border-radius:50%;object-fit:cover;">'
            if photo else '<div style="width:52px;height:52px;border-radius:50%;background:linear-gradient(135deg,#6C5CE7,#00CEC9);display:flex;align-items:center;justify-content:center;font-weight:800;font-size:1.3rem;">'
                          + user["name"][:1].upper() + '</div>'
        )
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:0.7rem;margin:0.6rem 0 1rem;">
            {avatar_html}
            <div>
                <div style="font-weight:700;">{user['name']}</div>
                <div style="font-size:0.78rem;color:var(--text-secondary);">Level {user['level']} · {user['xp']} XP</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(min(1.0, (user["xp"] % LEVEL_XP_STEP) / LEVEL_XP_STEP))
        st.caption(f"🔥 {user['streak']}-day streak")
        st.markdown("---")

        icons = [i for _, i in NAV_ITEMS]
        labels = [l for l, _ in NAV_ITEMS]
        default_idx = labels.index(st.session_state.get("nav_page", "Dashboard")) if st.session_state.get("nav_page") in labels else 0

        if HAS_OPTION_MENU:
            choice = option_menu(
                None, labels, icons=icons, default_index=default_idx,
                styles={
                    "container": {"padding": "0", "background-color": "transparent"},
                    "icon": {"color": "#a78bfa", "font-size": "16px"},
                    "nav-link": {"font-size": "14px", "text-align": "left", "margin": "3px 0",
                                  "border-radius": "10px", "--hover-color": "#1c1c2e"},
                    "nav-link-selected": {"background-color": "#6C5CE7"},
                },
            )
        else:
            choice = st.radio("Navigate", labels, index=default_idx, label_visibility="collapsed")

        st.markdown("---")
        ai_status = []
        ai_status.append("🟢 Groq" if groq_available() else "⚪ Groq (add key)")
        ai_status.append("🟢 Mistral" if mistral_available() else "⚪ Mistral (add key)")
        ai_status.append("🟢 Gemini" if gemini_available() else "⚪ Gemini (optional)")
        st.caption(" · ".join(ai_status))

        if st.button("🚪 Log Out", use_container_width=True):
            do_logout(); st.rerun()

        st.session_state["nav_page"] = choice
        return choice

# ============================================================================
# SECTION 15: DASHBOARD PAGE
# ============================================================================

def _plotly_dark_layout(fig: go.Figure, title: str = "") -> go.Figure:
    fig.update_layout(
        title=title, template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#F5F5F7"),
        margin=dict(l=30, r=30, t=50 if title else 20, b=30),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def render_dashboard() -> None:
    user = st.session_state["user"]
    interviews = db_get_user_interviews(user["id"])
    reports = db_get_user_reports(user["id"])
    completed = [i for i in interviews if i["status"] == "completed"]

    st.markdown(f'<h2>Welcome back, <span class="gradient-text">{user["name"].split()[0]}</span> 👋</h2>', unsafe_allow_html=True)

    scores = [r["overall_score"] for r in reports if r["overall_score"] is not None]
    avg_score = round(np.mean(scores), 1) if scores else 0.0
    best_score = round(max(scores), 1) if scores else 0.0

    # weakest topic: lowest average sub-score category across all reports
    weakest_topic = "—"
    if reports:
        cat_map = {
            "Technical": "technical_score", "Communication": "communication_score",
            "Confidence": "confidence_score", "Behavior": "behavior_score",
            "Leadership": "leadership_score", "Problem Solving": "problem_solving_score",
            "Grammar": "grammar_score",
        }
        cat_avgs = {label: np.mean([r[key] for r in reports if r[key] is not None] or [0])
                    for label, key in cat_map.items()}
        weakest_topic = min(cat_avgs, key=cat_avgs.get) if cat_avgs else "—"

    cols = st.columns(4)
    stats = [
        ("Total Interviews", str(len(completed)), "🎤"),
        ("Average Score", f"{avg_score}/100", "📊"),
        ("Best Score", f"{best_score}/100", "🏆"),
        ("Weakest Topic", weakest_topic, "🎯"),
    ]
    for col, (label, val, icon) in zip(cols, stats):
        with col:
            st.markdown(stat_card(label, val, icon), unsafe_allow_html=True)

    cols2 = st.columns(3)
    with cols2[0]:
        st.markdown(stat_card("XP", f"{user['xp']}", "⭐"), unsafe_allow_html=True)
    with cols2[1]:
        st.markdown(stat_card("Level", f"{user['level']}", "🚀"), unsafe_allow_html=True)
    with cols2[2]:
        st.markdown(stat_card("Streak", f"{user['streak']} days", "🔥"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if not reports:
        glass_card_open()
        st.markdown("### 🚀 Ready for your first interview?")
        st.write("Start a mock interview to unlock your personalized analytics, radar chart, and skill heatmap.")
        if st.button("Start New Interview →"):
            st.session_state["nav_page"] = "New Interview"; st.rerun()
        glass_card_close()
        return

    c1, c2 = st.columns([1, 1])
    with c1:
        glass_card_open()
        st.markdown("#### Skill Radar (Latest Report)")
        latest = reports[0]
        categories = ["Technical", "Communication", "Confidence", "Behavior", "Leadership", "Problem Solving", "Grammar"]
        values = [latest["technical_score"], latest["communication_score"], latest["confidence_score"],
                  latest["behavior_score"], latest["leadership_score"], latest["problem_solving_score"],
                  latest["grammar_score"]]
        fig = go.Figure(data=go.Scatterpolar(
            r=values + [values[0]], theta=categories + [categories[0]], fill="toself",
            line_color="#6C5CE7", fillcolor="rgba(108,92,231,0.35)",
        ))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100], gridcolor="#333"),
                                       bgcolor="rgba(0,0,0,0)"), showlegend=False)
        fig = _plotly_dark_layout(fig)
        st.plotly_chart(fig, use_container_width=True)
        glass_card_close()

    with c2:
        glass_card_open()
        st.markdown("#### Score Trend")
        df = pd.DataFrame({
            "Date": [r["created_at"][:10] for r in reversed(reports)],
            "Score": [r["overall_score"] for r in reversed(reports)],
        })
        fig2 = px.line(df, x="Date", y="Score", markers=True)
        fig2.update_traces(line_color="#00CEC9", marker=dict(size=8, color="#6C5CE7"))
        fig2.update_yaxes(range=[0, 100])
        fig2 = _plotly_dark_layout(fig2)
        st.plotly_chart(fig2, use_container_width=True)
        glass_card_close()

    c3, c4 = st.columns([1, 1])
    with c3:
        glass_card_open()
        st.markdown("#### Interviews by Track")
        track_counts = pd.Series([i["track"] for i in completed]).value_counts()
        if len(track_counts):
            fig3 = px.pie(values=track_counts.values, names=track_counts.index, hole=0.5,
                            color_discrete_sequence=px.colors.sequential.Purples_r)
            fig3 = _plotly_dark_layout(fig3)
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.caption("No completed interviews yet.")
        glass_card_close()

    with c4:
        glass_card_open()
        st.markdown("#### Skill Heatmap (Recent Reports)")
        recent = reports[:8]
        if recent:
            heat_df = pd.DataFrame([
                {"Report": r["created_at"][:10], "Technical": r["technical_score"],
                 "Communication": r["communication_score"], "Confidence": r["confidence_score"],
                 "Behavior": r["behavior_score"], "Leadership": r["leadership_score"],
                 "Problem Solving": r["problem_solving_score"], "Grammar": r["grammar_score"]}
                for r in reversed(recent)
            ]).set_index("Report")
            fig4 = px.imshow(heat_df.T, color_continuous_scale="Purples", aspect="auto")
            fig4 = _plotly_dark_layout(fig4)
            st.plotly_chart(fig4, use_container_width=True)
        glass_card_close()

    glass_card_open()
    st.markdown("#### Recent Interviews")
    for i in completed[:5]:
        rep = db_get_report_for_interview(i["id"])
        score_txt = f"{round(rep['overall_score'],1)}/100" if rep else "Not evaluated"
        rc1, rc2, rc3 = st.columns([3, 1.2, 1])
        with rc1:
            st.write(f"**{i['track']}** {('· ' + i['company']) if i['company'] and i['company']!='General / No Company' else ''} — {i['persona']}")
            st.caption(f"{i['started_at'][:16].replace('T',' ')} · {i['difficulty']} · {i['mode']}")
        with rc2:
            st.write(score_txt)
        with rc3:
            if st.button("View Report", key=f"dash_view_{i['id']}"):
                st.session_state["viewing_report_interview_id"] = i["id"]
                st.session_state["nav_page"] = "History"; st.rerun()
    glass_card_close()

    achievements = db_get_user_achievements(user["id"])
    if achievements:
        glass_card_open()
        st.markdown("#### 🏅 Achievements")
        badge_html = "".join(f'<span class="badge badge-accent" title="{a["description"]}">🏅 {a["title"]}</span> ' for a in achievements)
        st.markdown(badge_html, unsafe_allow_html=True)
        glass_card_close()

# ============================================================================
# SECTION 16: NEW INTERVIEW PAGE (setup + live chat + coding mode)
# ============================================================================

def render_interview_setup() -> None:
    glass_card_open()
    st.markdown("### 🎬 Configure Your Interview")
    c1, c2 = st.columns(2)
    with c1:
        track = st.selectbox("Interview Track", INTERVIEW_TRACKS, key="setup_track")
        company = st.selectbox("Company (optional)", ["General / No Company"] + list(COMPANIES.keys()), key="setup_company")
        persona = st.selectbox("Interviewer Persona", list(PERSONAS.keys()), key="setup_persona")
    with c2:
        difficulty = st.selectbox("Difficulty", DIFFICULTY_LEVELS, index=1, key="setup_difficulty")
        mode = st.selectbox("Interview Mode", INTERVIEW_MODES, key="setup_mode")
        language = st.selectbox("Coding Language (if applicable)", CODE_LANGUAGES, key="setup_language")

    if company != "General / No Company":
        cp = COMPANIES[company]
        st.caption(f"🏢 **{cp.name}** focuses on {cp.focus}. Interviewer style: {cp.style}.")

    if not groq_available():
        st.warning("⚠️ Groq isn't configured — this interview will run in **demo mode** with placeholder "
                    "responses. Add a Groq API key in Settings for a real, live AI interview.")

    if st.button("🚀 Start Interview", use_container_width=True):
        user = st.session_state["user"]
        iid = db_create_interview(user["id"], track, company, persona, mode, difficulty, language)
        st.session_state["active_interview_id"] = iid
        st.session_state["interview_config"] = {
            "track": track, "company": company, "persona": persona,
            "mode": mode, "difficulty": difficulty, "language": language,
        }
        st.session_state["interview_system_prompt"] = build_system_prompt(track, company, persona, mode, difficulty)
        st.session_state["interview_started_at"] = time.time()
        opening = get_opening_message(track, company, persona, mode, difficulty)
        st.session_state["interview_transcript"] = [{"role": "interviewer", "content": opening}]
        db_update_interview_transcript(iid, st.session_state["interview_transcript"])
        st.session_state["coding_question"] = None
        st.session_state["coding_review"] = None
        st.rerun()
    glass_card_close()


def render_chat_interview() -> None:
    cfg = st.session_state["interview_config"]
    st.markdown(f"#### 💬 {cfg['track']} Interview " +
                (f"· {cfg['company']}" if cfg['company'] != "General / No Company" else "") +
                f" · {cfg['persona']}")
    top_c1, top_c2, top_c3 = st.columns([2, 1, 1])
    with top_c2:
        elapsed = int(time.time() - st.session_state["interview_started_at"])
        st.caption(f"⏱️ {elapsed // 60}m {elapsed % 60}s elapsed")
    with top_c3:
        if st.button("🏁 End & Evaluate", use_container_width=True):
            _end_interview_and_evaluate()
            return

    is_coding = cfg["track"] in ("Coding", "System Design")
    chat_container = st.container(height=440)
    with chat_container:
        for turn in st.session_state["interview_transcript"]:
            if turn["role"] == "interviewer":
                st.markdown(f'<div class="chat-label">🧑‍💼 {cfg["persona"]}</div>'
                             f'<div class="chat-bubble-interviewer">{turn["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-label" style="text-align:right;">You</div>'
                             f'<div class="chat-bubble-candidate">{turn["content"]}</div>', unsafe_allow_html=True)

    if is_coding:
        _render_coding_panel(cfg)

    user_input = st.chat_input("Type your answer...")
    if user_input:
        st.session_state["interview_transcript"].append({"role": "candidate", "content": user_input})
        with chat_container:
            st.markdown(f'<div class="chat-label" style="text-align:right;">You</div>'
                         f'<div class="chat-bubble-candidate">{user_input}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-label">🧑‍💼 {cfg["persona"]}</div>', unsafe_allow_html=True)
            placeholder = st.empty()
            full_reply = ""
            for chunk in stream_interviewer_reply(st.session_state["interview_system_prompt"],
                                                     st.session_state["interview_transcript"]):
                full_reply += chunk
                placeholder.markdown(f'<div class="chat-bubble-interviewer">{full_reply}▌</div>', unsafe_allow_html=True)
            placeholder.markdown(f'<div class="chat-bubble-interviewer">{full_reply}</div>', unsafe_allow_html=True)
        st.session_state["interview_transcript"].append({"role": "interviewer", "content": full_reply})
        db_update_interview_transcript(st.session_state["active_interview_id"], st.session_state["interview_transcript"])
        st.rerun()


def _render_coding_panel(cfg: dict) -> None:
    with st.expander("💻 Coding Workspace", expanded=True):
        if st.session_state["coding_question"] is None:
            if st.button("🎲 Generate Coding Question"):
                st.session_state["coding_question"] = generate_coding_question(
                    cfg["track"], cfg["difficulty"], cfg["language"])
                st.rerun()
        else:
            q = st.session_state["coding_question"]
            st.markdown(f"**{q['title']}**")
            st.write(q["prompt"])
            code = st.text_area("Your Code", value=q.get("starter_code", ""), height=260,
                                  key="coding_editor", label_visibility="collapsed")
            cc1, cc2 = st.columns([1, 1])
            with cc1:
                if st.button("🔍 Get AI Code Review"):
                    with st.spinner("Reviewing your code..."):
                        st.session_state["coding_review"] = review_code(q, code, cfg["language"])
            with cc2:
                if st.button("🎲 New Question"):
                    st.session_state["coding_question"] = generate_coding_question(
                        cfg["track"], cfg["difficulty"], cfg["language"])
                    st.session_state["coding_review"] = None
                    st.rerun()
            if st.session_state.get("coding_review"):
                st.markdown("##### AI Code Review")
                st.markdown(st.session_state["coding_review"])


def _end_interview_and_evaluate() -> None:
    iid = st.session_state["active_interview_id"]
    cfg = st.session_state["interview_config"]
    duration = int(time.time() - st.session_state["interview_started_at"])
    db_end_interview(iid, duration)
    with st.spinner("🧠 Running multi-agent evaluation panel..."):
        evaluation = run_evaluation_agents(
            st.session_state["interview_transcript"], cfg["track"], cfg["company"], cfg["difficulty"])
    db_save_report(iid, st.session_state["user"]["id"], evaluation)
    db_add_xp(st.session_state["user"]["id"], XP_PER_INTERVIEW +
              (XP_PER_HIGH_SCORE if evaluation.get("overall_score", 0) >= 80 else 0))
    interviews_count = len(db_get_user_interviews(st.session_state["user"]["id"]))
    if interviews_count == 1:
        db_grant_achievement(st.session_state["user"]["id"], "First Interview", "Completed your first mock interview.")
    if evaluation.get("overall_score", 0) >= 90:
        db_grant_achievement(st.session_state["user"]["id"], "Top Scorer", "Scored 90+ on an interview.")
    refresh_user()
    st.session_state["active_interview_id"] = None
    st.session_state["viewing_report_interview_id"] = iid
    st.session_state["nav_page"] = "History"
    st.success("Interview evaluated! View your full report in History.")
    time.sleep(1)
    st.rerun()


def render_new_interview_page() -> None:
    st.markdown('<h2 class="gradient-text">New Interview</h2>', unsafe_allow_html=True)
    if st.session_state.get("active_interview_id"):
        render_chat_interview()
    else:
        render_interview_setup()

# ============================================================================
# SECTION 17: RESUME ANALYZER PAGE
# ============================================================================

def render_resume_analyzer_page() -> None:
    st.markdown('<h2 class="gradient-text">Resume Analyzer</h2>', unsafe_allow_html=True)
    glass_card_open()
    target_role = st.text_input("Target Role (optional)", placeholder="e.g. Backend Engineer")
    uploaded = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
    if uploaded and st.button("🔍 Analyze Resume"):
        file_bytes = uploaded.read()
        save_path = os.path.join(UPLOAD_DIR, f"{st.session_state['user']['id']}_{uploaded.name}")
        with open(save_path, "wb") as f:
            f.write(file_bytes)
        with st.spinner("Extracting and analyzing..."):
            text = extract_text_from_pdf(file_bytes)
            if not text.strip():
                st.error("Couldn't extract text from this PDF. Try a text-based (non-scanned) PDF.")
            else:
                analysis = analyze_resume(text, target_role)
                db_save_resume_analysis(st.session_state["user"]["id"], uploaded.name, analysis, text)
                st.session_state["resume_last_analysis"] = analysis
    glass_card_close()

    analysis = st.session_state.get("resume_last_analysis")
    if not analysis:
        past = db_get_user_resume_analyses(st.session_state["user"]["id"])
        if past:
            latest = past[0]
            analysis = {
                "ats_score": latest["ats_score"],
                "skills": json.loads(latest["skills_json"]),
                "missing_skills": json.loads(latest["missing_skills_json"]),
                "suggestions": json.loads(latest["suggestions_json"]),
            }

    if analysis:
        c1, c2 = st.columns([1, 2])
        with c1:
            glass_card_open()
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=analysis["ats_score"],
                gauge={"axis": {"range": [0, 100]},
                        "bar": {"color": "#6C5CE7"},
                        "steps": [{"range": [0, 50], "color": "rgba(255,107,107,0.25)"},
                                   {"range": [50, 75], "color": "rgba(253,203,110,0.25)"},
                                   {"range": [75, 100], "color": "rgba(0,227,150,0.25)"}]},
                title={"text": "ATS Score"},
            ))
            fig = _plotly_dark_layout(fig)
            st.plotly_chart(fig, use_container_width=True)
            glass_card_close()
        with c2:
            glass_card_open()
            st.markdown("##### Detected Skills")
            if analysis["skills"]:
                st.markdown(" ".join(badge(s, "success") for s in analysis["skills"]), unsafe_allow_html=True)
            else:
                st.caption("No common skills detected — consider adding a clear Skills section.")
            if analysis.get("missing_skills"):
                st.markdown("##### Suggested Missing Skills")
                st.markdown(" ".join(badge(s, "warning") for s in analysis["missing_skills"]), unsafe_allow_html=True)
            glass_card_close()

        glass_card_open()
        st.markdown("##### 💡 Suggestions to Improve")
        for s in analysis["suggestions"]:
            st.markdown(f"- {s}")
        glass_card_close()

# ============================================================================
# SECTION 18: HISTORY PAGE (list + full report viewer + export)
# ============================================================================

def render_report_detail(interview_row: sqlite3.Row) -> None:
    report = db_get_report_for_interview(interview_row["id"])
    if not report:
        st.info("This interview has no report yet.")
        return
    evaluation = json.loads(report["raw_json"])

    if evaluation.get("_offline_estimate"):
        st.info("This score is a heuristic offline estimate. Connect Mistral in Settings for a full AI evaluation.")

    rec = evaluation.get("hiring_recommendation", "Hold")
    rec_kind = {"Strong Hire": "success", "Hire": "success", "Hold": "warning", "Reject": "danger"}.get(rec, "accent")

    glass_card_open()
    hc1, hc2 = st.columns([3, 1])
    with hc1:
        st.markdown(f"### {interview_row['track']} " +
                     (f"· {interview_row['company']}" if interview_row['company'] and interview_row['company'] != 'General / No Company' else ""))
        st.caption(f"{interview_row['persona']} · {interview_row['difficulty']} · {interview_row['mode']} · "
                    f"{interview_row['started_at'][:16].replace('T',' ')}")
    with hc2:
        st.markdown(badge(rec, rec_kind), unsafe_allow_html=True)
        st.markdown(f"### {round(evaluation.get('overall_score',0),1)}/100")
    glass_card_close()

    metrics = [
        ("Technical", "technical_score"), ("Communication", "communication_score"),
        ("Confidence", "confidence_score"), ("Behavior", "behavior_score"),
        ("Leadership", "leadership_score"), ("Problem Solving", "problem_solving_score"),
        ("Grammar", "grammar_score"),
    ]
    mcols = st.columns(4)
    for i, (label, key) in enumerate(metrics):
        with mcols[i % 4]:
            st.metric(label, f"{round(evaluation.get(key,0),1)}")

    c1, c2 = st.columns(2)
    with c1:
        glass_card_open()
        st.markdown("##### ✅ Strengths")
        for s in evaluation.get("strengths", []):
            st.markdown(f"- {s}")
        glass_card_close()
    with c2:
        glass_card_open()
        st.markdown("##### ⚠️ Weaknesses")
        for w in evaluation.get("weaknesses", []):
            st.markdown(f"- {w}")
        glass_card_close()

    roadmap = evaluation.get("roadmap", {})
    if roadmap:
        glass_card_open()
        st.markdown("##### 🗺️ Learning Roadmap")
        rc = st.columns(4)
        for col, (label, key) in zip(rc, [("Courses", "courses"), ("Books", "books"),
                                             ("LeetCode", "leetcode"), ("Projects", "projects")]):
            with col:
                st.markdown(f"**{label}**")
                for item in roadmap.get(key, []):
                    st.markdown(f"- {item}")
        glass_card_close()

    with st.expander("📜 Full Transcript"):
        transcript = json.loads(interview_row["transcript_json"])
        for turn in transcript:
            speaker = "🧑‍💼 Interviewer" if turn["role"] == "interviewer" else "🙋 You"
            st.markdown(f"**{speaker}:** {turn['content']}")

    st.markdown("##### 📥 Export Report")
    ec1, ec2, ec3 = st.columns(3)
    user_name = st.session_state["user"]["name"]
    with ec1:
        pdf_bytes = generate_report_pdf(user_name, interview_row, evaluation)
        if pdf_bytes:
            st.download_button("Download PDF", pdf_bytes, file_name=f"report_{interview_row['id'][:8]}.pdf",
                                 mime="application/pdf", use_container_width=True)
        else:
            st.caption("ReportLab not installed — PDF export unavailable.")
    with ec2:
        md = generate_report_markdown(user_name, interview_row, evaluation)
        st.download_button("Download Markdown", md, file_name=f"report_{interview_row['id'][:8]}.md",
                             use_container_width=True)
    with ec3:
        html = generate_report_html(user_name, interview_row, evaluation)
        st.download_button("Download HTML", html, file_name=f"report_{interview_row['id'][:8]}.html",
                             mime="text/html", use_container_width=True)


def render_history_page() -> None:
    st.markdown('<h2 class="gradient-text">Interview History</h2>', unsafe_allow_html=True)
    interviews = db_get_user_interviews(st.session_state["user"]["id"])

    viewing_id = st.session_state.get("viewing_report_interview_id")
    if viewing_id:
        row = db_get_interview(viewing_id)
        if row:
            if st.button("← Back to History List"):
                st.session_state["viewing_report_interview_id"] = None; st.rerun()
            render_report_detail(row)
            return

    if not interviews:
        st.info("No interviews yet. Start one from the New Interview tab!")
        return

    f1, f2, f3 = st.columns(3)
    with f1:
        track_filter = st.selectbox("Filter by Track", ["All"] + INTERVIEW_TRACKS)
    with f2:
        company_filter = st.selectbox("Filter by Company", ["All"] + list(COMPANIES.keys()))
    with f3:
        status_filter = st.selectbox("Filter by Status", ["All", "completed", "in_progress"])

    for i in interviews:
        if track_filter != "All" and i["track"] != track_filter:
            continue
        if company_filter != "All" and i["company"] != company_filter:
            continue
        if status_filter != "All" and i["status"] != status_filter:
            continue
        glass_card_open()
        rc1, rc2, rc3, rc4 = st.columns([2.5, 1, 1, 1])
        with rc1:
            st.write(f"**{i['track']}**" + (f" · {i['company']}" if i['company'] and i['company'] != 'General / No Company' else ""))
            st.caption(f"{i['persona']} · {i['difficulty']} · {i['started_at'][:16].replace('T',' ')}")
        with rc2:
            st.markdown(badge(i["status"].replace("_", " ").title(),
                                "success" if i["status"] == "completed" else "warning"), unsafe_allow_html=True)
        with rc3:
            rep = db_get_report_for_interview(i["id"])
            st.write(f"{round(rep['overall_score'],1)}/100" if rep else "—")
        with rc4:
            if st.button("View", key=f"hist_view_{i['id']}"):
                st.session_state["viewing_report_interview_id"] = i["id"]; st.rerun()
        glass_card_close()

# ============================================================================
# SECTION 19: ANALYTICS PAGE
# ============================================================================

def render_analytics_page() -> None:
    st.markdown('<h2 class="gradient-text">Analytics</h2>', unsafe_allow_html=True)
    reports = db_get_user_reports(st.session_state["user"]["id"])
    if not reports:
        st.info("Complete an interview to unlock analytics.")
        return

    df = pd.DataFrame([{
        "Date": r["created_at"][:10], "Overall": r["overall_score"],
        "Technical": r["technical_score"], "Communication": r["communication_score"],
        "Confidence": r["confidence_score"], "Behavior": r["behavior_score"],
        "Leadership": r["leadership_score"], "Problem Solving": r["problem_solving_score"],
        "Grammar": r["grammar_score"],
    } for r in reversed(reports)])

    period = st.radio("Period", ["Weekly", "Monthly", "Yearly"], horizontal=True)
    df["DateP"] = pd.to_datetime(df["Date"])
    freq = {"Weekly": "W", "Monthly": "M", "Yearly": "Y"}[period]
    grouped = df.set_index("DateP").resample(freq)["Overall"].mean().reset_index()

    c1, c2 = st.columns(2)
    with c1:
        glass_card_open()
        st.markdown(f"#### {period} Trend")
        fig = px.bar(grouped, x="DateP", y="Overall")
        fig.update_traces(marker_color="#6C5CE7")
        fig = _plotly_dark_layout(fig)
        st.plotly_chart(fig, use_container_width=True)
        glass_card_close()
    with c2:
        glass_card_open()
        st.markdown("#### Category Averages")
        cat_avg = df[["Technical", "Communication", "Confidence", "Behavior",
                        "Leadership", "Problem Solving", "Grammar"]].mean().reset_index()
        cat_avg.columns = ["Category", "Average"]
        fig2 = px.bar(cat_avg, x="Average", y="Category", orientation="h", color="Average",
                       color_continuous_scale="Purples", range_x=[0, 100])
        fig2 = _plotly_dark_layout(fig2)
        st.plotly_chart(fig2, use_container_width=True)
        glass_card_close()

    glass_card_open()
    st.markdown("#### Full Timeline")
    fig3 = px.line(df, x="Date", y=["Overall", "Technical", "Communication", "Problem Solving"], markers=True)
    fig3 = _plotly_dark_layout(fig3)
    st.plotly_chart(fig3, use_container_width=True)
    glass_card_close()

    glass_card_open()
    st.markdown("#### Score Distribution Heatmap")
    heat = df.set_index("Date")[["Technical", "Communication", "Confidence", "Behavior",
                                    "Leadership", "Problem Solving", "Grammar"]]
    fig4 = px.imshow(heat.T, color_continuous_scale="Purples", aspect="auto")
    fig4 = _plotly_dark_layout(fig4)
    st.plotly_chart(fig4, use_container_width=True)
    glass_card_close()


# ============================================================================
# SECTION 20: PROFILE PAGE
# ============================================================================

def render_profile_page() -> None:
    st.markdown('<h2 class="gradient-text">Profile</h2>', unsafe_allow_html=True)
    user = st.session_state["user"]

    c1, c2 = st.columns([1, 2])
    with c1:
        glass_card_open()
        photo_file = st.file_uploader("Profile Photo", type=["png", "jpg", "jpeg"])
        if photo_file:
            b64 = base64.b64encode(photo_file.read()).decode("utf-8")
            db_update_user(user["id"], photo_b64=b64)
            refresh_user()
            st.rerun()
        if user.get("photo_b64"):
            st.markdown(f'<img src="data:image/png;base64,{user["photo_b64"]}" style="width:140px;height:140px;border-radius:50%;object-fit:cover;">', unsafe_allow_html=True)
        st.markdown(f"### {user['name']}")
        st.caption(user["email"])
        st.markdown(f"⭐ Level {user['level']} · {user['xp']} XP · 🔥 {user['streak']}-day streak")
        glass_card_close()

    with c2:
        glass_card_open()
        st.markdown("##### Edit Profile")
        with st.form("profile_form"):
            name = st.text_input("Full Name", value=user["name"])
            college = st.text_input("College / University", value=user.get("college", ""))
            github = st.text_input("GitHub URL", value=user.get("github", ""))
            linkedin = st.text_input("LinkedIn URL", value=user.get("linkedin", ""))
            saved = st.form_submit_button("Save Changes")
        if saved:
            db_update_user(user["id"], name=name, college=college, github=github, linkedin=linkedin)
            refresh_user()
            st.success("Profile updated.")
            st.rerun()
        glass_card_close()

    achievements = db_get_user_achievements(user["id"])
    glass_card_open()
    st.markdown("##### 🏅 Achievements")
    if achievements:
        for a in achievements:
            st.markdown(f"**🏅 {a['title']}** — {a['description']}  \n"
                         f"<span style='color:var(--text-secondary);font-size:0.8rem;'>{a['earned_at'][:10]}</span>",
                         unsafe_allow_html=True)
    else:
        st.caption("No achievements yet — complete interviews to earn badges!")
    glass_card_close()


# ============================================================================
# SECTION 21: SETTINGS PAGE
# ============================================================================

def render_settings_page() -> None:
    st.markdown('<h2 class="gradient-text">Settings</h2>', unsafe_allow_html=True)
    user = st.session_state["user"]

    glass_card_open()
    st.markdown("##### 🤖 AI Provider Keys")
    st.caption("Keys are stored locally in your account record for this app instance. "
                "Groq conducts interviews; Mistral evaluates and scores. Gemini is optional and never required.")
    with st.form("api_keys_form"):
        groq_key = st.text_input("Groq API Key", value=user.get("groq_key", ""), type="password")
        mistral_key = st.text_input("Mistral API Key", value=user.get("mistral_key", ""), type="password")
        gemini_key = st.text_input("Gemini API Key (optional)", value=user.get("gemini_key", ""), type="password")
        saved_keys = st.form_submit_button("Save API Keys")
    if saved_keys:
        db_update_user(user["id"], groq_key=groq_key, mistral_key=mistral_key, gemini_key=gemini_key)
        refresh_user()
        st.success("API keys updated.")
        st.rerun()

    if not user.get("gemini_key") and not os.getenv("GEMINI_API_KEY"):
        st.info("Gemini integration is optional. Add your API key to enable Gemini features.")
    glass_card_close()

    glass_card_open()
    st.markdown("##### 🎨 Preferences")
    with st.form("prefs_form"):
        pcol1, pcol2 = st.columns(2)
        with pcol1:
            theme = st.selectbox("Theme", ["dark"], index=0, help="Light theme coming soon — dark is the flagship experience.")
            font_size = st.select_slider("Font Size", ["Small", "Medium", "Large"],
                                            value=user.get("font_size", "Medium"))
        with pcol2:
            language = st.selectbox("Language", ["English", "Hindi", "Spanish", "French"],
                                       index=["English", "Hindi", "Spanish", "French"].index(user.get("language", "English")) if user.get("language", "English") in ["English","Hindi","Spanish","French"] else 0)
            notifications = st.toggle("Notifications", value=bool(user.get("notifications", 1)))
        saved_prefs = st.form_submit_button("Save Preferences")
    if saved_prefs:
        db_update_user(user["id"], theme=theme, font_size=font_size, language=language,
                         notifications=int(notifications))
        refresh_user()
        st.success("Preferences saved.")
    glass_card_close()

    glass_card_open()
    st.markdown("##### 🧩 System Status")
    status_rows = [
        ("Groq (Interview Engine)", groq_available()),
        ("Mistral (Evaluation Engine)", mistral_available()),
        ("Gemini (Optional)", gemini_available()),
        ("PDF Export (ReportLab)", HAS_REPORTLAB),
        ("Resume Parsing (pdfplumber/PyPDF2)", HAS_PDFPLUMBER or HAS_PYPDF2),
        ("Password Hashing (bcrypt)", HAS_BCRYPT),
        ("Session Tokens (JWT)", HAS_JWT),
        ("Sidebar Menu (streamlit-option-menu)", HAS_OPTION_MENU),
        ("Voice Input (SpeechRecognition)", HAS_SPEECH_RECOGNITION),
        ("Voice Output (pyttsx3)", HAS_PYTTSX3),
        ("Webcam (OpenCV)", HAS_OPENCV),
    ]
    for label, ok in status_rows:
        st.markdown(f"{'🟢' if ok else '⚪'} {label}")
    glass_card_close()

    glass_card_open()
    st.markdown("##### ⚠️ Danger Zone")
    if st.button("Log Out", use_container_width=True):
        do_logout(); st.rerun()
    glass_card_close()

# ============================================================================
# SECTION 22: MAIN APP ROUTER
# ============================================================================

def main() -> None:
    st.set_page_config(
        page_title=APP_NAME, page_icon="🧠", layout="wide",
        initial_sidebar_state="collapsed",
    )
    init_db()
    init_session_state()
    inject_css()

    if not st.session_state.get("user"):
        st.session_state["page"] = st.session_state.get("page") or "landing"
        if st.session_state["page"] == "auth":
            render_auth_page()
        else:
            render_landing_page()
        return

    # Authenticated area
    st.session_state.setdefault("nav_page", "Dashboard")
    choice = render_sidebar()

    router = {
        "Dashboard": render_dashboard,
        "New Interview": render_new_interview_page,
        "Resume Analyzer": render_resume_analyzer_page,
        "History": render_history_page,
        "Analytics": render_analytics_page,
        "Profile": render_profile_page,
        "Settings": render_settings_page,
    }
    page_fn = router.get(choice, render_dashboard)
    page_fn()


if __name__ == "__main__":
    main()
