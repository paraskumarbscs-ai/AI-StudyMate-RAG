import os
import re
import json
import uuid
import sqlite3
import hashlib
import hmac
import secrets
from datetime import date

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

load_dotenv()


# ---------------- SUPABASE AUTH / CLOUD STORAGE ----------------
# User accounts are managed by Supabase Auth. Study data is saved as one JSON
# payload per authenticated user, protected by Supabase Row Level Security.
# Uploaded PDF text/chunks/embeddings are deliberately NOT persisted to cloud.

from supabase import create_client


def get_config(name, default=""):
    try:
        value = st.secrets.get(name, default)
        if value:
            return str(value).strip()
    except Exception:
        pass
    return os.getenv(name, default).strip()


SUPABASE_URL = get_config("SUPABASE_URL")
SUPABASE_KEY = get_config("SUPABASE_ANON_KEY") or get_config("SUPABASE_PUBLISHABLE_KEY")
APP_URL = get_config("APP_URL", "http://localhost:8501").rstrip("/")


def get_supabase():
    """Keep one Supabase client per Streamlit browser session.

    The same client must survive Streamlit reruns so the PKCE verifier created
    during Google sign-in is still available when Supabase redirects back.
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError(
            "Supabase configuration missing. Set SUPABASE_URL and "
            "SUPABASE_ANON_KEY in your .env file."
        )

    if "_sb_client" not in st.session_state:
        st.session_state["_sb_client"] = create_client(
            SUPABASE_URL,
            SUPABASE_KEY
        )
    return st.session_state["_sb_client"]


try:
    supabase = get_supabase() if SUPABASE_URL and SUPABASE_KEY else None
except Exception:
    supabase = None


def get_session_user():
    """Restore the Supabase session from this Streamlit browser session."""
    access_token = st.session_state.get("sb_access_token")
    refresh_token = st.session_state.get("sb_refresh_token")
    if not supabase or not access_token or not refresh_token:
        return None
    try:
        result = supabase.auth.set_session(access_token, refresh_token)
        session = getattr(result, "session", None)
        user = getattr(result, "user", None)
        if session:
            st.session_state.sb_access_token = session.access_token
            st.session_state.sb_refresh_token = session.refresh_token
        if user:
            st.session_state.auth_user_id = str(user.id)
            st.session_state.auth_email = user.email or ""
            return user
    except Exception:
        st.session_state.pop("sb_access_token", None)
        st.session_state.pop("sb_refresh_token", None)
        st.session_state.pop("auth_user_id", None)
        st.session_state.pop("auth_email", None)
    return None


def accept_auth_session(session):
    if not session or not getattr(session, "user", None):
        return False
    st.session_state.sb_access_token = session.access_token
    st.session_state.sb_refresh_token = session.refresh_token
    st.session_state.auth_user_id = str(session.user.id)
    st.session_state.auth_email = session.user.email or ""
    st.session_state.account_data_loaded = False
    return True


def handle_oauth_callback():
    """Exchange the Google OAuth code using the client that started PKCE."""
    code = st.query_params.get("code")
    if not code or not supabase:
        return

    try:
        response = supabase.auth.exchange_code_for_session({"auth_code": code})
        if accept_auth_session(getattr(response, "session", None)):
            st.session_state.pop("google_oauth_url", None)
            st.query_params.clear()
            st.rerun()
            return

        st.error("Google sign-in did not return a valid session. Please start again.")
    except Exception as error:
        st.error(f"Google sign-in failed: {error}")
        st.caption("The sign-in code may have expired or its PKCE verifier is missing. Start a fresh Google sign-in.")
        st.query_params.clear()
        st.session_state.pop("google_oauth_url", None)


def load_user_payload(user_id):
    if not supabase or not user_id:
        return {}
    try:
        result = supabase.table("user_study_data").select("payload").eq("user_id", user_id).maybe_single().execute()
        row = result.data
        if not row:
            return {}
        payload = row.get("payload", {})
        return payload if isinstance(payload, dict) else json.loads(payload)
    except Exception as error:
        st.warning(f"Could not load your cloud study data: {error}")
        return {}


def make_cloud_safe_payload(payload):
    """Strip temporary PDF text, embeddings and source excerpts before cloud save."""
    safe = dict(payload)
    safe_chats = {}
    for chat_id, chat in (payload.get("chats") or {}).items():
        clean_chat = {
            "title": chat.get("title", "New chat"),
            "messages": [],
            "chunks": [],
            "embeddings": None,
            "filename": None,
        }
        for message in chat.get("messages", []):
            clean_message = {
                "role": message.get("role", "assistant"),
                "content": message.get("content", ""),
            }
            # Do not persist PDF excerpt text attached to source references.
            clean_chat["messages"].append(clean_message)
        safe_chats[chat_id] = clean_chat
    safe["chats"] = safe_chats
    return safe


def save_user_payload(user_id, payload):
    if not supabase or not user_id:
        raise RuntimeError("You are not signed in to Supabase.")
    safe_payload = make_cloud_safe_payload(payload)
    supabase.table("user_study_data").upsert({
        "user_id": str(user_id),
        "payload": safe_payload,
        "updated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    }, on_conflict="user_id").execute()


def show_auth_screen():
    st.markdown("""
    <style>
    .block-container {max-width: 520px; padding-top: 3rem !important;}
    </style>
    """, unsafe_allow_html=True)
    st.markdown("# 🎓 AI StudyMate")
    st.markdown("### Your personal AI learning space")
    st.write("Sign in to access your saved chats, study documents, quizzes, and planner.")

    if not supabase:
        st.error("Supabase configuration is missing. Check SUPABASE_URL and SUPABASE_ANON_KEY in .env.")
        st.code("SUPABASE_URL=https://YOUR_PROJECT_REF.supabase.co\nSUPABASE_ANON_KEY=YOUR_PUBLISHABLE_OR_ANON_KEY")
        return
        
    if st.button(
        "Continue with Google",
        key="start_google_oauth",
        use_container_width=True
    ):
        try:
            oauth = supabase.auth.sign_in_with_oauth({
                "provider": "google",
                "options": {
                    "redirect_to": APP_URL
                }
            })

            oauth_url = getattr(oauth, "url", None)

            if oauth_url:
                st.markdown(
                    f'<meta http-equiv="refresh" content="0;url={oauth_url}">',
                    unsafe_allow_html=True
                )
                st.link_button(
                    "Continue to Google",
                    oauth_url,
                    use_container_width=True
                )
            else:
                st.error("Supabase did not return a Google sign-in URL.")

        except Exception as error:
            st.error(f"Could not start Google sign-in: {error}")

    login_tab, signup_tab = st.tabs(["Log in", "Sign up"])

    with login_tab:
        with st.form("login_form"):
            email = st.text_input("Email address", placeholder="you@example.com", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Log in", use_container_width=True)
        if submitted:
            try:
                response = supabase.auth.sign_in_with_password({"email": email.strip().lower(), "password": password})
                if accept_auth_session(getattr(response, "session", None)):
                    st.rerun()
                st.error("Login needs email confirmation. Check your inbox and confirm your email first.")
            except Exception as error:
                st.error(f"Login failed: {error}")

    with signup_tab:
        with st.form("signup_form"):
            email = st.text_input("Email address", placeholder="you@example.com", key="signup_email")
            password = st.text_input("Create password", type="password", key="signup_password")
            confirm = st.text_input("Confirm password", type="password", key="signup_confirm")
            st.caption("Use at least 8 characters. A confirmation email will be sent if email confirmation is enabled.")
            submitted = st.form_submit_button("Create account", use_container_width=True)
        if submitted:
            email = email.strip().lower()
            if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
                st.error("Enter a valid email address.")
            elif len(password) < 8:
                st.error("Password must be at least 8 characters long.")
            elif password != confirm:
                st.error("Passwords do not match.")
            else:
                try:
                    response = supabase.auth.sign_up({
                        "email": email,
                        "password": password,
                        "options": {"email_redirect_to": APP_URL}
                    })
                    session = getattr(response, "session", None)
                    if accept_auth_session(session):
                        st.success("Account created. Opening your study space…")
                        st.rerun()
                    else:
                        st.success("Account created! Check your email and click the confirmation link, then return here to log in.")
                except Exception as error:
                    st.error(f"Signup failed: {error}")


st.set_page_config(
    page_title="AI StudyMate",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Handle Google OAuth callback before checking whether the user is signed in.
if "sb_access_token" not in st.session_state:
    st.session_state.sb_access_token = None
if "sb_refresh_token" not in st.session_state:
    st.session_state.sb_refresh_token = None
if "account_data_loaded" not in st.session_state:
    st.session_state.account_data_loaded = False

handle_oauth_callback()
current_user = get_session_user()
if current_user is None:
    st.session_state.auth_user_id = None
    show_auth_screen()
    st.stop()

# ---------------- SESSION STATE ----------------

defaults = {
    "dark_mode": True,
    "chats": {},
    "active_chat": None,
    "tasks": [],
    "quiz_history": [],
    "generated_summary": "",
    "generated_notes": "",
    "generated_quiz": "",
    "quiz_data": [],
    "quiz_version": 0,
    "show_pdf_uploader": False,
    "generated_flashcards": "",
    "generated_questions": "",
    "quiz_answers_input": "",
    "quiz_feedback": "",
    "language_preference": "Auto detect",
    "answer_mode": "Default"
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# Load saved study data once after login. Keep account data separated by user ID.
if not st.session_state.account_data_loaded:
    saved = load_user_payload(st.session_state.auth_user_id)
    if saved:
        for saved_key in ("chats", "tasks", "quiz_history", "generated_summary",
                          "generated_notes", "generated_quiz", "quiz_data",
                          "generated_flashcards", "generated_questions"):
            if saved_key in saved:
                st.session_state[saved_key] = saved[saved_key]
        # Embedding arrays are restored from JSON lists to NumPy arrays.
        for saved_chat in st.session_state.chats.values():
            vectors = saved_chat.get("embeddings")
            if vectors is not None:
                saved_chat["embeddings"] = np.asarray(vectors, dtype=np.float32)
    st.session_state.account_data_loaded = True

if not st.session_state.chats:
    chat_id = str(uuid.uuid4())
    st.session_state.chats[chat_id] = {
        "title": "New chat",
        "messages": [],
        "chunks": [],
        "embeddings": None,
        "filename": None
    }
    st.session_state.active_chat = chat_id

if st.session_state.active_chat not in st.session_state.chats:
    st.session_state.active_chat = next(iter(st.session_state.chats))

active = st.session_state.chats[st.session_state.active_chat]

# ---------------- THEME ----------------

if st.session_state.dark_mode:
    C = {
        "bg": "#111019",
        "surface": "#1D1A29",
        "surface2": "#282337",
        "text": "#F6F3FC",
        "muted": "#BDB4CF",
        "border": "#403851",
        "accent": "#7047D7",
        "hover": "#5932B5",
        "input": "#252132"
    }
else:
    C = {
        "bg": "#F8F7FC",
        "surface": "#FFFFFF",
        "surface2": "#F0EBFA",
        "text": "#282238",
        "muted": "#716980",
        "border": "#DED6EA",
        "accent": "#7047D7",
        "hover": "#5932B5",
        "input": "#FFFFFF"
    }

# ---------------- CUSTOM CSS ----------------

st.markdown(f"""
<style>
/* Hide Streamlit branding and top toolbar */
#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}

header[data-testid="stHeader"] {{
    background: {C["bg"]};
}}

[data-testid="stToolbar"] {{
    visibility: hidden;
    height: 0;
}}

[data-testid="stDecoration"] {{
    display: none;
}}

/* Application background */
.stApp {{
    background-color: {C["bg"]};
    color: {C["text"]};
}}

.block-container {{
    max-width: 1500px;
    padding-top: 2.5rem;
    padding-bottom: 2rem;
}}

/* Sidebar */
[data-testid="stSidebar"] {{
    background-color: {C["surface"]};
    border-right: 1px solid {C["border"]};
}}

[data-testid="stSidebar"] > div:first-child {{
    padding-top: 1rem;
}}

/* Text */
h1, h2, h3, h4, p, li, label {{
    color: {C["text"]};
}}

.brand {{
    font-size: 30px;
    font-weight: 800;
    color: {C["text"]};
}}

.subtitle {{
    color: {C["muted"]};
    font-size: 14px;
}}

/* Welcome section */
.hero {{
    padding: 25px;
    border: 1px solid {C["border"]};
    background-color: {C["surface2"]};
    border-radius: 18px;
    margin: 15px 0 22px 0;
}}

.hero h2 {{
    color: {C["text"]};
    font-size: 26px;
    margin: 0 0 10px 0;
}}

.hero p {{
    color: {C["muted"]};
    margin-bottom: 0;
}}

/* FIX: all Streamlit buttons have readable text */
.stButton > button,
.stDownloadButton > button,
[data-testid="stFormSubmitButton"] > button {{
    background-color: {C["accent"]} !important;
    color: #FFFFFF !important;
    border: 1px solid {C["accent"]} !important;
    border-radius: 10px !important;
    min-height: 42px;
    font-weight: 600 !important;
    opacity: 1 !important;
    box-shadow: none !important;
}}

.stButton > button p,
.stButton > button span,
.stButton > button div,
.stDownloadButton > button p,
.stDownloadButton > button span,
.stDownloadButton > button div,
[data-testid="stFormSubmitButton"] button p {{
    color: #FFFFFF !important;
    opacity: 1 !important;
}}

.stButton > button:hover,
.stDownloadButton > button:hover,
[data-testid="stFormSubmitButton"] > button:hover {{
    background-color: {C["hover"]} !important;
    color: #FFFFFF !important;
    border-color: {C["hover"]} !important;
}}

.stButton > button:focus,
.stDownloadButton > button:focus {{
    color: #FFFFFF !important;
    border-color: #AC8BFF !important;
    box-shadow: 0 0 0 2px #AC8BFF !important;
}}

/* Chat */
[data-testid="stChatMessage"] {{
    background-color: {C["surface"]};
    border: 1px solid {C["border"]};
    border-radius: 14px;
    padding: 14px;
}}

[data-testid="stChatInput"] {{
    background-color: {C["input"]} !important;
    border: 1px solid {C["border"]} !important;
    border-radius: 12px !important;
}}

[data-testid="stChatInput"] textarea {{
    color: {C["text"]} !important;
    -webkit-text-fill-color: {C["text"]} !important;
}}

[data-testid="stChatInput"] textarea::placeholder {{
    color: {C["muted"]} !important;
    -webkit-text-fill-color: {C["muted"]} !important;
}}

/* Inputs */
[data-baseweb="input"] > div,
[data-baseweb="textarea"] > div,
[data-baseweb="select"] > div {{
    background-color: {C["input"]} !important;
    border-color: {C["border"]} !important;
}}

[data-baseweb="input"] input,
[data-baseweb="textarea"] textarea {{
    color: {C["text"]} !important;
    -webkit-text-fill-color: {C["text"]} !important;
}}

[data-testid="stFileUploader"] {{
    background-color: {C["surface"]};
    border: 1px dashed {C["border"]};
    border-radius: 12px;
    padding: 10px;
}}

[data-testid="stMetric"] {{
    background-color: {C["surface"]};
    border: 1px solid {C["border"]};
    border-radius: 12px;
    padding: 15px;
}}

[data-testid="stExpander"] {{
    border-color: {C["border"]};
    border-radius: 12px;
}}

hr {{
    border-color: {C["border"]};
}}


[data-testid="stFileUploader"] {{
    border-radius: 14px !important;
}}
[data-testid="stVerticalBlockBorderWrapper"] {{
    border-radius: 14px;
}}

/* Links */
a {{
    color: {C["accent"]} !important;
}}

/* Theme button has a distinct appearance */
.theme-button .stButton > button {{
    background-color: {C["surface2"]} !important;
    color: {C["text"]} !important;
    border: 1px solid {C["border"]} !important;
}}

.theme-button .stButton > button p,
.theme-button .stButton > button span {{
    color: {C["text"]} !important;
}}

.theme-button .stButton > button:hover {{
    background-color: {C["surface"]} !important;
}}
</style>
""", unsafe_allow_html=True)

# ---------------- AI CONFIGURATION ----------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def get_secret_or_env(name, default=""):
    """Read deployment secrets first, then local .env environment variables."""
    try:
        value = st.secrets.get(name, default)
        if value:
            return str(value).strip()
    except Exception:
        pass
    return os.getenv(name, default).strip()


def get_client():
    api_key = get_secret_or_env("GEMINI_API_KEY", "")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY nahi mili. Project ki .env file check karo."
        )

    return genai.Client(api_key=api_key)


def get_model_name():
    return get_secret_or_env("GEMINI_MODEL", "gemini-2.5-flash")


def generate_ai(prompt):
    client = get_client()

    response = client.models.generate_content(
        model=get_model_name(),
        contents=prompt
    )

    return response.text or "AI ne koi answer return nahi kiya."


def language_instructions(question, preference):
    if preference == "Roman Urdu":
        return (
            "Answer in natural Pakistani Roman Urdu using English "
            "letters. Keep technical terms in English where appropriate. "
            "Do not use Urdu script unless explicitly requested."
        )

    if preference == "English":
        return "Answer in clear, simple English."

    if preference == "Urdu":
        return "Answer in Urdu script."

    roman_urdu_words = re.findall(
        r"\b(kya|kaise|kyun|hai|hain|mujhe|samjhao|batao|"
        r"karna|hota|hoti|mein|mera|meri|aur|nahi|karo|"
        r"chahiye|bata|acha|asaan|farq|samajh|isko|"
        r"ka|ki|ke|se|ko|par|liye)\b",
        question.lower()
    )

    if roman_urdu_words:
        return (
            "The question is in Roman Urdu. Answer in natural "
            "Roman Urdu using English letters. Keep technical terms "
            "in English. Do not answer in Urdu script."
        )

    if re.search(r"[\u0600-\u06FF]", question):
        return "Answer in Urdu script."

    return "Answer in clear English, matching the user's language."


def build_chat_prompt(question, history):
    if st.session_state.answer_mode == "Study mode":
        style = """
Explain step by step.
Start with the definition, then WHY it matters, HOW it works,
a simple example, and a short recap.
Use comparison tables where helpful.
"""
    elif st.session_state.answer_mode == "Concise answers":
        style = "Give a short, direct, accurate answer."
    else:
        style = """
Act as a friendly university study tutor.
Explain concepts clearly, use examples when useful,
and avoid unnecessary complexity.
"""

    return f"""
You are AI StudyMate, a university study assistant.

LANGUAGE:
{language_instructions(question, st.session_state.language_preference)}

ANSWER STYLE:
{style}

CONVERSATION HISTORY:
{history[-5000:]}

STUDENT QUESTION:
{question}

Answer directly. Do not invent sources or citations.
"""


# ---------------- PDF / RAG ----------------

def extract_pdf_text(uploaded_file):
    reader = PdfReader(uploaded_file)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            pages.append({
                "page": page_number,
                "text": text
            })

    return pages


def chunk_pages(pages, chunk_size=180, overlap=35):
    chunks = []
    step = chunk_size - overlap

    for page in pages:
        words = page["text"].split()

        for start in range(0, len(words), step):
            text = " ".join(words[start:start + chunk_size])

            if text.strip():
                chunks.append({
                    "page": page["page"],
                    "text": text
                })

    return chunks


def create_embeddings(chunks):
    model = load_embedding_model()

    vectors = model.encode(
        [chunk["text"] for chunk in chunks],
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return np.asarray(vectors, dtype=np.float32)


def retrieve(question, chunks, embeddings, top_k=5):
    model = load_embedding_model()

    query_vector = model.encode(
        [question],
        normalize_embeddings=True
    )[0]

    scores = embeddings @ query_vector
    indices = np.argsort(scores)[::-1][:top_k]

    return [
        {**chunks[i], "score": float(scores[i])}
        for i in indices
    ]


def answer_from_pdf(question, passages, history):
    context = "\n\n".join(
        f"[Page {p['page']}]\n{p['text']}"
        for p in passages
    )

    return generate_ai(f"""
You are an academic tutor answering questions about a student's PDF.

LANGUAGE:
{language_instructions(question, st.session_state.language_preference)}

ANSWER STYLE:
{st.session_state.answer_mode}

RULES:
- Use the supplied PDF excerpts for document-based facts.
- Cite supporting page numbers in the form [Page 3].
- Never invent page numbers or citations.
- If the PDF does not contain the answer, say so.
- You may explain general concepts, but distinguish them from PDF facts.
- Explain step by step when requested.

CONVERSATION HISTORY:
{history[-4000:]}

PDF EXCERPTS:
{context}

STUDENT QUESTION:
{question}
""")


def get_document_context():
    return "\n\n".join(
        f"[Page {chunk['page']}]\n{chunk['text']}"
        for chunk in active["chunks"][:15]
    )


# ---------------- STUDY TOOLS ----------------

def generate_study_tool(tool_name):
    if not active["chunks"]:
        st.warning("Pehle sidebar se PDF upload aur process karo.")
        return

    instructions = {
        "Summary": """
Create a useful lecture summary.
Include the main idea, key concepts, definitions,
examples and a quick revision section.
""",
        "Notes": """
Create organized exam revision notes.
Use headings, definitions, bullet points, examples,
important terms and likely exam points.
""",
        "MCQ Quiz": """
Create exactly 10 multiple-choice questions based only on the document.
Return ONLY valid JSON with this structure:
{
  "questions": [
    {
      "question": "Question text",
      "options": {"A": "Option text", "B": "Option text", "C": "Option text", "D": "Option text"},
      "correct_answer": "A",
      "explanation": "Short explanation"
    }
  ]
}
Rules:
- Exactly four options A, B, C, D for every question.
- Only one correct answer per question.
- Questions must be clear and not paragraph-like.
- Do not include the answer key outside the JSON.
- Do not use markdown fences or any text before/after the JSON.
""",
        "Flashcards": """
Create 12 revision flashcards.
Use this format:
CARD 1
FRONT: Question or term
BACK: Answer or explanation
""",
        "Important Questions": """
Create 10 important exam questions based on the document.
Include short-answer and long-answer questions.
Provide a separate answer guide at the end.
""",
        "Explain Simply": """
Explain the main lecture topics to a beginner.
Use simple analogies, examples and step-by-step explanations.
Define technical terms.
"""
    }

    language = language_instructions(
        "Explain my lecture",
        st.session_state.language_preference
    )

    prompt = f"""
You are an expert university study tutor.

LANGUAGE:
{language}

TASK:
{instructions[tool_name]}

Use the document as your main source.
Do not invent page citations.

DOCUMENT:
{get_document_context()}
"""

    result = generate_ai(prompt)

    if tool_name == "Summary":
        st.session_state.generated_summary = result
    elif tool_name == "Notes":
        st.session_state.generated_notes = result
    elif tool_name == "MCQ Quiz":
        cleaned = result.strip()
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise ValueError("AI did not return a valid quiz structure. Please generate the quiz again.")
        quiz_data = json.loads(match.group(0))
        questions = quiz_data.get("questions", [])
        if not questions:
            raise ValueError("No questions were returned. Please generate the quiz again.")
        for q in questions:
            if not all(k in q for k in ("question", "options", "correct_answer", "explanation")):
                raise ValueError("Quiz data is incomplete. Please generate the quiz again.")
        st.session_state.quiz_data = questions
        st.session_state.generated_quiz = "ready"
        st.session_state.quiz_feedback = ""
        st.session_state.quiz_version += 1
    elif tool_name == "Flashcards":
        st.session_state.generated_flashcards = result
    elif tool_name == "Important Questions":
        st.session_state.generated_questions = result


# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.markdown("## 🎓 AI StudyMate")
    st.caption("Your personal AI learning assistant")

    st.caption(f"Signed in as **{st.session_state.get('auth_email', '')}**")
    if st.button("Log out", key="account_logout", use_container_width=True):
        # Save before logging out, then clear account-specific session data.
        current_payload = {
            "chats": st.session_state.chats,
            "tasks": st.session_state.tasks,
            "quiz_history": st.session_state.quiz_history,
            "generated_summary": st.session_state.generated_summary,
            "generated_notes": st.session_state.generated_notes,
            "generated_quiz": st.session_state.generated_quiz,
            "quiz_data": st.session_state.quiz_data,
            "generated_flashcards": st.session_state.generated_flashcards,
            "generated_questions": st.session_state.generated_questions,
        }
        try:
            save_user_payload(st.session_state.auth_user_id, current_payload)
        except Exception as error:
            st.warning(f"Could not save before logout: {error}")
        try:
            if supabase:
                supabase.auth.sign_out()
        except Exception:
            pass
        for session_key in list(st.session_state.keys()):
            del st.session_state[session_key]
        st.rerun()

    st.markdown('<div class="theme-button">', unsafe_allow_html=True)

    theme_label = (
        "☀️ Switch to Light Mode"
        if st.session_state.dark_mode
        else "🌙 Switch to Dark Mode"
    )

    if st.button(
        theme_label,
        key="theme_toggle",
        use_container_width=True
    ):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    st.selectbox(
        "Response language",
        ["Auto detect", "Roman Urdu", "English", "Urdu"],
        key="language_preference"
    )

    st.selectbox(
        "Answer mode",
        ["Default", "Study mode", "Concise answers"],
        key="answer_mode"
    )

    st.divider()
    st.markdown("### 💬 Conversations")

    if st.button("＋ New Chat", use_container_width=True):
        new_id = str(uuid.uuid4())

        st.session_state.chats[new_id] = {
            "title": "New chat",
            "messages": [],
            "chunks": [],
            "embeddings": None,
            "filename": None
        }

        st.session_state.active_chat = new_id
        st.rerun()

    for chat_id in list(st.session_state.chats.keys())[::-1]:
        chat = st.session_state.chats[chat_id]
        row1, row2 = st.columns([5, 1], gap="small")
        with row1:
            prefix = "● " if chat_id == st.session_state.active_chat else ""
            title = chat.get("title", "New chat")
            if st.button(prefix + title[:27], key=f"chat_{chat_id}", use_container_width=True):
                st.session_state.active_chat = chat_id
                st.rerun()
        with row2:
            if st.button("🗑️", key=f"delete_chat_{chat_id}", help="Delete this conversation"):
                st.session_state[f"confirm_delete_{chat_id}"] = True
        if st.session_state.get(f"confirm_delete_{chat_id}", False):
            st.warning(f"Delete '{chat.get('title', 'New chat')}'?")
            confirm_col, cancel_col = st.columns(2)
            with confirm_col:
                if st.button("Delete", key=f"confirm_delete_btn_{chat_id}"):
                    del st.session_state.chats[chat_id]
                    st.session_state.pop(f"confirm_delete_{chat_id}", None)
                    if not st.session_state.chats:
                        new_id = str(uuid.uuid4())
                        st.session_state.chats[new_id] = {
                            "title": "New chat", "messages": [], "chunks": [],
                            "embeddings": None, "filename": None
                        }
                    if st.session_state.active_chat == chat_id:
                        st.session_state.active_chat = next(iter(st.session_state.chats))
                    st.rerun()
            with cancel_col:
                if st.button("Cancel", key=f"cancel_delete_btn_{chat_id}"):
                    st.session_state.pop(f"confirm_delete_{chat_id}", None)
                    st.rerun()

    st.divider()
    st.caption("📎 Add a PDF from the ＋ Add PDF button above the chat. Each conversation keeps its own document.")


# ---------------- HEADER ----------------

header_left, header_right = st.columns([5, 1])

with header_left:
    st.markdown(
        '<div class="brand">🎓 AI StudyMate</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="subtitle">Learn smarter. Prepare better. '
        'Study with AI.</div>',
        unsafe_allow_html=True
    )

with header_right:
    message_count = sum(
        len(chat["messages"])
        for chat in st.session_state.chats.values()
    )
    st.metric("Messages", message_count)


upload_col, hint_col = st.columns([1, 5], vertical_alignment="center")
with upload_col:
    if st.button("＋ Add PDF", key="open_pdf_upload", use_container_width=True):
        st.session_state.show_pdf_uploader = not st.session_state.show_pdf_uploader
with hint_col:
    if active.get("filename"):
        st.caption(f"📄 Connected document: **{active['filename']}**")
    else:
        st.caption("Attach lecture notes or a textbook PDF to ask document-based questions.")

if st.session_state.show_pdf_uploader:
    with st.container(border=True):
        st.markdown("#### 📎 Add a study document")
        st.write("Choose a PDF, then process it so StudyMate can search it and cite page numbers.")
        uploaded_pdf = st.file_uploader(
            "Choose PDF file",
            type=["pdf"],
            key=f"main_pdf_{st.session_state.active_chat}",
            help="Text-based PDFs work best. Scanned PDFs may require OCR."
        )
        process_col, close_col = st.columns([1, 1])
        with process_col:
            process_pdf = st.button(
                "📚 Process PDF",
                key=f"process_pdf_{st.session_state.active_chat}",
                disabled=uploaded_pdf is None,
                use_container_width=True
            )
        with close_col:
            if st.button("Close", key="close_pdf_upload", use_container_width=True):
                st.session_state.show_pdf_uploader = False
                st.rerun()
        if process_pdf and uploaded_pdf is not None:
            try:
                with st.spinner("Reading PDF and building its search index..."):
                    pages = extract_pdf_text(uploaded_pdf)
                    chunks = chunk_pages(pages)
                    if not chunks:
                        st.error("No readable text was found. This may be a scanned PDF that needs OCR.")
                    else:
                        embeddings = create_embeddings(chunks)
                        active["chunks"] = chunks
                        active["embeddings"] = embeddings
                        active["filename"] = uploaded_pdf.name
                        st.session_state.show_pdf_uploader = False
                        st.success(f"Ready to study: {uploaded_pdf.name} ({len(pages)} pages indexed).")
                        st.rerun()
            except Exception as error:
                st.error(f"Could not process PDF: {error}")

if active.get("filename"):
    if st.button("Remove current PDF", key=f"remove_pdf_{st.session_state.active_chat}"):
        active["chunks"] = []
        active["embeddings"] = None
        active["filename"] = None
        st.rerun()



# ---------------- MAIN TABS ----------------

chat_tab, tools_tab, planner_tab, progress_tab = st.tabs([
    "💬 AI Chat",
    "📚 Study Tools",
    "🗓️ Study Planner",
    "📊 Progress"
])


# ---------------- AI CHAT ----------------

with chat_tab:
    if active["filename"]:
        st.info(f"📄 Active PDF: {active['filename']}")
    else:
        st.caption("Ask general questions or upload a PDF for document-based answers.")

    for message in active["messages"]:
        avatar = "🎓" if message["role"] == "assistant" else "🧑‍🎓"

        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

            if message.get("sources"):
                with st.expander("📖 View PDF sources"):
                    for source in message["sources"]:
                        st.markdown(f"**Page {source['page']}**")
                        st.write(source["text"])

    question = st.chat_input(
        "Ask in English or Roman Urdu...",
        key=f"input_{st.session_state.active_chat}"
    )

    if question:
        if not active["messages"]:
            active["title"] = " ".join(question.split())[:32] + ("…" if len(question) > 32 else "")

        active["messages"].append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(question)

        sources = []

        try:
            history = "\n".join(
                f"{m['role']}: {m['content']}"
                for m in active["messages"][-8:-1]
            )

            with st.chat_message("assistant", avatar="🎓"):
                with st.spinner("StudyMate is thinking..."):
                    if (
                        active["chunks"]
                        and active["embeddings"] is not None
                    ):
                        sources = retrieve(
                            question,
                            active["chunks"],
                            active["embeddings"],
                            top_k=5
                        )

                        answer = answer_from_pdf(
                            question,
                            sources,
                            history
                        )
                    else:
                        answer = generate_ai(
                            build_chat_prompt(question, history)
                        )

                st.markdown(answer)

                if sources:
                    with st.expander("📖 View PDF sources"):
                        for source in sources:
                            st.markdown(f"**Page {source['page']}**")
                            st.write(source["text"])

            active["messages"].append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })

        except Exception as error:
            answer = (
                "AI response generate nahi ho saka. "
                "API key, model access, internet aur quota check karo."
            )

            active["messages"].append({
                "role": "assistant",
                "content": answer,
                "sources": []
            })

            st.error(answer)
            st.caption(f"Technical details: {error}")

        st.rerun()

    if active["messages"]:
        chat_export = "\n\n".join(
            f"{m['role'].upper()}:\n{m['content']}"
            for m in active["messages"]
        )

        st.download_button(
            "⬇️ Download Chat",
            chat_export,
            file_name="studymate_chat.txt",
            mime="text/plain",
            key=f"download_chat_{st.session_state.active_chat}"
        )


# ---------------- STUDY TOOLS ----------------

with tools_tab:
    st.subheader("📚 Your Study Toolkit")

    st.write(
        "Upload and process a PDF to create summaries, "
        "revision notes, quizzes and flashcards."
    )

    if not active["chunks"]:
        st.info("Upload a PDF from the sidebar to enable document-based tools.")

    tool_list = [
        ("📝 Summary", "Summary"),
        ("📒 Revision Notes", "Notes"),
        ("🧠 MCQ Quiz", "MCQ Quiz"),
        ("🃏 Flashcards", "Flashcards"),
        ("❓ Important Questions", "Important Questions"),
        ("💡 Explain Simply", "Explain Simply")
    ]

    for row_start in range(0, len(tool_list), 3):
        columns = st.columns(3)

        for index, (label, tool_name) in enumerate(
            tool_list[row_start:row_start + 3]
        ):
            with columns[index]:
                if st.button(
                    label,
                    key=f"tool_{tool_name}",
                    use_container_width=True
                ):
                    try:
                        with st.spinner(f"Generating {tool_name}..."):
                            generate_study_tool(tool_name)
                        st.rerun()
                    except Exception as error:
                        st.error(f"Study tool failed: {error}")

    outputs = [
        ("📝 Generated Summary", "generated_summary", "study_summary.txt"),
        ("📒 Revision Notes", "generated_notes", "revision_notes.txt"),
        ("🃏 Flashcards", "generated_flashcards", "flashcards.txt"),
        ("❓ Important Questions", "generated_questions", "important_questions.txt")
    ]

    for title, state_key, filename in outputs:
        content = st.session_state[state_key]

        if content:
            with st.expander(title, expanded=True):
                st.markdown(content)
                st.download_button(
                    f"Download {title}",
                    content,
                    file_name=filename,
                    mime="text/plain",
                    key=f"download_{state_key}"
                )

    if st.session_state.generated_quiz and st.session_state.quiz_data:
        st.subheader("🧠 Practice Quiz")
        st.caption("Choose one option for each question. Correct answers stay hidden until you submit.")

        with st.form(f"quiz_form_{st.session_state.quiz_version}"):
            selected_answers = {}
            for index, question in enumerate(st.session_state.quiz_data, start=1):
                st.markdown(f"**Question {index} of {len(st.session_state.quiz_data)}**")
                st.markdown(question["question"])
                options = question["options"]
                option_labels = [f"{letter}. {options[letter]}" for letter in ["A", "B", "C", "D"] if letter in options]
                selected = st.radio(
                    "Select one answer",
                    option_labels,
                    index=None,
                    key=f"quiz_{st.session_state.quiz_version}_{index}",
                    label_visibility="collapsed"
                )
                selected_answers[index - 1] = selected[:1] if selected else None
                st.divider()
            submitted = st.form_submit_button("Check Answers", use_container_width=True)

        if submitted:
            score = 0
            answered = 0
            results = []
            for index, question in enumerate(st.session_state.quiz_data):
                selected = selected_answers[index]
                correct = str(question["correct_answer"]).strip().upper()[:1]
                if selected:
                    answered += 1
                is_correct = selected == correct
                if is_correct:
                    score += 1
                options = question["options"]
                results.append({
                    "number": index + 1,
                    "question": question["question"],
                    "selected": selected,
                    "selected_text": options.get(selected, "") if selected else "",
                    "correct": correct,
                    "correct_text": options.get(correct, ""),
                    "is_correct": is_correct,
                    "explanation": question["explanation"]
                })
            st.session_state.quiz_feedback = json.dumps({
                "score": score,
                "total": len(results),
                "answered": answered,
                "results": results
            }, ensure_ascii=False)
            st.session_state.quiz_history.append({
                "date": str(date.today()),
                "score": score,
                "total": len(results),
                "feedback": st.session_state.quiz_feedback
            })
            st.rerun()

        if st.session_state.quiz_feedback:
            try:
                feedback = json.loads(st.session_state.quiz_feedback)
                st.markdown("### 📊 Your Results")
                score_col, answered_col, skipped_col = st.columns(3)
                score_col.metric("Score", f"{feedback['score']}/{feedback['total']}")
                answered_col.metric("Answered", feedback["answered"])
                skipped_col.metric("Not answered", feedback["total"] - feedback["answered"])
                st.progress(feedback["score"] / max(feedback["total"], 1))
                st.markdown("### Answer Review")
                for item in feedback["results"]:
                    icon = "✅" if item["is_correct"] else ("⚪" if item["selected"] is None else "❌")
                    with st.container(border=True):
                        st.markdown(f"**{icon} {item['number']}. {item['question']}**")
                        st.markdown(f"- **Your answer:** {item['selected'] + '. ' + item['selected_text'] if item['selected'] else 'Not answered'}")
                        st.markdown(f"- **Correct answer:** {item['correct']}. {item['correct_text']}")
                        st.markdown(f"- **Explanation:** {item['explanation']}")
                review_text = "\n".join(
                    f"{x['number']}. {x['question']}\n"
                    f"- Your answer: {x['selected'] or 'Not answered'}\n"
                    f"- Correct answer: {x['correct']}. {x['correct_text']}\n"
                    f"- Explanation: {x['explanation']}"
                    for x in feedback["results"]
                )
                st.download_button(
                    "⬇️ Download Quiz Results",
                    f"Score: {feedback['score']}/{feedback['total']}\n\n{review_text}",
                    file_name="quiz_results.txt",
                    mime="text/plain",
                    key=f"download_quiz_{st.session_state.quiz_version}"
                )
            except (json.JSONDecodeError, KeyError, TypeError):
                st.warning("Generate a new quiz to use the improved quiz format.")


    if st.button("Clear Generated Study Material"):
        for key in [
            "generated_summary",
            "generated_notes",
            "generated_quiz",
            "generated_flashcards",
            "generated_questions",
            "quiz_feedback"
        ]:
            st.session_state[key] = ""

        st.session_state.quiz_data = []
        st.session_state.quiz_version += 1
        st.rerun()


# ---------------- STUDY PLANNER ----------------

with planner_tab:
    st.subheader("🗓️ Daily Study Planner")
    st.write("Create study tasks and mark them complete.")

    with st.form("add_task_form", clear_on_submit=True):
        task_name = st.text_input(
            "Study task",
            placeholder="Example: Revise DBMS normalization"
        )

        task_date = st.date_input("Study date", value=date.today())

        task_minutes = st.number_input(
            "Estimated duration (minutes)",
            min_value=5,
            max_value=600,
            value=30,
            step=5
        )

        add_task = st.form_submit_button("＋ Add Task")

        if add_task:
            if task_name.strip():
                st.session_state.tasks.append({
                    "id": str(uuid.uuid4()),
                    "name": task_name.strip(),
                    "date": str(task_date),
                    "minutes": int(task_minutes),
                    "done": False
                })
                st.rerun()
            else:
                st.warning("Pehle task ka naam likho.")

    if st.session_state.tasks:
        st.markdown("### Your Tasks")

        for task in st.session_state.tasks:
            col1, col2 = st.columns([5, 1])

            with col1:
                status = "✅" if task["done"] else "⏳"

                st.markdown(
                    f"**{status} {task['name']}**  \n"
                    f"📅 {task['date']} · "
                    f"⏱️ {task['minutes']} minutes"
                )

            with col2:
                checked = st.checkbox(
                    "Done",
                    value=task["done"],
                    key=f"task_{task['id']}"
                )

                if checked != task["done"]:
                    task["done"] = checked
                    st.rerun()

        if st.button("Remove Completed Tasks"):
            st.session_state.tasks = [
                task for task in st.session_state.tasks
                if not task["done"]
            ]
            st.rerun()

        plan_text = "\n".join(
            f"{'[DONE]' if task['done'] else '[TODO]'} "
            f"{task['date']} - {task['name']} "
            f"({task['minutes']} minutes)"
            for task in st.session_state.tasks
        )

        st.download_button(
            "⬇️ Download Study Plan",
            plan_text,
            file_name="study_plan.txt",
            mime="text/plain"
        )

    else:
        st.info("Abhi koi task nahi hai. Apna pehla task add karo.")


# ---------------- PROGRESS TRACKER ----------------

with progress_tab:
    st.subheader("📊 Learning Progress")

    total_messages = sum(
        len(chat["messages"])
        for chat in st.session_state.chats.values()
    )

    completed_tasks = sum(
        task["done"] for task in st.session_state.tasks
    )

    total_tasks = len(st.session_state.tasks)

    study_minutes = sum(
        task["minutes"]
        for task in st.session_state.tasks
        if task["done"]
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Conversations",
        len(st.session_state.chats)
    )

    col2.metric(
        "Messages",
        total_messages
    )

    col3.metric(
        "Tasks Completed",
        f"{completed_tasks}/{total_tasks}"
    )

    col4.metric(
        "Study Minutes",
        study_minutes
    )

    if total_tasks:
        completion = completed_tasks / total_tasks

        st.markdown("### Study Task Completion")
        st.progress(completion)
        st.write(f"{completion * 100:.0f}% tasks completed.")

    st.markdown("### 🧠 Quiz Practice History")

    if st.session_state.quiz_history:
        for index, record in enumerate(
            reversed(st.session_state.quiz_history),
            start=1
        ):
            with st.expander(
                f"Quiz attempt {index} — {record['date']}"
            ):
                st.markdown(record["feedback"])
    else:
        st.info("Quiz history yahan show hogi jab tum quiz check karoge.")

    st.caption("Your saved chats, planner and progress sync through your Supabase account. PDF files and extracted PDF text are not saved permanently to cloud; upload the PDF again after a fresh session if needed.")


# ---------------- SAVE ACCOUNT DATA ----------------
# Save chats, processed PDF chunks/embeddings, planner tasks and study outputs
# after each successful app render, scoped to the signed-in account.
try:
    persistent_payload = {
        "chats": st.session_state.chats,
        "tasks": st.session_state.tasks,
        "quiz_history": st.session_state.quiz_history,
        "generated_summary": st.session_state.generated_summary,
        "generated_notes": st.session_state.generated_notes,
        "generated_quiz": st.session_state.generated_quiz,
        "quiz_data": st.session_state.quiz_data,
        "generated_flashcards": st.session_state.generated_flashcards,
        "generated_questions": st.session_state.generated_questions,
    }
    save_user_payload(st.session_state.auth_user_id, persistent_payload)
except Exception as persistence_error:
    st.warning(f"Your changes could not be saved right now: {persistence_error}")


# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "🎓 AI StudyMate | Learn smarter, one concept at a time."
)