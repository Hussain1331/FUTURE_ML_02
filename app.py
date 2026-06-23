import streamlit as st
import joblib
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
# -------------------------------
# Page Configuration
# -------------------------------

st.set_page_config(
    page_title="Support Ticket Classifier",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# Custom CSS
# -------------------------------

st.markdown("""
<style>
    /* ---- Google Fonts ---- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap');

    /* ---- Global Reset ---- */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ---- App background ---- */
    .stApp {
        background-color: #FFFFFF;
    }

    /* ---- Sidebar ---- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1A3A6B 0%, #1E4D8C 60%, #2563B0 100%);
        border-right: none;
    }
    [data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #FFFFFF !important;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.2);
    }
    [data-testid="stSidebar"] .stMarkdown p {
        color: rgba(255,255,255,0.82) !important;
        font-size: 0.875rem;
        line-height: 1.6;
    }
    /* Sidebar info boxes */
    .sidebar-info-card {
        background: rgba(255,255,255,0.10);
        border: 1px solid rgba(255,255,255,0.18);
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 12px;
    }
    .sidebar-info-card .label {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: rgba(255,255,255,0.55) !important;
        margin-bottom: 4px;
    }
    .sidebar-info-card .value {
        font-size: 0.95rem;
        font-weight: 500;
        color: #FFFFFF !important;
    }
    /* Sidebar badge */
    .sidebar-badge {
        display: inline-block;
        background: rgba(255,255,255,0.15);
        border: 1px solid rgba(255,255,255,0.25);
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.75rem;
        font-weight: 500;
        color: #FFFFFF !important;
        margin: 3px 3px 3px 0;
    }

    /* ---- Hero Header ---- */
    .hero-header {
        background: linear-gradient(135deg, #1A3A6B 0%, #2563B0 50%, #3B82C4 100%);
        border-radius: 18px;
        padding: 42px 48px;
        margin-bottom: 32px;
        position: relative;
        overflow: hidden;
    }
    .hero-header::before {
        content: '';
        position: absolute;
        top: -60px; right: -60px;
        width: 240px; height: 240px;
        background: rgba(255,255,255,0.06);
        border-radius: 50%;
    }
    .hero-header::after {
        content: '';
        position: absolute;
        bottom: -80px; left: 40%;
        width: 300px; height: 300px;
        background: rgba(255,255,255,0.04);
        border-radius: 50%;
    }
    .hero-eyebrow {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: rgba(255,255,255,0.60);
        margin-bottom: 10px;
    }
    .hero-title {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 2.4rem;
        font-weight: 800;
        color: #FFFFFF;
        line-height: 1.15;
        margin-bottom: 14px;
    }
    .hero-title span {
        color: #93C5FD;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: rgba(255,255,255,0.72);
        max-width: 540px;
        line-height: 1.6;
        font-weight: 400;
    }

    /* ---- Section label ---- */
    .section-label {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #2563B0;
        margin-bottom: 8px;
    }
    .section-title {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 1.25rem;
        font-weight: 700;
        color: #1A3A6B;
        margin-bottom: 4px;
    }

    /* ---- Text area override ---- */
    .stTextArea textarea {
        border: 2px solid #CBD5E1 !important;
        border-radius: 12px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.95rem !important;
        color: #1E293B !important;
        background: #F8FAFC !important;
        padding: 14px 16px !important;
        transition: border-color 0.2s ease;
        resize: vertical;
    }
    .stTextArea textarea:focus {
        border-color: #2563B0 !important;
        box-shadow: 0 0 0 3px rgba(37,99,176,0.12) !important;
        background: #FFFFFF !important;
    }
    .stTextArea label {
        font-weight: 600 !important;
        color: #1A3A6B !important;
        font-size: 0.9rem !important;
    }

    /* ---- Primary Button ---- */
    .stButton > button {
        background: linear-gradient(135deg, #1A3A6B 0%, #2563B0 100%) !important;
        color: #FFFFFF !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 14px 40px !important;
        height: auto !important;
        letter-spacing: 0.02em;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 14px rgba(37,99,176,0.30) !important;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 20px rgba(37,99,176,0.38) !important;
    }
    .stButton > button:active {
        transform: translateY(0px) !important;
    }

    /* ---- Result Cards ---- */
    .result-card {
        border-radius: 16px;
        padding: 28px 30px;
        margin-top: 8px;
        position: relative;
        overflow: hidden;
    }
    .result-card-category {
        background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
        border: 2px solid #BFDBFE;
    }
    .result-card-priority {
        background: linear-gradient(135deg, #F0F9FF 0%, #E0F2FE 100%);
        border: 2px solid #BAE6FD;
    }
    .result-card .rc-label {
        font-size: 0.70rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #2563B0;
        margin-bottom: 8px;
    }
    .result-card .rc-icon {
        font-size: 2rem;
        margin-bottom: 8px;
        display: block;
    }
    .result-card .rc-value {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 1.6rem;
        font-weight: 800;
        color: #1A3A6B;
        line-height: 1.2;
    }
    .result-card .rc-desc {
        font-size: 0.83rem;
        color: #64748B;
        margin-top: 6px;
        font-weight: 400;
    }

    /* Priority badge colouring */
    .priority-high   { color: #DC2626 !important; }
    .priority-medium { color: #D97706 !important; }
    .priority-low    { color: #16A34A !important; }

    /* ---- Divider ---- */
    .custom-divider {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, #2563B0 0%, #93C5FD 50%, transparent 100%);
        border-radius: 2px;
        margin: 24px 0;
    }

    /* ---- Warning / Info ---- */
    .stAlert {
        border-radius: 12px !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* ---- Hide Streamlit branding ---- */
    #MainMenu, footer, header { visibility: hidden; }

    /* ---- Stats row ---- */
    .stat-card {
        background: #F8FAFC;
        border: 1.5px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px 22px;
        text-align: center;
    }
    .stat-card .stat-num {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 1.6rem;
        font-weight: 800;
        color: #1A3A6B;
    }
    .stat-card .stat-label {
        font-size: 0.78rem;
        color: #64748B;
        font-weight: 500;
        margin-top: 2px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Load Models
# -------------------------------

@st.cache_resource
def load_models():
    queue_model     = joblib.load("models/queue_model.pkl")
    priority_model  = joblib.load("models/priority_model.pkl")
    tfidf           = joblib.load("models/tfidf.pkl")
    queue_encoder   = joblib.load("models/queue_encoder.pkl")
    priority_encoder = joblib.load("models/priority_encoder.pkl")
    return queue_model, priority_model, tfidf, queue_encoder, priority_encoder

queue_model, priority_model, tfidf, queue_encoder, priority_encoder = load_models()

# -------------------------------
# NLP Preprocessing
# -------------------------------

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text   = text.lower()
    text   = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)

# -------------------------------
# Prediction Function
# -------------------------------

def predict_ticket(text):
    cleaned       = preprocess_text(text)
    vector        = tfidf.transform([cleaned])
    queue_pred    = queue_model.predict(vector)
    priority_pred = priority_model.predict(vector)
    queue_label    = queue_encoder.inverse_transform(queue_pred)[0]
    priority_label = priority_encoder.inverse_transform(priority_pred)[0]
    return queue_label, priority_label

def priority_color_class(p):
    p_lower = p.lower()
    if "high" in p_lower or "critical" in p_lower or "urgent" in p_lower:
        return "priority-high"
    if "medium" in p_lower or "normal" in p_lower or "moderate" in p_lower:
        return "priority-medium"
    return "priority-low"

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("""
        <div style='text-align:center; padding: 10px 0 20px;'>
            <div style='font-size:2.8rem;'>🎫</div>
            <div style='font-family:"Plus Jakarta Sans",sans-serif;
                        font-size:1.15rem; font-weight:800;
                        color:#FFFFFF; margin-top:6px;'>
                TicketAI
            </div>
            <div style='font-size:0.78rem; color:rgba(255,255,255,0.55);
                        margin-top:2px; letter-spacing:0.05em;'>
                Support Classifier v2.0
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(255,255,255,0.15); margin-bottom:20px;'>",
                unsafe_allow_html=True)

    st.markdown("### 📋 How It Works")
    st.markdown("""
        <p>1. Paste your support ticket text into the input field.</p>
        <p>2. Click <strong>Classify Ticket</strong> to run the model.</p>
        <p>3. The system will output the <strong>queue category</strong> and <strong>priority level</strong> instantly.</p>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(255,255,255,0.15); margin: 20px 0;'>",
                unsafe_allow_html=True)

    st.markdown("### 🤖 Model Info")
    st.markdown("""
        <div class="sidebar-info-card">
            <div class="label">Algorithm</div>
            <div class="value">ML Classifier</div>
        </div>
        <div class="sidebar-info-card">
            <div class="label">Vectorizer</div>
            <div class="value">TF-IDF</div>
        </div>
        <div class="sidebar-info-card">
            <div class="label">NLP Pipeline</div>
            <div class="value">NLTK — Lemmatization + Stopword Removal</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(255,255,255,0.15); margin: 20px 0;'>",
                unsafe_allow_html=True)

    st.markdown("### 🏷️ Supported Queues")
    example_queues = ["Billing", "Technical", "Account", "Returns", "General"]
    badges = " ".join(f'<span class="sidebar-badge">{q}</span>' for q in example_queues)
    st.markdown(badges, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(255,255,255,0.15); margin: 20px 0;'>",
                unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center; font-size:0.72rem; color:rgba(255,255,255,0.35);'>"
        "Powered by Streamlit · Built with ❤️</p>",
        unsafe_allow_html=True
    )

# ============================================================
# MAIN CONTENT
# ============================================================

# --- Hero ---
st.markdown("""
<div class="hero-header">
    <div class="hero-eyebrow">AI-Powered Support Intelligence</div>
    <div class="hero-title">Support Ticket<br><span>Classification System</span></div>
    <div class="hero-subtitle">
        Automatically route and prioritise incoming support tickets using
        machine learning — saving your team hours of manual triage every day.
    </div>
</div>
""", unsafe_allow_html=True)

# --- Stats row ---
col_s1, col_s2, col_s3, col_s4 = st.columns(4)
with col_s1:
    st.markdown("""
        <div class="stat-card">
            <div class="stat-num">⚡ Fast</div>
            <div class="stat-label">Real-time prediction</div>
        </div>""", unsafe_allow_html=True)
with col_s2:
    st.markdown("""
        <div class="stat-card">
            <div class="stat-num">🧠 NLP</div>
            <div class="stat-label">Smart text processing</div>
        </div>""", unsafe_allow_html=True)
with col_s3:
    st.markdown("""
        <div class="stat-card">
            <div class="stat-num">🎯 Accurate</div>
            <div class="stat-label">Trained ML models</div>
        </div>""", unsafe_allow_html=True)
with col_s4:
    st.markdown("""
        <div class="stat-card">
            <div class="stat-num">📊 2-in-1</div>
            <div class="stat-label">Queue + Priority output</div>
        </div>""", unsafe_allow_html=True)

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# --- Input Section ---
left_col, right_col = st.columns([3, 2], gap="large")

with left_col:
    st.markdown('<div class="section-label">Step 1 — Ticket Input</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Enter the support ticket</div>', unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#64748B; font-size:0.875rem; margin-bottom:16px;'>"
        "Paste or type the customer's message below. The model will analyse "
        "the intent and tone to classify it automatically."
        "</p>",
        unsafe_allow_html=True
    )

    ticket_text = st.text_area(
        label="Ticket description",
        placeholder="e.g. — I've been charged twice for my subscription this month "
                    "and I need an urgent refund processed before end of day...",
        height=220,
        label_visibility="collapsed"
    )

    char_count = len(ticket_text.strip())
    word_count = len(ticket_text.split()) if ticket_text.strip() else 0
    st.markdown(
        f"<p style='font-size:0.75rem; color:#94A3B8; text-align:right; margin-top:4px;'>"
        f"{word_count} words · {char_count} characters</p>",
        unsafe_allow_html=True
    )

    submitted = st.button("🔍 Classify Ticket", use_container_width=True)

with right_col:
    st.markdown('<div class="section-label">Step 2 — Results</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Classification output</div>', unsafe_allow_html=True)

    if submitted:
        if ticket_text.strip() == "":
            st.warning("⚠️  Please enter a ticket before classifying.")
        else:
            with st.spinner("Analysing ticket…"):
                queue, priority = predict_ticket(ticket_text)

            p_class = priority_color_class(priority)

            st.markdown(f"""
                <div class="result-card result-card-category">
                    <div class="rc-label">Queue Category</div>
                    <span class="rc-icon">📂</span>
                    <div class="rc-value">{queue}</div>
                    <div class="rc-desc">This ticket has been routed to the <strong>{queue}</strong> support queue.</div>
                </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
                <div class="result-card result-card-priority" style="margin-top:16px;">
                    <div class="rc-label">Priority Level</div>
                    <span class="rc-icon">🚦</span>
                    <div class="rc-value {p_class}">{priority}</div>
                    <div class="rc-desc">Respond according to your <strong>{priority}</strong> SLA guidelines.</div>
                </div>
            """, unsafe_allow_html=True)

    else:
        st.markdown("""
            <div style='
                background:#F8FAFC;
                border: 2px dashed #CBD5E1;
                border-radius: 16px;
                padding: 48px 28px;
                text-align: center;
                color: #94A3B8;
            '>
                <div style='font-size:2.5rem; margin-bottom:12px;'>🎫</div>
                <div style='font-weight:600; font-size:0.95rem; color:#64748B; margin-bottom:6px;'>
                    Awaiting classification
                </div>
                <div style='font-size:0.82rem; line-height:1.6;'>
                    Enter a ticket on the left and<br>click <strong>Classify Ticket</strong> to see results here.
                </div>
            </div>
        """, unsafe_allow_html=True)

# --- Tip Section ---
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
with st.expander("💡  Tips for better classification accuracy"):
    st.markdown("""
    - **Be descriptive**: Include as much context as possible — the model performs best on complete sentences.
    - **Include key details**: Mention the product, error messages, or account actions to help routing.
    - **Avoid abbreviations**: Spell out terms where possible for more accurate NLP parsing.
    - **One issue per ticket**: Single-issue tickets classify more reliably than multi-issue ones.
    """)
