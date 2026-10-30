
import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

# ---------------- PROJECT PATHS ----------------

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"
DATASET_PATH = PROJECT_DIR / "data" / "cleaned_reviews.csv"

# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="ReviewGuard | Fake Review Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS ----------------


st.markdown("""
<style>
.stApp {
    background-color: #0e1117;
    color: #ffffff;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

[data-testid="stSidebar"] {
    background-color: #161b22;
}

/* Dashboard metric cards */
.metric-card {
    background-color: #1c2533;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #303b4d;
    text-align: center;
}

.metric-label {
    color: #aab7c8;
    font-size: 14px;
}

.metric-value {
    color: #58a6ff;
    font-size: 28px;
    font-weight: bold;
}

/* Section headings */
.section-title {
    color: #58a6ff;
    font-size: 23px;
    font-weight: bold;
}

/* Review input box */
.stTextArea textarea {
    background-color: #ffffff !important;
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    caret-color: #000000 !important;
    border: 2px solid #58a6ff !important;
    font-size: 16px !important;
    font-weight: 500 !important;
    opacity: 1 !important;
}

.stTextArea textarea::placeholder {
    color: #555555 !important;
    -webkit-text-fill-color: #555555 !important;
    opacity: 1 !important;
}

/* Analyze button */
div.stButton > button {
    background-color: #238636;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 10px 25px;
    font-weight: bold;
    width: 100%;
}

div.stButton > button:hover {
    background-color: #2ea043;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_models():
    model = joblib.load(MODEL_DIR / "logistic_regression_model.pkl")
    vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_models()

# ---------------- LOAD DATASET ----------------

@st.cache_data
def load_dataset():
    return pd.read_csv(DATASET_PATH)

df = load_dataset()

# ---------------- SIDEBAR ----------------

st.sidebar.title("🔍 ReviewGuard")

st.sidebar.write("Fake Review Detection System")

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Review Analysis", "About Project"]
)

st.sidebar.divider()

st.sidebar.caption("Data Mining Project")
st.sidebar.caption("TF-IDF + Logistic Regression")

# ---------------- DASHBOARD PAGE ----------------

if page == "Dashboard":

    st.title("📊 ReviewGuard Dashboard")

    st.write(
        "An intelligent text classification system "
        "for analyzing product reviews."
    )

    st.divider()

    # Dataset metrics
    total_reviews = len(df)
    original_reviews = int((df["label"] == "OR").sum())
    generated_reviews = int((df["label"] == "CG").sum())

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Reviews</div>
            <div class="metric-value">{total_reviews:,}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Original Reviews</div>
            <div class="metric-value">{original_reviews:,}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Computer-generated</div>
            <div class="metric-value">{generated_reviews:,}</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.subheader("📈 Dataset Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Review Type Distribution")
        review_counts = df["review_type"].value_counts()
        st.bar_chart(review_counts)

    with col2:
        st.write("Rating Distribution")
        rating_counts = df["rating"].value_counts().sort_index()
        st.bar_chart(rating_counts)

    st.divider()

    st.subheader("🤖 Machine Learning Performance")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Naive Bayes Accuracy", "88.44%")

    with col2:
        st.metric("Logistic Regression Accuracy", "90.78%")

    st.success(
        "Logistic Regression achieved the highest accuracy "
        "on the held-out test dataset."
    )

# ---------------- REVIEW ANALYSIS PAGE ----------------

elif page == "Review Analysis":

    st.title("🔎 Review Analysis")

    st.write(
        "Enter a product review to analyze its similarity "
        "to the dataset's computer-generated or original labels."
    )

    review = st.text_area(
        "Enter Product Review",
        placeholder="Write your product review here...",
        height=180
    )

    if st.button("Analyze Review"):

        if not review.strip():
            st.warning("Please enter a review first.")

        else:
            review_tfidf = vectorizer.transform([review])

            probabilities = model.predict_proba(review_tfidf)[0]
            probability_dict = dict(
                zip(model.classes_, probabilities)
            )

            cg_probability = probability_dict["CG"]
            or_probability = probability_dict["OR"]

            highest_probability = max(
                cg_probability,
                or_probability
            )

            st.divider()
            st.subheader("Prediction Result")

            if highest_probability < 0.70:

                st.warning("⚠️ Uncertain Prediction")

            elif cg_probability > or_probability:

                st.error("Computer-generated Review")

            else:

                st.success("Original Review")

            col1, col2 = st.columns(2)

            with col1:
                st.write("Computer-generated")
                st.metric(
                    "Model Probability",
                    f"{cg_probability * 100:.2f}%"
                )
                st.progress(float(cg_probability))

            with col2:
                st.write("Original")
                st.metric(
                    "Model Probability",
                    f"{or_probability * 100:.2f}%"
                )
                st.progress(float(or_probability))

            st.caption(
                "Probabilities represent model estimates, "
                "not proof of review authenticity."
            )

# ---------------- ABOUT PROJECT PAGE ----------------

elif page == "About Project":

    st.title("📚 About the Project")

    st.subheader("Fake Review Detection Using Machine Learning")

    st.write("""
    This project uses text mining and machine learning
    techniques to classify product reviews into two
    dataset categories: Computer-generated (CG) and
    Original (OR).
    """)

    st.subheader("Technologies Used")

    technologies = {
        "Python": "Core programming language",
        "Pandas": "Dataset manipulation",
        "NumPy": "Numerical operations",
        "Scikit-learn": "Machine learning and TF-IDF",
        "Matplotlib": "Data visualization",
        "Seaborn": "Statistical visualization",
        "Streamlit": "Web application development"
    }

    for technology, description in technologies.items():
        st.write(f"**{technology}:** {description}")

    st.subheader("Machine Learning Algorithms")

    st.write("- Multinomial Naive Bayes")
    st.write("- Logistic Regression")

    st.subheader("Model Results")

    results = pd.DataFrame({
        "Algorithm": [
            "Naive Bayes",
            "Logistic Regression"
        ],
        "Accuracy": [
            "88.44%",
            "90.78%"
        ]
    })

    st.table(results)

    st.info(
        "The results measure prediction performance on "
        "the dataset's CG/OR labels. They do not establish "
        "real-world fraud detection accuracy."
    )

# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "ReviewGuard | BCA Data Mining Project | "
    "Machine Learning Based Text Classification"
)
