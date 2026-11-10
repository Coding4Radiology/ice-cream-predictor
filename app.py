import os

import pandas as pd
import psycopg2
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline


# --- DATABASE CONFIGURATION ---
def get_setting(name, default):
    """Read a setting from Streamlit secrets or an environment variable."""
    try:
        return st.secrets.get(name, os.getenv(name, default))
    except Exception:
        # Allow the app to run with sample data before secrets are configured.
        return os.getenv(name, default)


DB_HOST = get_setting("DB_HOST", "localhost")
DB_NAME = get_setting("DB_NAME", "postgres")
DB_USER = get_setting("DB_USER", "postgres")
DB_PASS = get_setting("DB_PASS", "password")
DB_PORT = get_setting("DB_PORT", "5432")


def get_db_connection():
    try:
        return psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASS,
            port=DB_PORT,
        )
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return None


def init_db():
    conn = get_db_connection()
    if conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ice_cream_votes (
                    id SERIAL PRIMARY KEY,
                    username VARCHAR(100) NOT NULL,
                    favorite_flavor VARCHAR(100) NOT NULL
                );
            """)
            conn.commit()
        conn.close()


def save_to_db(name, flavor):
    conn = get_db_connection()
    if conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO ice_cream_votes (username, favorite_flavor) VALUES (%s, %s);",
                (name, flavor),
            )
            conn.commit()
        conn.close()


def load_data_from_db():
    conn = get_db_connection()
    if conn:
        df = pd.read_sql_query(
            "SELECT username, favorite_flavor FROM ice_cream_votes;", conn
        )
        conn.close()
        return df

    # Fallback sample data if a database connection is not configured.
    return pd.DataFrame([
        {"username": "Alice", "favorite_flavor": "Vanilla"},
        {"username": "Bob", "favorite_flavor": "Chocolate"},
        {"username": "Charlie", "favorite_flavor": "Strawberry"},
        {"username": "Danielle", "favorite_flavor": "Vanilla"},
        {"username": "Eric", "favorite_flavor": "Chocolate"},
    ])


# Initialize the database table when the database is available.
try:
    init_db()
except Exception:
    pass


# --- STREAMLIT UI ---
st.title("🍦 Ice Cream Flavor Predictor & Storage")
st.write(
    "Enter your details to save your favorite flavor and let AI guess "
    "what you like based on your name!"
)

with st.form("user_form"):
    name_input = st.text_input("Your Name", placeholder="e.g., Alex")
    flavor_input = st.selectbox(
        "Favorite Ice Cream Flavor",
        [
            "Chocolate",
            "Vanilla",
            "Strawberry",
            "Mint Chocolate Chip",
            "Cookie Dough",
            "Rocky Road",
        ],
    )
    submitted = st.form_submit_button("Save Preference")

if submitted:
    if name_input.strip():
        try:
            save_to_db(name_input, flavor_input)
            st.success(
                f"Successfully saved {name_input}'s favorite flavor ({flavor_input})!"
            )
        except Exception:
            st.warning(
                "Could not save to PostgreSQL. Configure the database secrets "
                "for persistent cloud storage."
            )
    else:
        st.error("Please enter a valid name.")


# --- MACHINE LEARNING MODEL ---
df_data = load_data_from_db()

if not df_data.empty and len(df_data) >= 2:
    st.subheader("🔮 AI Favorite Flavor Predictor")
    predict_name = st.text_input(
        "Enter a name to predict their favorite ice cream:", placeholder="e.g., Sam"
    )

    if predict_name:
        X = df_data["username"]
        y = df_data["favorite_flavor"]
        model = make_pipeline(
            TfidfVectorizer(analyzer="char", ngram_range=(2, 4)),
            MultinomialNB(),
        )
        model.fit(X, y)
        prediction = model.predict([predict_name])[0]
        st.write(
            f"Based on our dataset, the AI predicts that **{predict_name}** "
            f"loves **{prediction}**! 🍨"
        )
else:
    st.info("Add more entries to train the Machine Learning prediction model!")

if st.checkbox("Show Saved Records"):
    st.dataframe(df_data)
