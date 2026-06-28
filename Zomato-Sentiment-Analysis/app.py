import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import re
import string
from pathlib import Path

st.set_page_config(page_title="Zomato Review Intelligence", layout="wide")
BASE_DIR = Path(__file__).parent
st.markdown("""
<style>

.main {
    background-color: #fafafa;
}

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
}

h1,h2,h3{
    color:#E23744;
}

[data-testid="stSidebar"]{
    background:#ffffff;
    border-right:2px solid #f2f2f2;
}

div[data-testid="metric-container"]{
    background:white;
    border-radius:12px;
    padding:18px;
    box-shadow:0px 2px 8px rgba(0,0,0,0.08);
}

.stButton>button{
    background:#E23744;
    color:white;
    border:none;
    border-radius:10px;
    height:3em;
    font-weight:bold;
}

.stButton>button:hover{
    background:#c81d3a;
}

</style>
""", unsafe_allow_html=True)
# Load saved artifacts
@st.cache_resource
def load_artifacts():
    with open("sentiment_model.pkl", "rb") as f:
        sentiment_model = pickle.load(f)

    with open("tfidf_vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)

    with open("kmeans_model.pkl", "rb") as f:
        kmeans = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    restaurant_clusters = pd.read_csv("restaurant_clusters.csv")

    return sentiment_model, tfidf, kmeans, scaler, restaurant_clusters


# ⭐ ADD THIS LINE
sentiment_model, tfidf, kmeans, scaler, restaurant_clusters = load_artifacts()
if "history" not in st.session_state:
    st.session_state.history = []
# Text cleaning function (must match notebook preprocessing)
contractions_dict = {
    "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "can't": "cannot", "couldn't": "could not", "shouldn't": "should not",
    "wouldn't": "would not", "won't": "will not", "isn't": "is not",
    "aren't": "are not", "wasn't": "was not", "weren't": "were not",
    "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
    "i'm": "i am", "you're": "you are", "he's": "he is", "she's": "she is",
    "it's": "it is", "we're": "we are", "they're": "they are",
    "i've": "i have", "you've": "you have", "we've": "we have",
    "they've": "they have", "i'll": "i will", "you'll": "you will",
    "he'll": "he will", "she'll": "she will", "we'll": "we will",
    "they'll": "they will", "i'd": "i would", "you'd": "you would"
}

def clean_review(text):
    text = text.lower()
    for c, e in contractions_dict.items():
        text = text.replace(c, e)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Sidebar navigation
st.sidebar.image("assets/zomato_logo.png", width=170)

st.sidebar.markdown(
    """
# 🍽️ Zomato

### Review Intelligence Dashboard
"""
)

st.sidebar.markdown("---")

st.sidebar.markdown("## 🤖 Model")
st.sidebar.success("Multinomial Naive Bayes")

st.sidebar.markdown("## 🧠 Vectorizer")
st.sidebar.success("TF-IDF")

st.sidebar.markdown("## 📊 Clustering")
st.sidebar.success("K-Means")

st.sidebar.markdown("## ⚙️ Framework")
st.sidebar.success("Streamlit")

st.sidebar.markdown("---")

st.sidebar.success("✅ Model Loaded Successfully")
page = st.sidebar.radio(
    "📂 Navigation",
    [
        "📝 Sentiment Analysis",
        "🏪 Restaurant Explorer",
        "📊 Analytics Dashboard"
    ],
    index=0
)

if page == "📝 Sentiment Analysis":
    st.markdown("""
<div style='
background:linear-gradient(90deg,#E23744,#ff6b6b);
padding:25px;
border-radius:15px;
text-align:center;
box-shadow:0px 4px 12px rgba(0,0,0,0.2);
margin-bottom:20px;
'>

<h1 style='color:white;margin:0;'>
🍽️ Restaurant Review Sentiment Analysis
</h1>

<p style='color:white;font-size:18px;'>
Analyze customer reviews using NLP and Machine Learning
</p>

</div>
""", unsafe_allow_html=True)
    st.write("Enter a restaurant review below to predict its sentiment.")

    user_review = st.text_area("Type a review:", height=120, placeholder="e.g. The food was amazing and service was quick!")
    if st.button("Analyze Sentiment"):
        if user_review.strip() == "":
            st.warning("Please enter a review first.")

        else:
            cleaned = clean_review(user_review)
            vector = tfidf.transform([cleaned])

            prediction = sentiment_model.predict(vector)[0]
            probabilities = sentiment_model.predict_proba(vector)[0]
            classes = sentiment_model.classes_

            confidence = float(np.max(probabilities) * 100)

            if prediction == "Positive":
                st.success("😊 Positive Review")
            elif prediction == "Negative":
                st.error("😞 Negative Review")
            else:
                st.warning("😐 Neutral Review")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Prediction", prediction)

            with col2:
                st.metric("Confidence", f"{confidence:.2f}%")

            with col3:
                st.metric("Words", len(user_review.split()))

            st.subheader("Prediction Confidence")

            for cls, prob in zip(classes, probabilities):
                st.write(f"**{cls}**")
                st.progress(float(prob))

            new_prediction = {
                "Review": user_review,
                "Prediction": prediction,
                "Confidence": f"{confidence:.2f}%"
            }

            if (
                len(st.session_state.history) == 0
                or st.session_state.history[-1] != new_prediction
            ):
                st.session_state.history.append(new_prediction)

    st.markdown("---")
    st.subheader("🕒 Recent Predictions")

    if len(st.session_state.history) > 0:

        history_df = pd.DataFrame(st.session_state.history)

        st.dataframe(
            history_df.iloc[::-1],
            use_container_width=True,
            hide_index=True
        )

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🗑 Clear History"):
                st.session_state.history.clear()
                st.rerun()

        with col2:
            st.download_button(
                "📥 Download Prediction History",
                history_df.to_csv(index=False),
                "prediction_history.csv",
                "text/csv"
            )

    else:
        st.info("No predictions yet.")

elif page == "🏪 Restaurant Explorer":
    st.title("🏪 Restaurant Cluster Explorer")
    st.write("Explore the 4 restaurant segments identified through unsupervised clustering.")

    cluster_names = {
        0: 'Premium Excellence',
        1: 'Budget Decent',
        2: 'Premium Engaged',
        3: 'Underperformers'
    }
    restaurant_clusters['Cluster_Name'] = restaurant_clusters['Cluster'].map(cluster_names)

    selected_cluster = st.selectbox("Select a segment:", list(cluster_names.values()))
    search = st.text_input(
        "🔍 Search Restaurant",
        placeholder="Type restaurant name...",
        key="restaurant_search"
    )

    filtered = restaurant_clusters[
    restaurant_clusters["Cluster_Name"] == selected_cluster
]

    if search.strip():
     filtered = filtered[
        filtered["Name"].str.contains(
            search,
            case=False,
            na=False
        )
    ]

    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Rating", f"{filtered['Avg_Rating'].mean():.2f}")
    col2.metric("Avg Sentiment", f"{filtered['Avg_Sentiment'].mean():.2f}")
    col3.metric("Avg Cost", f"₹{filtered['Cost'].mean():.0f}")

    st.dataframe(filtered[['Name', 'Avg_Rating', 'Avg_Sentiment', 'Cost', 'Avg_Review_Length', 'Avg_Follower_Count']].reset_index(drop=True))

    st.subheader("All Clusters Comparison")
    summary = restaurant_clusters.groupby('Cluster_Name')[['Avg_Rating', 'Avg_Sentiment', 'Cost', 'Avg_Follower_Count']].mean()
    st.bar_chart(summary)

elif page == "📊 Analytics Dashboard":

    st.title("📊 Analytics Dashboard")

    st.write("Overall statistics of the restaurant dataset.")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Restaurants",
        len(restaurant_clusters)
    )

    c2.metric(
        "Average Rating",
        f"{restaurant_clusters['Avg_Rating'].mean():.2f}"
    )

    c3.metric(
        "Average Cost",
        f"₹{restaurant_clusters['Cost'].mean():.0f}"
    )

    c4.metric(
        "Average Sentiment",
        f"{restaurant_clusters['Avg_Sentiment'].mean():.2f}"
    )

    st.markdown("---")

    st.subheader("📊 Cluster Distribution")

    cluster_count = (
        restaurant_clusters["Cluster_Name"]
        .value_counts()
        .reset_index()
    )

    cluster_count.columns = ["Cluster", "Count"]

    fig = px.pie(
        cluster_count,
        names="Cluster",
        values="Count",
        title="Restaurant Cluster Distribution",
        hole=0.4
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("⭐ Rating Distribution")

    fig = px.histogram(
        restaurant_clusters,
        x="Avg_Rating",
        nbins=20,
        title="Restaurant Rating Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("💰 Cost Distribution")

    fig = px.histogram(
        restaurant_clusters,
        x="Cost",
        nbins=20,
        title="Restaurant Cost Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("🏆 Restaurant Rankings")

ranking_option = st.selectbox(
    "Choose Ranking",
    [
        "⭐ Highest Rated",
        "💰 Lowest Cost",
        "👥 Most Popular"
    ]
)

if ranking_option == "⭐ Highest Rated":
    top10 = restaurant_clusters.sort_values(
        "Avg_Rating",
        ascending=False
    ).head(10)

elif ranking_option == "💰 Lowest Cost":
    top10 = restaurant_clusters.sort_values(
        "Cost",
        ascending=True
    ).head(10)

else:
    top10 = restaurant_clusters.sort_values(
        "Avg_Follower_Count",
        ascending=False
    ).head(10)

st.dataframe(
    top10[
        [
            "Name",
            "Avg_Rating",
            "Cost",
            "Avg_Sentiment",
            "Avg_Follower_Count"
        ]
    ],
    use_container_width=True,
    hide_index=True
)
st.markdown("---")

st.markdown(
    """
    <div style='text-align:center; font-size:16px; color:#808080;'>
        🍽️ Zomato Review Intelligence Dashboard <br><br>
        <b>Developed by Chaitanya Virigineni</b>
    </div>
    """,
    unsafe_allow_html=True
)
