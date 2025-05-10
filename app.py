import streamlit as st
import pandas as pd
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

# --- Streamlit Page Setup ---
st.set_page_config(page_title="Amazon Sentiment Classifier", layout="centered")
st.title("📦 Amazon Product Review Sentiment Analysis")
st.markdown("Upload a CSV file containing reviews and analyze sentiment using Naive Bayes and SVM.")

# --- File Upload and Model Training ---
uploaded_file = st.file_uploader("📁 Upload your CSV file", type=["csv"])
models_trained = False
tfidf = None
nb_model = None
svm_model = None
X_test = None
y_test = None

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        st.success("✅ File uploaded successfully!")
        st.write("### Preview of Data:")
        st.dataframe(df.head())

        # --- Identify Review Column ---
        review_column = None
        for col in df.columns:
            if any(key in col.lower() for key in ['review', 'review_body', 'text']):
                review_column = col
                break

        if review_column is None:
            st.error("❌ No review-related column found in uploaded CSV. Please include 'review', 'review_body', or 'Text'.")
            st.stop()

        # --- Preprocessing ---
        df = df[[review_column]].dropna()
        df = df.rename(columns={review_column: "review"})

        # --- Sentiment Analysis ---
        df['sentiment'] = df['review'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
        df['sentiment'] = df['sentiment'].apply(lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral'))

        # --- Vectorization ---
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        X = tfidf.fit_transform(df['review'].astype(str))
        y = df['sentiment'].map({'positive': 2, 'neutral': 1, 'negative': 0})

        # --- Train/Test Split ---
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # --- Naive Bayes ---
        nb_model = MultinomialNB()
        nb_model.fit(X_train, y_train)

        # --- SVM ---
        svm_model = SVC(kernel='linear')
        svm_model.fit(X_train, y_train)

        # --- Model Evaluation ---
        st.subheader("📊 Model Performance Metrics")

        # Predictions for evaluation
        nb_pred = nb_model.predict(X_test)
        svm_pred = svm_model.predict(X_test)

        # Sentiment mapping for display
        sentiment_map = {2: 'Positive', 1: 'Neutral', 0: 'Negative'}

        # Naive Bayes Metrics
        st.write("### Naive Bayes Metrics")
        st.write(f"**Accuracy:** {accuracy_score(y_test, nb_pred):.2f}")
        st.write(f"**Precision:** {precision_score(y_test, nb_pred, average='weighted'):.2f}")
        st.write(f"**Recall:** {recall_score(y_test, nb_pred, average='weighted'):.2f}")
        st.write(f"**F1-Score:** {f1_score(y_test, nb_pred, average='weighted'):.2f}")
        st.write("**Confusion Matrix:**")
        cm_nb = confusion_matrix(y_test, nb_pred)
        # Display confusion matrix as a dataframe for better readability
        cm_nb_df = pd.DataFrame(cm_nb, index=sentiment_map.values(), columns=sentiment_map.values())
        st.write(cm_nb_df)

        # SVM Metrics
        st.write("### SVM Metrics")
        st.write(f"**Accuracy:** {accuracy_score(y_test, svm_pred):.2f}")
        st.write(f"**Precision:** {precision_score(y_test, svm_pred, average='weighted'):.2f}")
        st.write(f"**Recall:** {recall_score(y_test, svm_pred, average='weighted'):.2f}")
        st.write(f"**F1-Score:** {f1_score(y_test, svm_pred, average='weighted'):.2f}")
        st.write("**Confusion Matrix:**")
        cm_svm = confusion_matrix(y_test, svm_pred)
        # Display confusion matrix as a dataframe for better readability
        cm_svm_df = pd.DataFrame(cm_svm, index=sentiment_map.values(), columns=sentiment_map.values())
        st.write(cm_svm_df)

        models_trained = True

    except Exception as e:
        st.error(f"❌ Error reading the file: {e}")

# --- Review Input (Always Visible) ---
st.subheader("📝 Try Your Own Review")
user_review = st.text_area("Enter a product review here:", "")

if user_review:
    if not models_trained:
        st.warning("⚠️ Please upload a CSV file and train the models first.")
    else:
        input_vector = tfidf.transform([user_review])
        nb_result = nb_model.predict(input_vector)[0]
        svm_result = svm_model.predict(input_vector)[0]
        sentiment_map = {2: 'Positive 😊', 1: 'Neutral 😐', 0: 'Negative 😠'}

        st.write("### Results")
        st.write(f"**Naive Bayes Prediction:** {sentiment_map[nb_result]}")
        st.write(f"**SVM Prediction:** {sentiment_map[svm_result]}")

if uploaded_file is None:
    st.info("📂 Please upload a CSV file containing product reviews.")
    