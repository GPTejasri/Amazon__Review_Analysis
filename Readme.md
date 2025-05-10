---

## ⚙️ Workflow Steps:

### 1. Preprocessing

- Load the dataset.
- Perform text cleaning, such as lowering casing, removing numbers and stopwords.
- Utilize NLTK/Spacy for tokenization and lemmatization.

- Fill in missing data, remove duplicate values.

### 2. Feature Engineering

- Convert text data into vectors via TF-IDF.
- Perform One-Hot Encoding for the categorical columns.
- Normalize the numeric data.

### 3. Clustering (KMeans)

- Implement KMeans to create 3 identified clusters; good, bad and neutral.
- Visualize the cluster’s data with PCA or t-SNE.
- Label the cluster designated reviews with the appropriate labels.

### 4. Sentiment Analysis (TextBlob)

- Determine polarity value for each particular review.

- Ranges score are as follows:
  - Anything greater than 0.1 becomes positive
  - Anything below -0.1 becomes negative
  - Everything else will be considered neutral

### 5. Classification (Logistic Regression)

- Train a model with sentiment targets using TF-IDF vectors.
- Sentiment categories will be predicted.
- Evaluate using: measuring accuracy, confusion matrix, precision and recall.

### 6. Topic Modeling (LDA)

- Derive topics through the clustered reviews.
- Present results within WordClouds and bar chart visuals.

---

## 💻 Streamlit Web Application
### Instructions to Run the Application Locally:
```bash
# Clone this repository.
git clone https://github.com/GPTejasri/Amazon_Review_Analysis.git
# Navigate to the directory
cd Amazon_Review_Analysis
# Install requirements
pip install -r requirements.txt
# Launch the application
streamlit run app.py

