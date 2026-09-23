# 💬 Product Review Sentiment Analysis

## End-to-End Machine Learning and Streamlit Deployment Project

An end-to-end Natural Language Processing (NLP) project that analyzes product reviews and classifies them into **Negative, Neutral, or Positive** sentiment.

The project covers the complete machine learning workflow:

**Data Preparation → Exploratory Data Analysis → Text Preprocessing → Feature Engineering → Model Training → Model Evaluation → Error Analysis → Pipeline Creation → Model Serialization → Streamlit Deployment**

---

# 📌 1. Project Overview

Online product reviews contain valuable information about customer experiences and opinions. Automatically identifying the sentiment of these reviews can help organizations understand customer feedback at scale.

This project develops a **three-class sentiment classification system** for product reviews.

The original product ratings are converted into three sentiment categories:

| Rating | Sentiment |
| ------ | --------- |
| 1–2    | Negative  |
| 3      | Neutral   |
| 4–5    | Positive  |

The project focuses on traditional machine learning techniques using **TF-IDF text representation**, **Logistic Regression**, and **LinearSVC**, followed by deployment through **Streamlit**.

---

# 🎯 2. Project Objectives

The main objectives of this project are:

1. Understand and explore the product review dataset.
2. Convert product ratings into three sentiment categories.
3. Perform exploratory data analysis on reviews and sentiment distribution.
4. Clean and preprocess review text.
5. Preserve important sentiment information such as negation.
6. Convert text into numerical features using TF-IDF.
7. Train multiple machine learning classifiers.
8. Compare model performance using appropriate evaluation metrics.
9. Investigate the difficulty of classifying Neutral reviews.
10. Perform error analysis on incorrectly classified reviews.
11. Build a reusable machine learning Pipeline.
12. Serialize the trained model using Joblib.
13. Develop an interactive Streamlit application.
14. Deploy the sentiment analysis application for real-world use.

---

# 📊 3. Dataset

The dataset contains product reviews and their corresponding product ratings.

The original rating values are mapped into three sentiment classes:

```text
Rating 1–2 → Negative
Rating 3   → Neutral
Rating 4–5 → Positive
```

## Dataset Distribution

The complete dataset contains **1,440 reviews**.

| Sentiment |   Reviews |
| --------- | --------: |
| Negative  |       512 |
| Neutral   |       199 |
| Positive  |       729 |
| **Total** | **1,440** |

The dataset is therefore not perfectly balanced.

The **Positive** class contains the largest number of reviews, while **Neutral** is the smallest class.

This imbalance was considered during model development and evaluation.

---

# 🔎 4. Exploratory Data Analysis

Exploratory Data Analysis was performed before model training to understand the structure and characteristics of the review data.

The analysis included:

* Dataset shape and structure
* Data types
* Missing-value inspection
* Rating distribution
* Sentiment distribution
* Review-length analysis
* Word-frequency analysis
* Bigram analysis
* Data-quality checks
* Basic language detection for exploratory purposes

## Review Length

The reviews vary considerably in length, ranging from short comments to detailed product experiences.

This variation is important because some reviews contain only a few sentiment-bearing words while others contain multiple opinions and product characteristics.

---

# 🧹 5. Text Preprocessing

The goal of preprocessing was to remove unnecessary noise while preserving information that could be useful for sentiment classification.

The main preprocessing steps were:

### 5.1 Convert text to string

Review values are converted to strings to ensure consistent processing.

### 5.2 Remove HTML tags

HTML tags are removed from review text.

Example:

```text
"This phone is <b>good</b>"
```

becomes approximately:

```text
"This phone is good"
```

### 5.3 Replace URLs

URLs are replaced with a special token:

```text
URL
```

rather than being removed completely.

### 5.4 Normalize whitespace

Multiple spaces and unnecessary whitespace are normalized.

### 5.5 Convert text to lowercase

Text is converted to lowercase so that words such as:

```text
Good
GOOD
good
```

are treated consistently.

---

# 🌍 6. Multilingual Text Investigation

Language detection was performed as part of exploratory analysis.

The dataset contains predominantly English reviews, along with a small number of reviews or fragments detected as other languages.

The project did **not** remove non-English reviews simply based on language detection.

This decision was made because automatic language detection can produce incorrect classifications, particularly for short reviews and code-mixed text.

Therefore, the modeling workflow preserves Unicode text rather than applying ASCII-only conversion.

---

# ⚙️ 7. Feature Extraction Using TF-IDF

Machine learning algorithms cannot directly process raw text.

Therefore, the reviews are converted into numerical representations using:

## TF-IDF

**TF-IDF (Term Frequency–Inverse Document Frequency)** assigns numerical importance to words based on their occurrence within individual reviews and across the collection of reviews.

The main TF-IDF configuration used in the project was:

```python
TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)
```

### N-grams

Both unigrams and bigrams are used.

For example:

```text
battery
battery life
camera
camera quality
good
good phone
```

This allows the model to capture short phrases rather than relying only on individual words.

---

# 🤖 8. Machine Learning Models

Several traditional machine learning approaches were evaluated.

The primary models were:

* Logistic Regression
* Linear Support Vector Classification (`LinearSVC`)

Additional feature and preprocessing experiments were also performed.

These included:

* Word-level TF-IDF
* Character-level TF-IDF
* Word + character TF-IDF
* Different n-gram ranges
* Class weighting
* Additional sentiment-count features
* Contraction expansion

The purpose of these experiments was to determine whether changes in representation or model configuration could improve three-class sentiment classification.

---

# 🧪 9. Train-Test Split

The dataset was divided into training and testing sets.

```text
Training samples: 1,152
Testing samples:    288
```

The test-set distribution was:

| Sentiment | Test Samples |
| --------- | -----------: |
| Negative  |          102 |
| Neutral   |           40 |
| Positive  |          146 |

A stratified split was used so that the sentiment distribution was maintained across the training and testing datasets.

---

# 📈 10. Model Evaluation

Because this is a multiclass classification problem with class imbalance, accuracy alone is not sufficient.

The following evaluation metrics were considered:

* Accuracy
* Precision
* Recall
* F1-score
* Macro F1
* Confusion Matrix

### Why Macro F1?

Macro F1 calculates the F1-score independently for each class and then averages them.

This gives equal importance to:

```text
Negative
Neutral
Positive
```

rather than allowing the largest class to dominate the overall metric.

---

# 🏆 11. Model Comparison

Several experiments were performed during development.

| Model / Experiment                          | Accuracy | Macro F1 | Neutral F1 |
| ------------------------------------------- | -------: | -------: | ---------: |
| Logistic Regression                         |     0.76 | **0.65** |   **0.29** |
| Logistic Regression with Trigrams           |     0.77 | **0.65** |   **0.29** |
| LinearSVC                                   | **0.79** |     0.63 |       0.20 |
| LinearSVC with Neutral Weight 1.5           | **0.80** |     0.63 |       0.19 |
| Word + Character TF-IDF                     |     0.75 |     0.63 |       0.25 |
| TF-IDF + Sentiment Features                 |     0.70 |     0.59 |       0.25 |
| Contraction Expansion + Logistic Regression |     0.77 |     0.64 |       0.24 |

The results demonstrate that higher accuracy does not necessarily mean better performance across all classes.

For example, the LinearSVC experiment achieved higher accuracy, while Logistic Regression produced stronger macro-level performance and better Neutral F1.

---

# 📌 12. Logistic Regression Results

The main Logistic Regression configuration achieved approximately:

```text
Accuracy: 0.76
Macro F1: 0.65
```

Classification results:

| Class                | Precision | Recall | F1-score | Support |
| -------------------- | --------: | -----: | -------: | ------: |
| Negative             |      0.78 |   0.81 |     0.80 |     102 |
| Neutral              |      0.31 |   0.28 |     0.29 |      40 |
| Positive             |      0.85 |   0.86 |     0.85 |     146 |
| **Overall Accuracy** |           |        | **0.76** | **288** |

The confusion matrix was:

```text
                 Predicted
              Negative Neutral Positive

Actual Negative    83      11       8
       Neutral     15      11      14
       Positive     8      13     125
```

The model performs considerably better on clearly Negative and Positive reviews than on Neutral reviews.

---

# ⚠️ 13. Neutral Class Challenge

The most important finding from the project is that **Neutral sentiment is substantially harder to classify**.

Only a small proportion of Neutral reviews were correctly classified compared with the Positive and Negative classes.

The reason is largely related to the nature of three-star reviews.

Many Neutral reviews are not purely neutral statements.

Instead, they contain:

* Positive and negative opinions together
* Qualified opinions
* Comparisons
* Product limitations
* Statements such as "okay", "average", or "fine"
* Mixed product experiences

Examples of this type of language include:

```text
good phone but performance is not smooth
```

```text
average phone for the price
```

```text
good battery but poor camera
```

```text
camera is okay but performance is slow
```

These reviews contain vocabulary associated with multiple sentiment classes.

Therefore, the Neutral class has a much less distinct linguistic boundary.

---

# 🔬 14. Neutral-Class Error Analysis

Error analysis was performed to understand why Neutral reviews were frequently misclassified.

The analysis showed that Neutral reviews can be broadly characterized as:

### Clearly Neutral

Reviews containing words such as:

```text
average
okay
fine
```

### Mostly Positive with Limitations

For example:

```text
Good phone but there are some issues.
```

### Mostly Negative with Some Positive Features

For example:

```text
Camera is good but the phone is slow.
```

### Mixed Sentiment

Reviews containing substantial positive and negative feedback.

This makes the classification problem more difficult than simply identifying individual positive or negative words.

---

# 📊 15. Probability Analysis

Logistic Regression probabilities were also examined for Neutral reviews.

For several incorrectly classified Neutral reviews, the predicted probabilities were relatively close between two classes.

For example, one Neutral review had approximately:

```text
Negative: 0.435
Neutral : 0.388
Positive: 0.177
```

Another had:

```text
Negative: 0.129
Neutral : 0.384
Positive: 0.487
```

These examples demonstrate that some Neutral reviews lie close to the decision boundary between sentiment classes.

Therefore, the errors are not always cases where the model completely ignores Neutral sentiment.

---

# 🧪 16. Additional Experiments

Several approaches were tested to determine whether Neutral classification could be improved.

## 16.1 Neutral Class Weighting

Increasing the importance of the Neutral class was tested.

However, this did not improve Neutral F1 and in some cases reduced Neutral recall.

Therefore, simply assigning a larger class weight was not sufficient.

---

## 16.2 Word + Character TF-IDF

Character-level TF-IDF features were combined with word-level TF-IDF.

The resulting performance did not improve sufficiently over the baseline.

Therefore, the additional character representation was not retained as the primary approach.

---

## 16.3 Additional Sentiment Features

Three manually designed features were added:

```text
positive word count
negative word count
neutral/mixed word count
```

The combined model performed worse than the original TF-IDF approach.

This demonstrated that manually counting sentiment words did not capture the contextual relationships present in mixed reviews.

---

## 16.4 Trigram Features

TF-IDF was extended from:

```text
(1, 2)
```

to:

```text
(1, 3)
```

This allowed the model to consider phrases such as:

```text
good phone but
not very good
```

The result was similar to the original Logistic Regression model and did not provide a substantial improvement.

---

## 16.5 Contraction Expansion

Common contractions were expanded:

```text
don't → do not
isn't → is not
can't → can not
it's → it is
```

Although this preserves explicit negation in some reviews, the experiment did not improve overall macro performance or Neutral F1 sufficiently.

Therefore, the original preprocessing strategy was retained.

---

# 🔄 17. Machine Learning Pipeline

A major part of the project is the use of a **Scikit-learn Pipeline**.

Conceptually, the prediction workflow is:

```text
                  Product Review
                        │
                        ▼
                Text Preprocessing
                        │
                        ▼
                  TF-IDF Vectorizer
                        │
                        ▼
                ML Classification Model
                        │
                        ▼
              Negative / Neutral / Positive
```

The Pipeline combines feature transformation and classification into one reusable object.

### Why use a Pipeline?

Without a Pipeline, the vectorizer and classifier would have to be handled separately.

With a Pipeline:

```text
Text → TF-IDF → Classifier → Prediction
```

becomes a single workflow.

This provides:

* Consistent preprocessing
* Reproducibility
* Easier model management
* Reduced risk of inconsistent transformations
* Simpler deployment

---

# 💾 18. Model Serialization

The trained model pipeline is saved using Joblib:

```text
sentiment_pipeline.joblib
```

This allows the Streamlit application to load the already-trained model instead of retraining the model whenever the application starts.

The saved model contains the trained machine learning workflow required for prediction.

---

# 🌐 19. Streamlit Application

A web application was developed using **Streamlit**.

The application provides a simple interface where users can enter a product review and obtain a sentiment prediction.

### Application Workflow

```text
User enters review
        ↓
Text preprocessing
        ↓
Saved ML pipeline
        ↓
TF-IDF transformation
        ↓
Classification
        ↓
Sentiment displayed
```

The application displays:

```text
😊 Positive
😐 Neutral
😞 Negative
```

depending on the prediction.

---

# 🚀 20. Live Application

The deployed application is available here:

**[Open Product Review Sentiment Analysis App](https://sentiment-analysis-3class-k79mw38u36u6pzpndb4tdz.streamlit.app/)**

---

# 📁 21. Project Structure

```text
sentiment-analysis-3class/
│
├── sentiment_analysis.ipynb
│       └── EDA, preprocessing, feature extraction,
│           model training, evaluation and experiments
│
├── sentimentapp.py
│       └── Streamlit application
│
├── sentiment_pipeline.joblib
│       └── Serialized trained ML pipeline
│
├── requirements.txt
│       └── Python dependencies
│
├── README.md
│       └── Project documentation
│
└── .gitignore
        └── Files excluded from version control
```

---

# 🛠️ 22. Technologies Used

| Technology          | Purpose                         |
| ------------------- | ------------------------------- |
| Python              | Programming language            |
| Pandas              | Data manipulation               |
| NumPy               | Numerical operations            |
| Matplotlib          | Data visualization              |
| Scikit-learn        | Machine learning and NLP        |
| TF-IDF              | Text feature extraction         |
| Logistic Regression | Sentiment classification        |
| LinearSVC           | Sentiment classification        |
| Joblib              | Model serialization             |
| Streamlit           | Web application and deployment  |
| Jupyter Notebook    | Development and experimentation |
| Git                 | Version control                 |
| GitHub              | Source code hosting             |

---

# 📦 23. Installation

Clone the repository:

```bash
git clone https://github.com/RAVANAM-SRIHARI/sentiment-analysis-3class.git
```

Navigate to the project directory:

```bash
cd sentiment-analysis-3class
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ 24. Run the Application Locally

Start the Streamlit application:

```bash
streamlit run sentimentapp.py
```

The application will open in a web browser.

---

# 🧪 25. Example Predictions

### Positive Review

```text
The phone has excellent battery life and a great camera.
```

Expected sentiment:

```text
😊 Positive
```

### Negative Review

```text
The phone is very slow and the camera quality is poor.
```

Expected sentiment:

```text
😞 Negative
```

### Neutral Review

```text
The phone is average for the price and the battery is okay.
```

Expected sentiment:

```text
😐 Neutral
```

These examples represent typical sentiment categories. Actual predictions depend on the trained model.

---

# 🔐 26. Data and Repository Considerations

The raw dataset is not included in the GitHub repository.

The `.gitignore` file excludes the dataset:

```text
dataset.csv
```

This keeps the repository focused on the project code, notebook, trained model, application, and docum
