# 💬 Product Review Sentiment Analysis

A Machine Learning project that classifies product reviews into three sentiment categories:

* 🔴 **Negative**
* 🟡 **Neutral**
* 🟢 **Positive**

The project covers the complete machine learning workflow, from data exploration and text preprocessing to model training, evaluation, model serialization, and deployment using **Streamlit**.

---

## 🚀 Live Demo

Try the deployed application:

**[Product Review Sentiment Analysis — Streamlit App](https://sentiment-analysis-3class-k79mw38u36u6pzpndb4tdz.streamlit.app/)**

Enter a product review and the application predicts whether the review is Negative, Neutral, or Positive.

---

## 📌 Project Overview

Customer reviews contain valuable information about how users perceive a product. Sentiment analysis can be used to automatically identify the overall sentiment expressed in these reviews.

In this project, product ratings are converted into three sentiment classes:

| Rating | Sentiment |
| ------ | --------- |
| 1–2    | Negative  |
| 3      | Neutral   |
| 4–5    | Positive  |

The project uses **TF-IDF** to convert textual reviews into numerical features and compares multiple traditional machine learning approaches for multiclass classification.

---

## 🎯 Objectives

The main objectives of this project are to:

* Explore and understand product review data.
* Analyze rating and sentiment distributions.
* Perform text preprocessing and cleaning.
* Convert text into numerical features using TF-IDF.
* Train and compare multiple machine learning models.
* Evaluate model performance using accuracy, precision, recall, F1-score, and confusion matrices.
* Analyze errors, particularly for the Neutral class.
* Build a reusable machine learning pipeline.
* Save the trained model using Joblib.
* Deploy the sentiment analysis application using Streamlit.

---

## 📊 Dataset

The dataset contains product reviews along with their corresponding ratings.

The original ratings are mapped into three sentiment classes:

```text
1–2 → Negative
3   → Neutral
4–5 → Positive
```

### Class Distribution

| Sentiment | Number of Reviews |
| --------- | ----------------: |
| Negative  |               512 |
| Neutral   |               199 |
| Positive  |               729 |
| **Total** |         **1,440** |

The dataset is not perfectly balanced, with Positive reviews representing the largest class and Neutral reviews representing the smallest class.

---

## 🔎 Exploratory Data Analysis

The exploratory analysis includes:

* Dataset structure and information
* Missing-value inspection
* Rating distribution
* Sentiment distribution
* Review length analysis
* Word-frequency analysis
* Bigram analysis
* Data-quality checks
* Basic language detection for exploratory purposes

Some common words and phrases were examined separately across sentiment classes to understand patterns in the reviews.

The analysis also showed that Neutral reviews are often more ambiguous than clearly positive or negative reviews. Many Neutral reviews contain a mixture of positive and negative opinions.

---

## 🧹 Text Preprocessing

The project uses relatively light text preprocessing to avoid removing useful sentiment information.

The preprocessing includes:

* Converting text to string
* Removing HTML tags
* Replacing URLs with a special `URL` token
* Normalizing whitespace
* Converting text to lowercase
* Removing leading and trailing spaces

### Important preprocessing decisions

The project intentionally does **not**:

* Remove the word `not`
* Convert text to ASCII
* Remove non-English Unicode characters
* Aggressively remove stopwords
* Perform aggressive stemming
* Perform aggressive lemmatization
* Remove emojis

This helps preserve information that can be useful for sentiment classification.

---

## 🔤 Feature Extraction

### TF-IDF

**Term Frequency–Inverse Document Frequency (TF-IDF)** is used to convert text into numerical feature vectors.

The main configuration includes:

```python
TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)
```

Both unigrams and bigrams are considered.

For example:

```text
good
phone
good phone
battery
battery life
```

This allows the model to use both individual words and short phrases.

---

## 🤖 Machine Learning Models

Several models and feature configurations were evaluated during experimentation.

### Models tested

* Logistic Regression
* Linear Support Vector Machine (`LinearSVC`)

### Additional experiments

The project also evaluated:

* TF-IDF with different n-gram ranges
* Word + character TF-IDF
* Class weighting
* Additional sentiment-count features
* Contraction expansion
* Model probability analysis
* Neutral-class error analysis

These experiments were used to understand which approaches improved performance and which did not.

---

## 📈 Model Evaluation

The dataset was divided into training and testing sets.

```text
Training samples: 1,152
Testing samples:    288
```

The test set contains:

| Sentiment | Test Samples |
| --------- | -----------: |
| Negative  |          102 |
| Neutral   |           40 |
| Positive  |          146 |

### Logistic Regression

The main Logistic Regression configuration achieved approximately:

```text
Accuracy : 0.76
Macro F1 : 0.65
```

Classification results:

| Class    | Precision | Recall | F1-score |
| -------- | --------: | -----: | -------: |
| Negative |      0.78 |   0.81 |     0.80 |
| Neutral  |      0.31 |   0.28 |     0.29 |
| Positive |      0.85 |   0.86 |     0.85 |

### LinearSVC

A LinearSVC configuration achieved:

```text
Accuracy : 0.79
Macro F1 : 0.63
```

LinearSVC achieved higher overall accuracy in this experiment, while Logistic Regression provided a slightly higher macro F1 and better Neutral-class F1.

---

## ⚠️ Neutral Class Analysis

One of the main findings of the project is that **Neutral sentiment is significantly more difficult to classify** than Positive and Negative sentiment.

Many Neutral reviews contain mixed opinions such as:

```text
"good phone but performance is not smooth"

"camera is good but battery could be better"

"average phone for the price"

"good battery but poor camera"
```

Because these reviews contain both positive and negative language, the boundary between the three classes is not always clear.

Several experiments were performed to improve Neutral classification, including:

* Increasing Neutral class weight
* Adding sentiment-count features
* Adding character-level TF-IDF
* Using trigrams
* Expanding contractions

These experiments did not produce a substantial improvement in Neutral performance.

This demonstrates an important practical machine learning lesson: **model performance is influenced not only by the algorithm but also by the characteristics and ambiguity of the underlying data.**

---

## 🔬 Error Analysis

Error analysis was performed to understand why Neutral reviews were frequently misclassified.

The analysis showed that many Neutral reviews fall between clearly positive and clearly negative sentiment.

For example, a review may contain:

```text
Positive opinion + Negative opinion
```

rather than expressing a single sentiment.

This makes Neutral classification inherently more challenging for a traditional TF-IDF-based classifier.

---

## 🔄 Machine Learning Pipeline

The final machine learning workflow uses a `Pipeline` to combine text feature extraction and classification.

Conceptually:

```text
User Review
     ↓
Text Preprocessing
     ↓
TF-IDF Vectorization
     ↓
Machine Learning Classifier
     ↓
Sentiment Prediction
```

Using a pipeline ensures that the same TF-IDF transformation learned during training is applied when new reviews are submitted.

It also allows the complete prediction workflow to be saved as a single Joblib file.

---

## 💾 Model Serialization

The trained model is saved using:

```text
sentiment_pipeline.joblib
```

This allows the Streamlit application to load the already-trained model instead of retraining it every time the application starts.

---

## 🌐 Streamlit Deployment

The application is built using **Streamlit**.

The application allows users to:

1. Enter a product review.
2. Click **Analyze Sentiment**.
3. Receive the predicted sentiment.
4. View the result as Negative, Neutral, or Positive.

The application loads the serialized machine learning model using Joblib.

---

## 📁 Project Structure

```text
sentiment-analysis-3class/
│
├── sentiment_analysis.ipynb
├── sentimentapp.py
├── sentiment_pipeline.joblib
├── requirements.txt
├── README.md
└── .gitignore
```

### File Description

| File                        | Description                                                                            |
| --------------------------- | -------------------------------------------------------------------------------------- |
| `sentiment_analysis.ipynb`  | Complete data analysis, preprocessing, model training, evaluation, and experimentation |
| `sentimentapp.py`           | Streamlit web application                                                              |
| `sentiment_pipeline.joblib` | Serialized trained machine learning pipeline                                           |
| `requirements.txt`          | Python dependencies required for the project                                           |
| `README.md`                 | Project documentation                                                                  |
| `.gitignore`                | Files excluded from Git version control                                                |

---

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Scikit-learn**
* **NLTK / LangDetect** for selected text-analysis tasks
* **Joblib**
* **Streamlit**
* **Jupyter Notebook**
* **Git & GitHub**

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/RAVANAM-SRIHARI/sentiment-analysis-3class.git
```

Navigate to the project directory:

```bash
cd sentiment-analysis-3class
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Streamlit Application

Run:

```bash
streamlit run sentimentapp.py
```

The application will open in your browser.

---

## 🧪 Example Reviews

### Positive

```text
The phone has excellent battery life and a great camera.
```

Expected sentiment:

```text
😊 Positive
```

### Negative

```text
The phone is very slow and the camera quality is poor.
```

Expected sentiment:

```text
😞 Negative
```

### Neutral

```text
The phone is average for the price and the battery is okay.
```

Expected sentiment:

```text
😐 Neutral
```

The examples above illustrate the intended behavior of the application; actual predictions depend on the trained model.

---

## 📚 Key Learning Outcomes

This project provided practical experience with:

* Text preprocessing
* Exploratory Data Analysis
* TF-IDF feature extraction
* N-gram features
* Multiclass classification
* Logistic Regression
* LinearSVC
* Hyperparameter tuning
* Class weighting
* Confusion matrix analysis
* Classification reports
* Error analysis
* Scikit-learn Pipelines
* Model serialization
* Streamlit application development
* GitHub project management
* Machine learning deployment

---

## 🔮 Future Improvements

Potential future improvements include:

* Increasing the size and diversity of the dataset
* Improving the quality and consistency of Neutral labels
* Using more advanced NLP representations
* Experimenting with transformer-based models
* Handling multilingual and code-mixed reviews more systematically
* Improving spelling and informal-text normalization
* Collecting additional real-world product reviews
* Adding prediction confidence visualization to the application

---

## 👤 Author

**Srihari Ravanam**

GitHub:
https://github.com/RAVANAM-SRIHARI

---

## ⭐ Project Summary

This project demonstrates an end-to-end **3-class product review sentiment analysis system**, starting from raw review data and exploratory analysis and progressing through text preprocessing, TF-IDF feature extraction, machine learning model comparison, error analysis, model serialization, and Streamlit deployment.

The project also highlights an important aspect of real-world NLP: **ambiguous and mixed-sentiment reviews can be substantially harder to classify than clearly positive or negative reviews.**

```

This version is more suitable for a **GitHub portfolio/college project** because it documents not just the final model, but also your **experimentation, evaluation, error analysis, and deployment workflow**.
```
