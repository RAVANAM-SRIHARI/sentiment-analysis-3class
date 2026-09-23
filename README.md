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
* Convert text to A
