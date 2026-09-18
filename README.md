# NLP Sentiment Analysis

## Project Overview

This project develops a machine-learning sentiment analysis system
for customer product reviews.

Reviews are classified into three sentiment categories:

- Negative
- Neutral
- Positive

## Sentiment Mapping

| Rating | Sentiment |
|--------|-----------|
| 1-2 | Negative |
| 3 | Neutral |
| 4-5 | Positive |

## Workflow

The project follows this workflow:

1. Data loading
2. Data inspection
3. Rating distribution analysis
4. Sentiment creation
5. Text length analysis
6. Word count analysis
7. Noise investigation
8. Multilingual text investigation
9. Text preprocessing
10. Word-frequency analysis
11. Train/test split
12. TF-IDF feature extraction
13. Logistic Regression
14. Linear SVM
15. Multinomial Naive Bayes
16. Cross-validation
17. Model comparison
18. Error analysis
19. Final model selection
20. Streamlit deployment

## Preprocessing

The project uses light preprocessing.

HTML tags and URLs are handled, whitespace is normalized,
and text is converted to lowercase.

Aggressive removal of stopwords, Unicode characters,
and sentiment-related information is avoided.

## Models

The following machine-learning algorithms are compared:

- Logistic Regression
- Linear SVM
- Multinomial Naive Bayes

Models are selected using stratified cross-validation
with macro F1 as the primary model-selection metric.

## Evaluation

The final model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Macro F1
- Confusion Matrix

## Deployment

The final trained pipeline is saved as:

`sentiment_pipeline.joblib`

The application is built using Streamlit.

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt