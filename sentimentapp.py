import streamlit as st
import joblib

# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="💬"
)


# ============================================
# LOAD TRAINED MODEL
# ============================================

@st.cache_resource
def load_model():
    return joblib.load("sentiment_pipeline.joblib")


model = load_model()


# ============================================
# APPLICATION TITLE
# ============================================

st.title("💬 Product Review Sentiment Analysis")

st.write(
    "Enter a product review to predict whether "
    "the sentiment is Negative, Neutral, or Positive."
)


# ============================================
# USER INPUT
# ============================================

review = st.text_area(
    "Enter your review:",
    placeholder=(
        "Example: The phone has excellent battery "
        "life and a great camera."
    )
)


# ============================================
# PREDICT SENTIMENT
# ============================================

if st.button("Analyze Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review.")

    else:

        prediction = model.predict([review])[0]

        st.subheader("Prediction")


        # ========================================
        # DISPLAY SENTIMENT
        # ========================================

        if prediction == "positive":

            st.success("😊 Positive")

        elif prediction == "negative":

            st.error("😞 Negative")

        elif prediction == "neutral":

            st.info("😐 Neutral")

        else:

            st.warning(f"Predicted class: {prediction}")


        # ========================================
        # PREDICTION PROBABILITIES
        # ========================================

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba([review])[0]

            classes = model.classes_

            st.subheader("Prediction Probabilities")

            for class_name, probability in zip(
                classes,
                probabilities
            ):

                st.write(
                    f"{class_name.capitalize()}: "
                    f"{probability:.2%}"
                )