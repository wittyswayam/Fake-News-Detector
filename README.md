## 📰 Fake News Detector: SVC with TF-IDF Vectorization

This repository contains a machine learning project designed to automatically classify news articles as either **"Real"** or **"Fake"** (Scam). The solution employs traditional **Natural Language Processing (NLP)** techniques and a powerful **Support Vector Classifier (SVC)** to analyze text data and predict its veracity.

The project is complete with a training pipeline in a Jupyter Notebook and a deployable interactive web application built with **Streamlit**.

-----

### ✨ Key Technical Features

#### 1\. Text Feature Extraction: TF-IDF Vectorizer

To make text understandable to a machine learning model, it must be converted into numerical features. This project uses the **Term Frequency-Inverse Document Frequency (TF-IDF) Vectorizer**.

  * **TF-IDF** weighs the importance of a word based on how often it appears in a single document (Term Frequency, TF) versus how unique it is across the entire dataset (Inverse Document Frequency, IDF).
  * The implementation excludes common **English stop words** (e.g., "the," "a," "is") to focus on more meaningful, distinctive terms.

#### 2\. Chosen Model: Support Vector Classifier (SVC)

After comparing multiple classification algorithms, the **Support Vector Classifier (SVC)** was selected as the final model due to its superior performance.

  * SVC works by finding the optimal **hyperplane** that maximizes the margin of separation between the two classes (Real and Fake) in the high-dimensional feature space created by the TF-IDF vectorizer. This boundary is defined by the "support vectors," which are the data points closest to the hyperplane.

#### 3\. Interactive Web Application (Streamlit)

The trained SVC model and the fitted TF-IDF Vectorizer are persisted (`model.pkl` and `tfidf.pkl`) and loaded by an application built using the **Streamlit** framework. This provides a clean, user-friendly interface where users can paste text and receive an immediate classification, allowing for easy demonstration and practical use of the model.

-----

### 📊 Dataset and Data Processing

#### Dataset Source

The model was trained on a sampled portion of the **WELFAKE\_Dataset.csv**. This dataset contains text columns (title and body `text`) and a binary `label` (0 or 1).

#### Data Sampling and Balancing

The full dataset of 72,134 records was sampled down to **5,000 records** to facilitate faster training and model experimentation. The sampling process ensured the classes were well-balanced, with approximately 2,500 records for each class, preventing bias toward one category.

#### Text Cleaning Pipeline

A critical step is text cleaning to ensure consistent input for the model. The `clean_text` function performs three operations:

1.  **Lowercasing:** Converts all text to lowercase.
2.  **Punctuation/Special Character Removal:** Removes any characters that are not a-z, 0-9, or whitespace.
3.  **Whitespace Normalization:** Removes any extra spaces.

-----

### 📈 Model Training and Performance

The training process involved comparing three different classifiers on the TF-IDF transformed data.

| Model | Accuracy (on Test Set) | Macro Average F1-Score |
| :--- | :--- | :--- |
| **SVC** | **0.92** | **0.92** |
| Logistic Regression | 0.91 | 0.91 |
| Random Forest | 0.89 | 0.88 |

The **SVC model** achieved the highest overall performance, demonstrating a **92% accuracy** and a high, balanced F1-score, indicating excellent performance across both the "Real" and "Fake" classes.

-----

### 🚀 Installation and Usage

#### Prerequisites

You will need Python 3 and the dependencies listed in `requirements.txt`.

```bash
# Clone the repository
git clone https://github.com/wittyswayam/Fake-News-Detector.git
cd Fake-News-Detector

# Install dependencies
pip install -r requirements.txt
```

#### Running the Streamlit App

To launch the interactive web detector, simply run the `app.py` file:

```bash
streamlit run app.py
```

-----

### 💡 Future Works and Enhancements

#### 1\. Integration of Deep Learning Models

While SVC with TF-IDF performs well, it primarily captures keyword importance. A significant improvement could be achieved by integrating a **Deep Learning model** (such as a **Recurrent Neural Network (RNN)**, **Long Short-Term Memory (LSTM)**, or a **Transformer-based model like BERT**). Deep Learning models capture the **semantic context, grammatical structure, and sequential relationship** between words, allowing for a more nuanced understanding of text deception.

#### 2\. Advanced Text Preprocessing (Stop Word Removal and Lemmatization)

  * **Custom Stop Word Lists:** Tailoring the stop words to the domain (e.g., common political terms that might not be standard stop words but don't add value).
  * **Stemming or Lemmatization:** Reducing words to their root form (e.g., "running," "ran," "runs" to "run"). This reduces the feature space and helps the vectorizer treat variants of the same word consistently.
