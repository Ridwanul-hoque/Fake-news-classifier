# Fake News Classification & Gradio Application

## Overview
This project builds, evaluates, and deploys a machine learning model designed to classify news articles as **Real** or **Fake**. Using natural language processing (NLP) techniques and scikit-learn, various model pipelines were trained and evaluated via cross-validation to find the optimal balance between performance and stability. The final trained pipeline is integrated into an interactive web interface built with Gradio for real-time predictions.

## Dataset
- **Input:** Text (News Article Content / Headlines)
- **Target:** label (`Fake` / `Real`)
- **Total samples:** 44,898
- **Class distribution:** 23,481 Fake / 21,417 Real

## Key EDA Findings
- **Publisher Datelines:** Real news articles consistently feature formal publisher credits and location markers (e.g., *"WASHINGTON (Reuters)"*), whereas Fake news articles lack consistent wire service tags.
- **Sensational Vocabulary:** Fake news titles and bodies rely heavily on emotional adjectives, capitalized clickbait cues (e.g., *"BOMBSHELL:"*), and subjective commentary.
- **Length Distribution:** Word count distributions across classes show slight variations, but text structure and vocabulary choice serve as significantly stronger predictive signals than document length alone.

## Text Representation
- **TF-IDF Vectorizer (`TfidfVectorizer`):** Converts raw text into numerical feature vectors by scoring word frequency adjusted for document rarity across the corpus.
- **Preprocessing:** Built directly into the scikit-learn pipeline to automatically handle tokenization, lowercasing, and term weighting without manual data transformations during inference.

```

---

Full Stack AI & Data Science — PROJECT 02

## Cross-Validation Model Comparison

| Pipeline | CV Accuracy | CV Macro F1 |
| --- | --- | --- |
| CountVectorizer + MultinomialNB | 0.9520 ± 0.0031 | 0.9515 ± 0.0032 |
| TF-IDF + MultinomialNB | 0.9385 ± 0.0040 | 0.9379 ± 0.0041 |
| **TF-IDF + Logistic Regression** | **0.9865 ± 0.0018** | **0.9863 ± 0.0019** |

## Final Model

* **Pipeline:** `TfidfVectorizer()` + `LogisticRegression()`
* **Test Accuracy:** 0.9872
* **Test Precision:** 0.9870
* **Test Recall:** 0.9874
* **Test F1-Score:** 0.9872

### Why this model?

The **TF-IDF + Logistic Regression** pipeline achieved the highest mean accuracy and F1-score across 5-fold cross-validation while exhibiting the lowest standard deviation (highest stability). Logistic Regression handles high-dimensional, sparse TF-IDF feature matrices exceptionally well without overfitting, outperforming naive Bayes baselines on subtle contextual cues.

## Web Application

The interface is built with Gradio and can be run locally or embedded inside a Jupyter/Kaggle notebook environment.

**Optional live app:** [Hosted Hugging Face Space / Local Link]

### Screenshots

## Installation

```bash
git clone [https://github.com/kawchar-husain/fake-news-classifier.git](https://github.com/kawchar-husain/fake-news-classifier.git)
cd fake-news-classifier
pip install -r requirements.txt

```

## Usage

Run the web app locally:

```bash
python app.py

```

## Project Structure

```text
fake-news-classifier/
│-- data/
│   └── fake_and_real_news.csv
│-- notebooks/
│   ├── 1_eda.ipynb
│   └── 2_training.ipynb
│-- app.py
│-- models/
│   └── best_model.pkl
│-- screenshots/
│   └── gradio_interface.png
│-- README.md
└── requirements.txt

```

## Technologies Used

* Python
* Pandas, NumPy, Matplotlib, Seaborn
* Scikit-learn
* Gradio
* Joblib
