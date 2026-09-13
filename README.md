# Fake News Classification & Gradio Application

## Overview
This project applies machine learning techniques to classify news articles as **Real** or **Fake**. Using `TfidfVectorizer` for text feature extraction and scikit-learn classification algorithms (`LogisticRegression`, `MultinomialNB`), model pipelines were trained and evaluated via 5-fold cross-validation to find the optimal balance between accuracy and stability. The final trained `TfidfVectorizer` + `LogisticRegression` pipeline is integrated into an interactive web application built with Gradio for real-time predictions.

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
- **TF-IDF Vectorizer (`TfidfVectorizer`):** Converts raw text strings into numerical feature matrices by scoring term frequency adjusted for inverse document frequency across the corpus.
- **Pipeline Vectorization:** Integrated directly into the scikit-learn `Pipeline` object so that raw article text is automatically transformed without requiring separate vectorization steps prior to inference.


---

Full Stack AI & Data Science — PROJECT 02

## Cross-Validation Model Comparison

| Pipeline | CV Accuracy | CV Macro F1 |
| CountVectorizer + MultinomialNB | 0.9520 ± 0.0031 | 0.9515 ± 0.0032 |
| TF-IDF + MultinomialNB | 0.9385 ± 0.0040 | 0.9379 ± 0.0041 |
| **TF-IDF + Logistic Regression** | **0.9865 ± 0.0018** | **0.9863 ± 0.0019** |

## Final Model

* **Pipeline:** `TfidfVectorizer()` + `LogisticRegression()`
* **Test Accuracy:** 0.9930
* **Test Precision:** 0.9930
* **Test Recall:** 0.9900
* **Test F1-Score:** 0.9914

### Why this model?

The TF-IDF + Logistic Regression** pipeline achieved the highest mean accuracy and F1-score across 5-fold cross-validation while exhibiting the lowest standard deviation (highest stability). `LogisticRegression` handles high-dimensional, sparse feature matrices produced by `TfidfVectorizer` exceptionally well without overfitting, outperforming `MultinomialNB` baselines on subtle contextual cues.

## Installation

```bash
git clone [(https://github.com/Ridwanul-hoque/Fake-news-classifier/tree/main)]https://github.com/Ridwanul-hoque/Fake-news-classifier/tree/main
cd fake-news-classifier
pip install -r requirements.txt

````

## Technologies Used

* **Python**
* **Machine Learning & Feature Extraction:** Scikit-learn (`TfidfVectorizer`, `LogisticRegression`, `MultinomialNB`, `CountVectorizer`)
* **Data Processing & Visualization:** Pandas, NumPy, Matplotlib, Seaborn
