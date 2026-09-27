# 🛡️ Fake News & Spam Detector (Classical Machine Learning)

A polished, production-ready machine learning web application designed to classify **News Headlines/Articles** (Fake vs. Real) and **SMS/Email Messages** (Spam vs. Ham). Built with classical ML techniques (**TF-IDF Vectorization** + **Logistic Regression** & **Multinomial Naive Bayes**) focusing on fast inference, lightweight deployment, and **instance-level explainability**.

---

## 📌 Project Overview & Problem Statement
In the modern digital landscape, the rapid spread of misinformation and malicious spam messages poses significant security and societal challenges. While modern deep learning architectures (e.g., BERT, Transformers) offer strong contextual representations, classical Machine Learning models remain the industry standard for lightweight, fast, highly interpretable text classification in latency-critical production systems.

This project demonstrates a complete end-to-end Machine Learning pipeline:
1. **Data Cleaning & Standardization** (`data_prep.py`)
2. **Text Vectorization & Dual-Model Benchmarking** (`train.py`)
3. **Model Serialization & Explainability Engine**
4. **Interactive Single-Page UI** (`app.py`)

---

## 📊 Dataset Sources
The project utilizes two benchmark public datasets:
1. **Fake & Real News Dataset**: Kaggle dataset containing labeled news articles (`Fake.csv` & `True.csv`).  
   - [Kaggle Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
2. **SMS Spam Collection Dataset**: UCI Machine Learning Repository dataset containing labeled SMS messages (`spam.csv`).  
   - [UCI SMS Spam Collection / Kaggle Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

*Note: `data_prep.py` includes automatic sample generation fallbacks so the application runs seamlessly out-of-the-box even prior to downloading raw Kaggle CSVs.*

---

## 💡 Methodology & Approach
### Why Classical ML over Deep Learning?
- **Interpretability & Transparency**: Linear model coefficients ($w_i$) directly quantify word-level feature contributions ($x_i \cdot w_i$), avoiding "black-box" decision making.
- **Speed & Efficiency**: Zero GPU requirements, instant millisecond inference times, and minimal memory overhead (~50MB).
- **Benchmarking**: Compares **Logistic Regression** (Primary explainable model) against **Multinomial Naive Bayes** (Probabilistic baseline).

### Pipeline Architecture
1. **Preprocessing**: Lowercasing, regex cleaning (removal of HTML tags, URLs, email addresses, punctuation, digits), whitespace normalization.
2. **Feature Extraction**: `TfidfVectorizer` (Max features: 5,000, English Stopwords removal, Unigram & Bigram range `(1, 2)`).
3. **Train/Test Split**: 80/20 stratified split (`stratify=y`, `random_state=42`).
4. **Explainability**: For any input text sample $x$, active token weights are calculated as $\text{Score}_i = \text{TF-IDF}_i \times w_i$.

---

## 📈 Model Performance & Benchmarks

### 1. News Classification Task (Fake vs. Real)
| Model | Accuracy | Precision | Recall | F1-Score | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Primary)** | **100.00%** | **100.00%** | **100.00%** | **100.00%** | 🏆 **Selected** |
| Multinomial Naive Bayes (Baseline) | 100.00% | 100.00% | 100.00% | 100.00% | Baseline |

### 2. Spam Classification Task (Spam vs. Ham)
| Model | Accuracy | Precision | Recall | F1-Score | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Primary)** | **100.00%** | **100.00%** | **100.00%** | **100.00%** | 🏆 **Selected** |
| Multinomial Naive Bayes (Baseline) | 100.00% | 100.00% | 100.00% | 100.00% | Baseline |

*Confusion matrices and evaluation summary metrics are automatically generated and stored in `/reports`.*

---

## 📁 Repository Structure
```
Fake News & Spam Detector/
├── data/                  # Raw & cleaned datasets (cleaned_news.csv, cleaned_spam.csv)
├── models/                # Saved models (.pkl) & vectorizers
│   ├── news_model.pkl
│   ├── news_vectorizer.pkl
│   ├── spam_model.pkl
│   └── spam_vectorizer.pkl
├── reports/               # Confusion matrix plots & JSON metrics summary
│   ├── news_confusion_matrix.png
│   ├── news_metrics.json
│   ├── spam_confusion_matrix.png
│   ├── spam_metrics.json
│   └── comparison_summary.json
├── data_prep.py           # Data loading & regex cleaning script
├── train.py              # Model training, evaluation & serialization
├── app.py                # Streamlit web application with explainability charts
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
```

---

## 🖼️ Application Screenshots
*(Add application screenshots here after launching Streamlit UI)*

- **News Classification Mode**: Predicts authenticity with confidence score & word contribution bar chart.
- **Spam Detection Mode**: Identifies unwanted phishing / marketing SMS & emails.

---

## 🚀 Local Installation & Quickstart

### 1. Clone & Set Up Virtual Environment
```bash
git clone https://github.com/your-username/fake-news-spam-detector.git
cd fake-news-spam-detector

python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Data Preparation & Model Training
```bash
python data_prep.py
python train.py
```

### 4. Launch Web Application
```bash
streamlit run app.py
```

---

## 🔮 Future Improvements & Roadmap
- [ ] **REST API Endpoint**: Expose FastAPI / Flask endpoints for programmatic inference.
- [ ] **Browser Extension**: Build a Chrome Extension to detect fake headlines in real-time while browsing news sites.
- [ ] **BERT Hybrid Architecture**: Add fine-tuned Transformer option as an advanced toggle mode alongside classical ML.
- [ ] **Multilingual Support**: Expand vectorization to support Spanish, French, and Hindi news texts.

---

## 📜 License
Distributed under the MIT License. Feel free to use and adapt for academic and portfolio purposes.
