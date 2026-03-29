# 📧 Spam Classification Project

## 📌 Overview

This project implements a **comparative study of traditional Machine Learning models and Transformer-based models** for spam detection.

The goal is to classify messages into:

* ✅ Ham (Not Spam)
* ❌ Spam

---

## 🧠 Models Implemented

### 🔹 Traditional Machine Learning

* Logistic Regression
* Naive Bayes
* TF-IDF Vectorization

### 🔹 Transformer Model

* DistilBERT (from Hugging Face)
* Fine-tuned using PyTorch

---

## 📂 Project Structure

```
spam-classification-project/
│
├── data/               # Dataset
├── src/                # Source code
│   ├── traditional/    # ML models
│   ├── transformer/    # BERT model
│   ├── utils/          # Metrics & plots
│   └── main.py         # Entry point
│
├── outputs/            # Generated results
├── report/             # Documentation
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```
pip install -r requirements.txt
```

---

## ▶️ How to Run

```
python -m src.main
```

---

## 📊 Outputs

The following outputs are automatically generated:

* 📄 metrics.txt → Accuracy, Precision, Recall, F1-score
* 📊 lr_confusion.png → Logistic Regression confusion matrix
* 📊 nb_confusion.png → Naive Bayes confusion matrix
* 📈 comparison.png → Model comparison graph
* 📉 bert_loss.png → Transformer training loss

---

## 📈 Results

| Model               | Accuracy | Precision | Recall   | F1-score |
| ------------------- | -------- | --------- | -------- | -------- |
| Logistic Regression | ~96%     | High      | Moderate | Good     |
| Naive Bayes         | ~96.5%   | High      | Moderate | Better   |
| DistilBERT          | High     | High      | High     | Best     |

---

## 🧪 Dataset

Spam Email Classification Dataset (Kaggle)

---

## ⚠️ Notes

* Transformer model may take a few minutes to train on CPU
* DistilBERT is fine-tuned for spam classification
* Missing weights during loading are expected

---

#Conclusion

* Traditional models are fast and efficient
* Transformer models understand context better
* DistilBERT gives better overall performance

---

#Author

S.SAI ROHITH

---

#License

This project is licensed under the MIT License.
