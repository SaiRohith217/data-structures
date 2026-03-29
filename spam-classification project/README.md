# Spam Classification Project

## Overview

This project implements spam detection using both traditional machine learning models and a transformer-based model. The goal is to classify messages as spam or not spam.

## Models Used

Traditional Machine Learning:

* Logistic Regression
* Naive Bayes
* TF-IDF Vectorization

Transformer Model:

* DistilBERT (fine-tuned using PyTorch)

## Project Structure

```
spam-classification-project/
│
├── data/
├── src/
│   ├── traditional/
│   ├── transformer/
│   ├── utils/
│   └── main.py
│
├── outputs/
├── report/
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

Install required dependencies:

```
pip install -r requirements.txt
```

## Usage

Run the project from the root directory:

```
python -m src.main
```

## Outputs

The following outputs are generated in the outputs folder:

* metrics.txt
* lr_confusion.png
* nb_confusion.png
* comparison.png
* bert_loss.png

## Dataset

Spam Email Classification dataset.

## Notes

* Transformer training may take a few minutes on CPU.
* DistilBERT is fine-tuned for this classification task.

## Author

Sai Rohith

## License

This project is licensed under the Apache License 2.0.