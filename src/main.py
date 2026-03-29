import os

from src.traditional.preprocess import load_and_preprocess
from src.traditional.train_ml import train_ml_models
from src.utils.metrics import evaluate
from src.utils.plots import plot_confusion, plot_comparison
from src.transformer.train_bert import train_bert
# Ensure outputs folder exists
os.makedirs("../outputs", exist_ok=True)

# Load data
data = load_and_preprocess("data/spam.csv")

# Train ML models
lr, nb, X_test, y_test, vectorizer = train_ml_models(data)

# Predictions
y_pred_lr = lr.predict(X_test)
y_pred_nb = nb.predict(X_test)

# Evaluate
metrics_lr = evaluate(y_test, y_pred_lr)
metrics_nb = evaluate(y_test, y_pred_nb)

print("Logistic Regression:", metrics_lr)
print("Naive Bayes:", metrics_nb)

# Save metrics
with open("../outputs/metrics.txt", "w") as f:
    f.write("Logistic Regression:\n")
    f.write(str(metrics_lr) + "\n\n")

    f.write("Naive Bayes:\n")
    f.write(str(metrics_nb))

# Save confusion matrices
plot_confusion(y_test, y_pred_lr, "lr_confusion.png")
plot_confusion(y_test, y_pred_nb, "nb_confusion.png")

# Save comparison graph
plot_comparison(metrics_lr, metrics_nb)

# Train BERT (Part B)
train_bert(data)

print("\n✅ All outputs saved in 'outputs/' folder")