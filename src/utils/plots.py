import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

def plot_confusion(y_true, y_pred, filename):
    os.makedirs("../outputs", exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)

    sns.heatmap(cm, annot=True, fmt='d')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.savefig(f"../outputs/{filename}")
    plt.close()


def plot_comparison(metrics_lr, metrics_nb):
    os.makedirs("../outputs", exist_ok=True)

    labels = ['accuracy', 'precision', 'recall', 'f1']

    lr_values = [metrics_lr[l] for l in labels]
    nb_values = [metrics_nb[l] for l in labels]

    x = range(len(labels))

    plt.plot(x, lr_values, marker='o', label='Logistic Regression')
    plt.plot(x, nb_values, marker='o', label='Naive Bayes')

    plt.xticks(x, labels)
    plt.legend()
    plt.title("Model Comparison")

    plt.savefig("../outputs/comparison.png")
    plt.close()