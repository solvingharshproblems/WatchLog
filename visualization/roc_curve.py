import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def plot_roc(errors, labels):

    fpr, tpr, thresholds = roc_curve(labels, errors)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6,6))

    plt.plot(fpr, tpr)
    plt.plot([0,1], [0,1], linestyle="--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title(f"ROC Curve (AUC = {roc_auc:.4f})")

    plt.show()