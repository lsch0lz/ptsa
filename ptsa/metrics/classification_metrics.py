import numpy as np
from sklearn.metrics import (
    roc_curve, precision_recall_curve, average_precision_score,
    f1_score, accuracy_score, precision_score, recall_score, 
    roc_auc_score, confusion_matrix, log_loss, brier_score_loss, auc
)
import matplotlib.pyplot as plt

from ptsa.tasks.in_hospital_mortality.inference_deterministic import IHMModelInference


def compute_classification_metrics(
    y_true: np.ndarray, 
    y_pred_proba: np.ndarray, 
    threshold: float = 0.5
    ):
    """
    Compute various binary classification metrics.
    
    Args:
        y_true: Ground truth binary labels (0 or 1)
        y_pred_proba: Predicted probabilities for class 1
        threshold: Decision threshold for converting probabilities to binary predictions
        
    Returns:
        Dictionary containing various classification metrics
    """
    # Convert probabilities to binary predictions using threshold
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Probability-based metrics
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    avg_precision = average_precision_score(y_true, y_pred_proba)
    
    # Calibration metrics
    brier = brier_score_loss(y_true, y_pred_proba)
    log_loss_value = log_loss(y_true, y_pred_proba)
    
    # Confusion matrix elements
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
    
    # Additional derived metrics
    prevalence = np.mean(y_true)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "sensitivity": recall,  # Same as recall
        "specificity": specificity,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "average_precision": avg_precision,
        "brier_score": brier,
        "log_loss": log_loss_value,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "prevalence": prevalence,
        "negative_predictive_value": npv
    }

def plot_roc_curve(y_true, y_pred_proba):
    """
    Plot ROC curve and compute AUC.
    
    Args:
        y_true: Ground truth binary labels
        y_pred_proba: Predicted probabilities
        
    Returns:
        AUC score
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig("roc_curve_classification.png")


def plot_precision_recall_curve(y_true, y_pred_proba):
    """
    Plot Precision-Recall curve and compute Average Precision.
    
    Args:
        y_true: Ground truth binary labels
        y_pred_proba: Predicted probabilities
        
    Returns:
        Average precision score
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    avg_precision = average_precision_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {avg_precision:.3f})')
    
    # Calculate baseline based on class imbalance
    baseline = np.sum(y_true) / len(y_true)
    plt.axhline(y=baseline, color='red', linestyle='--', label=f'Baseline (prevalence = {baseline:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.savefig("precision_recall_curve_classification.png")
    
    return avg_precision

if __name__ == "__main__":
    PROBABILISTIC_MODEL = False
    model_type = "transformer"

    config = {
            "input_size": 47,
            "hidden_size": 99,
            "num_layers": 2,
            "learning_rate": 0.0009139142870085612,
            "dropout": 0.2018675522864059,
            "batch_size": 31,
            "num_epochs": 29,
            "weight_decay": 0.00003018240933325477,
            "num_mc_samples": 100,
            "d_model": 256,
            "dim_feedforward": 448,
            "nhead": 4
            }
    data_path = "/vol/tmp/scholuka/mimic-iv-benchmarks/data/in-hospital-mortality-own"
    model_path = "/vol/tmp/scholuka/ptsa/data/models/in_hospital_mortality/deterministic/final_model_transformer_ihm.pth"


    inference_session = IHMModelInference(config=config, 
                                                  data_path=data_path, 
                                                  model_path=model_path,
                                                  model_name=model_type, 
                                                  device="cuda:3",
                                                  probabilistic=PROBABILISTIC_MODEL
                                                  )
    train_data, _, test_data = inference_session.load_test_data()

    predictions, y_true, all_uncertainties = inference_session.infer_on_data_points(test_data)
    
    predictions = np.array(predictions).flatten()
    all_uncertainties = np.array(all_uncertainties).flatten()
    y_true = np.array(y_true).flatten()

    classification_metrics = compute_classification_metrics(y_true, predictions)
    print("*" * 20)
    print(f"Model Type: {model_type}")
    print(classification_metrics)

    plot_roc_curve(y_true, predictions)

    print("*" * 20)
    avg_precision = plot_precision_recall_curve(y_true, predictions)
    print(f"AVG Precision: {avg_precision}")

