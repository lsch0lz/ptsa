import numpy as np

from ptsa.tasks.in_hospital_mortality.inference_deterministic import IHMModelInference

def compute_ece(y_true, y_pred, n_bins=10):
    """
    Compute Expected Calibration Error for binary classification.
    
    Args:
        y_true: Array of true labels (0 or 1)
        y_pred: Array of predicted probabilities (between 0 and 1)
        n_bins: Number of bins to use for calculating calibration
        
    Returns:
        ece: Expected Calibration Error
        bin_confidences: Mean predicted probability for each bin
        bin_accuracies: Mean actual accuracy for each bin
        bin_counts: Number of predictions in each bin
    """
    
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Create bins and assign predictions to them
    bins = np.linspace(0., 1. + 1e-8, n_bins + 1)
    binids = np.digitize(y_pred, bins) - 1
    
    bin_sums = np.zeros(n_bins)
    bin_true = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    
    # Accumulate statistics for each bin
    for i in range(len(y_pred)):
        bin_sums[binids[i]] += y_pred[i]
        bin_true[binids[i]] += y_true[i]
        bin_counts[binids[i]] += 1
    
    # Calculate mean accuracy and confidence for each bin
    bin_confidences = bin_sums / (bin_counts + 1e-8)
    bin_accuracies = bin_true / (bin_counts + 1e-8)
    
    # Calculate ECE
    ece = np.sum(np.abs(bin_accuracies - bin_confidences) * (bin_counts / len(y_pred)))
    
    return ece, bin_confidences, bin_accuracies, bin_counts


if __name__ == "__main__":
    config = {
            "input_size": 76,
            "hidden_size": 87,
            "num_layers": 1,
            "learning_rate": 0.00004097747048384285,
            "dropout": 0.2,
            "batch_size": 32,
            "num_epochs": 23,
            "weight_decay": 0.001288495142480056,
            "num_mc_samples": 100
            }
    data_path = "/vol/tmp/scholuka/mimic-iv-benchmarks/data/in-hospital-mortality/"
    model_path = "/vol/tmp/scholuka/ptsa/data/models/in_hospital_mortality/lstm/lstm_ihm_deteministic.pth"


    inference_session = IHMModelInference(config=config, 
                                                  data_path=data_path, 
                                                  model_path=model_path,
                                                  model_name="LSTM", 
                                                  device="cuda:3",
                                                  )
    _, _, test_data = inference_session.load_test_data()

    prediction, y_true = inference_session.infer_on_data_points(test_data)
    
    predictions = np.array(prediction).flatten()
    y_true = np.array(y_true).flatten()

    ece, bin_confidence, bin_acc, bin_counts = compute_ece(y_true, predictions)

    ece, confidences, accuracies, counts = compute_ece(y_true, predictions)
    print(f"Expected Calibration Error: {ece:.4f}")

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k--", label="Identity Line")
    
    plt.plot(confidences, accuracies, 'b-', linewidth=2, label='Model calibration')
    plt.fill_between(confidences, 
                    accuracies, 
                    confidences,
                    alpha=0.2,
                    color='blue',
                    label='Miscalibration area')
    plt.plot(confidences, accuracies, 'b.', markersize=10)

    plt.text(0.05, 0.95, f'Expected Calibration Error = {ece:.2f}', 
            transform=plt.gca().transAxes,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    plt.xlabel("Predicted Probability")
    plt.ylabel("Actual Accuracy")
    plt.title("Calibration Curve")

    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)

    plt.savefig("ece_plot.png")



