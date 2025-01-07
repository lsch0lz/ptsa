import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.ticker import MaxNLocator

from ptsa.tasks.in_hospital_mortality.inference_probabilistic import IHMProbabilisticInference


def calculate_bca_confidence_intervals(data, statistic_func, confidence_level=0.95, n_bootstraps=1000):
    """Calculate BCa (bias-corrected and accelerated) bootstrap confidence intervals"""
    n = len(data)
    # Generate bootstrap samples
    bootstrap_stats = []
    for _ in range(n_bootstraps):
        bootstrap_sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic_func(bootstrap_sample))
    
    # Calculate bias correction factor
    observed_stat = statistic_func(data)
    prop_less = np.mean(np.array(bootstrap_stats) < observed_stat)
    z0 = stats.norm.ppf(prop_less)
    
    # Calculate acceleration factor
    jackknife_stats = []
    for i in range(n):
        jackknife_sample = np.delete(data, i)
        jackknife_stats.append(statistic_func(jackknife_sample))
    jackknife_mean = np.mean(jackknife_stats)
    num = np.sum((jackknife_mean - jackknife_stats) ** 3)
    den = 6 * (np.sum((jackknife_mean - jackknife_stats) ** 2)) ** 1.5
    a = num / (den + np.finfo(float).eps)  # Add small epsilon to prevent division by zero
    
    # Calculate BCa intervals
    z_alpha = stats.norm.ppf((1 - confidence_level) / 2)
    z_1_alpha = stats.norm.ppf(1 - (1 - confidence_level) / 2)
    
    for z in [z_alpha, z_1_alpha]:
        w = z0 + (z0 + z) / (1 - a * (z0 + z))
        p = stats.norm.cdf(w)
        yield np.percentile(bootstrap_stats, p * 100)

def create_error_based_calibration_plot(y_true, predicted_means, predicted_variances, n_bins=20):
    """
    Create error-based calibration plot with improved scaling and fixed aspect ratio
    """
    # Calculate errors
    errors = y_true - predicted_means
    
    # Sort by predicted uncertainty
    sorted_indices = np.argsort(predicted_variances)
    sorted_errors = errors[sorted_indices]
    sorted_variances = predicted_variances[sorted_indices]
    
    # Calculate bin size
    bin_size = len(y_true) // n_bins
    
    rmse_values = []
    rmv_values = []
    ci_lower = []
    ci_upper = []
    
    # Calculate RMSE and RMV for each bin
    for i in range(n_bins):
        start_idx = i * bin_size
        end_idx = start_idx + bin_size if i < n_bins - 1 else len(y_true)
        
        # Get bin data
        bin_errors = sorted_errors[start_idx:end_idx]
        bin_variances = sorted_variances[start_idx:end_idx]
        
        # Calculate RMSE and RMV
        rmse = np.sqrt(np.mean(bin_errors**2))
        rmv = np.sqrt(np.mean(bin_variances))
        
        rmse_values.append(rmse)
        rmv_values.append(rmv)
        
        # Calculate 95% confidence intervals using BCa bootstrap
        ci = list(calculate_bca_confidence_intervals(
            bin_errors,
            lambda x: np.sqrt(np.mean(x**2)),
            confidence_level=0.95
        ))
        ci_lower.append(ci[0])
        ci_upper.append(ci[1])
    
    # Convert to numpy arrays
    rmse_values = np.array(rmse_values)
    rmv_values = np.array(rmv_values)
    ci_lower = np.array(ci_lower)
    ci_upper = np.array(ci_upper)
    
    # Fit linear regression
    slope, intercept = np.polyfit(rmv_values, rmse_values, 1)
    r_squared = np.corrcoef(rmv_values, rmse_values)[0,1]**2
    
    # Create plot with fixed size
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot confidence intervals
    ax.fill_between(rmv_values, ci_lower, ci_upper, alpha=0.2, color='blue',
                   label='95% Confidence Interval')
    
    # Plot calibration points
    ax.scatter(rmv_values, rmse_values, color='blue', label='Observed')
    
        
        
    # Set equal limits for both axes
    ax.set_xlim(0, np.max(rmv_values))
    ax.set_ylim(0, np.max(rmse_values))
    
    # Plot fitted line using the full range
    x_range = np.array([0, np.max(rmse_values)])
    ax.plot(x_range, slope * x_range + intercept, 'r--', 
            label=f'Fitted (R²={r_squared:.2f})')
    
    # Plot identity line
    ax.plot(x_range, x_range, 'k-', label='Identity')
    
    # Customize plot
    ax.set_xlabel('RMV (Root Mean Variance)')
    ax.set_ylabel('RMSE (Root Mean Square Error)')
    ax.set_title('Error-based Calibration Plot')
    ax.legend()
    
    # Add grid with better visibility
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Ensure square aspect ratio
    ax.set_aspect('equal')
    
    # Add more tick marks for better readability
    ax.xaxis.set_major_locator(MaxNLocator(10))
    ax.yaxis.set_major_locator(MaxNLocator(10))
    
    metrics = {
        'R2': r_squared,
        'slope': slope,
        'intercept': intercept
    }
    
    return fig, metrics

# Example usage
if __name__ == "__main__":
    config = {
            "input_size": 76,
            "hidden_size": 248,
            "num_layers": 1,
            "learning_rate": 0.00001069901749223866,
            "dropout": 0.5767284766833869,
            "batch_size": 64,
            "num_epochs": 38,
            "weight_decay": 0.001288495142480056,
            "num_mc_samples": 100
            }
    data_path = "/vol/tmp/scholuka/mimic-iv-benchmarks/data/in-hospital-mortality/"
    model_path = "/vol/tmp/scholuka/ptsa/data/models/in_hospital_mortality/rnn_probabilistic/final_model_trial_8.pth"


    inference_session = IHMProbabilisticInference(config=config, data_path=data_path, model_path=model_path, model_name="RNN")
    _, _, test_data = inference_session.load_test_data()

    predicted_means, predicted_variances, y_true = inference_session.infer_on_data_points(test_data)
    
    predicted_means = np.array(predicted_means).flatten()
    predicted_variances = np.array(predicted_variances).flatten()
    y_true = np.array(y_true).flatten()
    

    np.random.seed(42)
    n_samples = 5000
    
    # Generate more representative example data
    # True values follow a normal distribution
    # y_true = np.random.normal(0, 2, n_samples)
    
    # Add heteroscedastic noise to predictions
    noise_scale = 0.1 + 0.2 * np.abs(y_true)  # Noise increases with magnitude
    # predicted_means = y_true + np.random.normal(0, noise_scale, n_samples)
    
    # Generate variances that somewhat correlate with actual errors
    base_uncertainty = noise_scale**2  # True uncertainty
    # predicted_variances = base_uncertainty * (0.7 + 0.6 * np.random.random(n_samples))
    
    # Sort predictions by uncertainty to create a clearer trend
    sort_idx = np.argsort(predicted_variances)
    y_true = y_true[sort_idx]
    predicted_means = predicted_means[sort_idx]
    predicted_variances = predicted_variances[sort_idx]

    fig, metrics = create_error_based_calibration_plot(
        y_true, predicted_means, predicted_variances, n_bins=20
    )
    
    print("Calibration Metrics:")
    print(f"R² = {metrics['R2']:.3f}")
    print(f"Slope = {metrics['slope']:.3f}")
    print(f"Intercept = {metrics['intercept']:.3f}")
   
    fig.savefig('calibration_plot_hq.png', dpi=300)
