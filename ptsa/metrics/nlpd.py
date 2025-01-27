import numpy as np
from scipy.stats import norm

from ptsa.tasks.in_hospital_mortality.inference_deterministic import IHMModelInference

def calculate_nlpd(y_true, predictions, uncertainties, eps=1e-7):
    y_true = np.array(y_true)
    predictions = np.array(predictions)
    uncertainties = np.array(uncertainties)
    
    predictions = np.clip(predictions, eps, 1 - eps)
    
    nlpd_values = -np.log(predictions * y_true + (1 - predictions) * (1 - y_true))
    
    uncertainty_penalty = np.log(2 * np.pi * uncertainties**2) / 2
    nlpd_values += uncertainty_penalty
    
    positive_mask = y_true == 1
    negative_mask = y_true == 0
    
    metrics = {
        'average_nlpd': np.mean(nlpd_values),
        'positive_class_nlpd': np.mean(nlpd_values[positive_mask]) if np.any(positive_mask) else None,
        'negative_class_nlpd': np.mean(nlpd_values[negative_mask]) if np.any(negative_mask) else None,
        'uncertainty_penalty_mean': np.mean(uncertainty_penalty),
        'prediction_term_mean': np.mean(-np.log(predictions * y_true + (1 - predictions) * (1 - y_true)))
    }
    
    return metrics

if __name__ == "__main__":
    PROBABILISTIC_MODEL = True

    config = {
            "input_size": 76,
            "hidden_size": 80,
            "num_layers": 3,
            "learning_rate": 0.00004097747048384285,
            "dropout": 0.27197093767228886,
            "batch_size": 64,
            "num_epochs": 23,
            "weight_decay": 0.001288495142480056,
            "num_mc_samples": 100
            }
    data_path = "/vol/tmp/scholuka/mimic-iv-benchmarks/data/in-hospital-mortality/"
    model_path = "/vol/tmp/scholuka/ptsa/data/models/in_hospital_mortality/rnn/rnn_ihm_probabilistic.pth"


    inference_session = IHMModelInference(config=config, 
                                                  data_path=data_path, 
                                                  model_path=model_path,
                                                  model_name="RNN", 
                                                  device="cuda:3",
                                                  probabilistic=PROBABILISTIC_MODEL
                                                  )
    _, _, test_data = inference_session.load_test_data()

    predictions, y_true, all_uncertainties = inference_session.infer_on_data_points(test_data)
    
    predictions = np.array(predictions).flatten()
    all_uncertainties = np.array(all_uncertainties).flatten()
    y_true = np.array(y_true).flatten()

    mean_nlpd_normalized = calculate_nlpd(y_true, predictions, all_uncertainties)

    print(f"NLPD metrics: {mean_nlpd_normalized}")

        
