import numpy as np

from ptsa.tasks.in_hospital_mortality.inference_deterministic import IHMModelInference

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
