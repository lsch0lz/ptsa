import numpy as np

from ptsa.tasks.in_hospital_mortality.inference_deterministic import IHMModelInference

if __name__ == "__main__":
    config = {
            "input_size": 76,
            "hidden_size": 64,
            "num_layers": 2,
            "learning_rate": 0.001,
            "dropout": 0.2,
            "batch_size": 64,
            "num_epochs": 5,
            "weight_decay": 0.001288495142480056,
            "num_mc_samples": 100
            }
    data_path = "/vol/tmp/scholuka/mimic-iv-benchmarks/data/in-hospital-mortality/"
    model_path = "/vol/tmp/scholuka/ptsa/data/models/in_hospital_mortality/"


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



