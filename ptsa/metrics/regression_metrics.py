import numpy as np


from ptsa.tasks.length_of_stay.inference_probabilistic import LOSProbabilisticInference

if __name__ == "__main__":
    config = {
            "input_size": 47,
            "hidden_size": 1,
            "num_layers": 1,
            "learning_rate": 0.00015133860634638263,
            "dropout": 0.4372281049790314,
            "batch_size": 64,
            "num_epochs": 10,
            "weight_decay": 0.001288495142480056,
            "num_mc_samples": 100,
            "d_model": 128,
            "dim_feedforward": 384,
            "nhead": 2,
            }

    data_path = "/vol/tmp/scholuka/mimic-iv-benchmarks/data/length-of-stay-own/"
    model_path = "/vol/tmp/scholuka/ptsa/data/models/length_of_stay/probabilistic/final/transformer_final_model.pth"


    inference_session = LOSProbabilisticInference(config=config, 
                                                  data_path=data_path, 
                                                  model_path=model_path,
                                                  model_name="transformer", 
                                                  device="cuda:3",
                                                  num_batches_inference=200,
                                                  limit_num_test_sampled=True)
    train_data , _, test_data = inference_session.load_test_data()

    predicted_means, predicted_variances, y_true = inference_session.infer_on_data_points(test_data)
    
    predicted_means = np.array(predicted_means).flatten()
    predicted_variances = np.array(predicted_variances).flatten()
    y_true = np.array(y_true).flatten()

