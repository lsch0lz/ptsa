from ptsa.tasks.in_hospital_mortality.inference_probabilistic import IHMProbabilisticInference


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


mean, variance = inference_session.infer_on_data_points(test_data)

print(variance)

