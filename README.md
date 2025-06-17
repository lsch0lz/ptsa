# Probabilistic Time Series Analysis (PTSA)

This projects several deterministic and probabilistic models to classify In-hospital Mortality and Length-of-Stay Prediction.

## Features

- **Probabilistic Models**: Support for various probabilistic time series models.
- **Visualizations**: Generate calibration plots and other visualizations.
- **Customizable Configurations**: Flexible model and training parameters.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Lukas-Scholz-dbahn/ptsa.git
   cd ptsa
   pip install -r requirements.txt
   ```
   
2. Train networks using the following command:

Classification:
```bash
python ptsa/tasks/in_hospital_mortality/train_deterministic.py --network ptsa/models/deterministic/lstm.py --partition custom --data PATH_TO_YOUR_DATA --model lstm  --model_name model_name.pth --output_dir PATH_TO_OUTPUT_FOLDER
```

Regression:
```bash
python ptsa/tasks/length_of_stay/optimize_probabilistic.py --network ptsa/models/probabilistic/rnn.py --partition custom --data PATH_TO_YOUR_DATA --model rnn  --model_name model_name.pth --output_dir PATH_TO_OUTPUT_FOLDER --num_mc_samples 25 --num_trials 10 --dataset_fraction 0.1
```

