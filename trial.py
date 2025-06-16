import pyro
import torch
import os
import joblib

# Define the directory
save_dir = 'bayesian_lstm_model'

# Load scalers
loaded_xscaler = joblib.load(os.path.join(save_dir, 'xscaler.pkl'))
loaded_yscaler = joblib.load(os.path.join(save_dir, 'yscaler.pkl'))
print("Scalers loaded.")

# Instantiate the model (assuming BayesianLSTM class is defined)
loaded_bayesian_lstm = BayesianLSTM(input_dim=X_train.shape[2])

# Load the saved parameters with weights_only=False
pyro.get_param_store().load(os.path.join(save_dir, 'bayesian_lstm_params.pt'), map_location=torch.device('cpu'), weights_only=False)
print("Bayesian LSTM model parameters loaded.")