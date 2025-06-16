import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
pyro.set_rng_seed(42)

# ----------------- Hampel Filter Function -----------------
def hampel_filter(data, window_size, n_sigma):
    """
    Apply Hampel filter to detect and mark outliers in time series data.
    
    Parameters:
    - data: pandas Series with datetime index
    - window_size: size of the rolling window
    - n_sigma: number of standard deviations for outlier threshold
    
    Returns:
    - outlier_mask: boolean mask where True indicates outlier
    - filtered_data: data with outliers replaced by rolling median
    """
    rolling_median = data.rolling(window=window_size, center=True).median()
    rolling_mad = data.rolling(window=window_size, center=True).apply(
        lambda x: np.median(np.abs(x - np.median(x)))
    )
    threshold = n_sigma * rolling_mad * 1.4826
    outlier_mask = np.abs(data - rolling_median) > threshold
    filtered_data = data.copy()
    filtered_data[outlier_mask] = rolling_median[outlier_mask]
    return outlier_mask, filtered_data

# ----------------- Load and Clean Data -----------------
df = pd.read_csv('AirQualityUCI.csv', delimiter=';')
df = df.drop('NMHC(GT)', axis=1)

df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H.%M.%S')
df.set_index('Datetime', inplace=True)
df = df.drop(['Date', 'Time'], axis=1)
df = df.drop(df.columns[[12, 13]], axis=1)

for col in df.columns:
    df[col] = df[col].astype(str).str.replace(',', '.', regex=False).str.strip()
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna()
df.replace(-200, np.nan, inplace=True)
df.interpolate(method='linear', inplace=True)

print("Available columns:", df.columns.tolist())
print(f"Data shape before outlier removal: {df.shape}")

# ----------------- Apply Hampel Filter to Target Variable -----------------
target_col = 'CO(GT)'
print(f"\nApplying Hampel filter to target variable: {target_col}")

window_size = 24  # 24 hours for hourly data
n_sigma = 2.5     # 2.5 standard deviations

outlier_mask, filtered_target = hampel_filter(df[target_col], window_size=window_size, n_sigma=n_sigma)

n_outliers = outlier_mask.sum()
outlier_percentage = (n_outliers / len(df)) * 100

print(f"Detected {n_outliers} outliers ({outlier_percentage:.2f}% of data)")
print(f"Outlier detection parameters: window_size={window_size}, n_sigma={n_sigma}")

outlier_info = pd.DataFrame({
    'timestamp': df.index[outlier_mask],
    'original_value': df[target_col][outlier_mask],
    'filtered_value': filtered_target[outlier_mask]
})

print(f"\nRemoving {n_outliers} rows with outliers in target variable...")
df_clean = df[~outlier_mask].copy()
print(f"Data shape after outlier removal: {df_clean.shape}")

df = df_clean.copy()

# ----------------- Prepare Features and Target -----------------
target = df[target_col].copy()
feature_columns = [col for col in df.columns if col != target_col]
X = df[feature_columns].copy()

X['hour'] = X.index.hour
X['day_of_week'] = X.index.dayofweek
X['month'] = X.index.month

X['hour_sin'] = np.sin(2 * np.pi * X['hour'] / 24)
X['hour_cos'] = np.cos(2 * np.pi * X['hour'] / 24)
X['day_sin'] = np.sin(2 * np.pi * X['day_of_week'] / 7)
X['day_cos'] = np.cos(2 * np.pi * X['day_of_week'] / 7)
X['month_sin'] = np.sin(2 * np.pi * X['month'] / 12)
X['month_cos'] = np.cos(2 * np.pi * X['month'] / 12)

X = X.drop(['hour', 'day_of_week', 'month'], axis=1)
feature_columns = X.columns.tolist()

print(f"\nOriginal number of features (including cyclic): {len(feature_columns)}")
print("Features (including cyclic):", feature_columns)

# ----------------- Apply Basic Preprocessing -----------------
X = X.diff(24).dropna()
target = target.diff(24).dropna()

target = target.rolling(window=3, center=True).mean().dropna()
X = X.rolling(window=3, center=True).mean().dropna()

common_index = X.index.intersection(target.index)
X = X.loc[common_index]
target = target.loc[common_index]

# ----------------- Recursive Feature Elimination -----------------
rf_estimator = RandomForestRegressor(n_estimators=100, random_state=42)
n_features_to_select = min(10, len(feature_columns))

rfe = RFE(estimator=rf_estimator, n_features_to_select=n_features_to_select, step=1)
X_rfe = rfe.fit_transform(X, target)

selected_features = [feature_columns[i] for i in range(len(feature_columns)) if rfe.support_[i]]
print(f"\nSelected {len(selected_features)} features using RFE:")
print("Selected features:", selected_features)
print("Feature ranking:", dict(zip(feature_columns, rfe.ranking_)))

# ----------------- Scale Data -----------------
xscaler = StandardScaler()
yscaler = StandardScaler()

X_scaled = xscaler.fit_transform(X_rfe)
y_scaled = yscaler.fit_transform(target.values.reshape(-1, 1)).flatten()

# ----------------- Sequence Creation -----------------
def create_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(Xs), np.array(ys)

SEQ_LEN = 24
X_seq, y_seq = create_sequences(X_scaled, y_scaled, SEQ_LEN)

print(f"\nSequence shape: {X_seq.shape}")
print(f"Target shape: {y_seq.shape}")

# ----------------- Split Data -----------------
n = len(X_seq)
train_size = int(n * 0.7)
val_size = int(n * 0.15)

X_train = torch.tensor(X_seq[:train_size], dtype=torch.float32)
y_train = torch.tensor(y_seq[:train_size], dtype=torch.float32)
X_val = torch.tensor(X_seq[train_size:train_size+val_size], dtype=torch.float32)
y_val = torch.tensor(y_seq[train_size:train_size+val_size], dtype=torch.float32)
X_test = torch.tensor(X_seq[train_size+val_size:], dtype=torch.float32)
y_test = torch.tensor(y_seq[train_size+val_size:], dtype=torch.float32)

print(f"\nTrain shape: {X_train.shape}")
print(f"Validation shape: {X_val.shape}")
print(f"Test shape: {X_test.shape}")

# ----------------- Bayesian LSTM Model -----------------
class BayesianLSTM(pyro.nn.PyroModule):
    def __init__(self, input_dim, hidden_dim1=352, hidden_dim2=128, hidden_dim3=64, dense_dim1=72, dense_dim2=32):
        super(BayesianLSTM, self).__init__()
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.hidden_dim3 = hidden_dim3
        
        # LSTM layers
        self.lstm1 = pyro.nn.PyroModule[nn.LSTM](input_dim, hidden_dim1, batch_first=True, bidirectional=True)
        self.bn1 = pyro.nn.PyroModule[nn.BatchNorm1d](hidden_dim1 * 2)
        self.lstm2 = pyro.nn.PyroModule[nn.LSTM](hidden_dim1 * 2, hidden_dim2, batch_first=True, bidirectional=True)
        self.bn2 = pyro.nn.PyroModule[nn.BatchNorm1d](hidden_dim2 * 2)
        self.lstm3 = pyro.nn.PyroModule[nn.LSTM](hidden_dim2 * 2, hidden_dim3, batch_first=True, bidirectional=True)
        self.bn3 = pyro.nn.PyroModule[nn.BatchNorm1d](hidden_dim3 * 2)
        
        # Dense layers
        self.dense1 = pyro.nn.PyroModule[nn.Linear](hidden_dim3 * 2, dense_dim1)
        self.dense2 = pyro.nn.PyroModule[nn.Linear](dense_dim1, dense_dim2)
        self.dense_out = pyro.nn.PyroModule[nn.Linear](dense_dim2, 2)  # Predict mean and log-variance
    
    def forward(self, x, y=None):
        # LSTM layers
        x, _ = self.lstm1(x)
        x = x.permute(0, 2, 1)  # For batch norm: (batch, features, seq_len)
        x = self.bn1(x).permute(0, 2, 1)  # Back to (batch, seq_len, features)
        
        x, _ = self.lstm2(x)
        x = x.permute(0, 2, 1)
        x = self.bn2(x).permute(0, 2, 1)
        
        x, _ = self.lstm3(x)
        x = x[:, -1, :]  # Take the last time step
        x = self.bn3(x)
        
        # Dense layers
        x = torch.relu(self.dense1(x))
        x = torch.relu(self.dense2(x))
        output = self.dense_out(x)  # Outputs [mean, log-variance]
            
        mean, log_var = output[:, 0], output[:, 1]
        sigma = torch.exp(0.5 * log_var) + 1e-5  # Ensure positive scale
                
        # Define observation distribution
        with pyro.plate("data", x.shape[0]):
            pyro.sample("obs", dist.Normal(mean, sigma), obs=y)
                
        return mean, sigma

    def model(self, x, y=None):
        # Define priors for all parameters
        for name, param in self.named_parameters():
            prior = dist.Normal(0.0, 1.0).expand(param.shape).to_event(len(param.shape))
            pyro.sample(f"prior_{name}", prior)
        
        # Forward pass with observation
        mean, sigma = self.forward(x, y)
        return mean, sigma

    def guide(self, x, y=None):
        # Define variational distributions
        for name, param in self.named_parameters():
            loc = pyro.param(f"{name}_loc", torch.randn_like(param))
            scale = pyro.param(f"{name}_scale", torch.ones_like(param) * 0.1, constraint=dist.constraints.positive)
            pyro.sample(f"prior_{name}", dist.Normal(loc, scale).to_event(len(param.shape)))

# ----------------- Initialize Model and Optimizer -----------------
bayesian_lstm = BayesianLSTM(input_dim=X_train.shape[2])
svi = SVI(bayesian_lstm.model, bayesian_lstm.guide, pyro.optim.Adam({"lr": 0.001}), loss=Trace_ELBO())

# ----------------- Training Loop -----------------
def train_model(model, x_train, y_train, x_val, y_val, epochs=100, batch_size=32):
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        epoch_train_loss = 0.0
        permutation = torch.randperm(x_train.shape[0])
        for i in range(0, x_train.shape[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = x_train[indices], y_train[indices]
            loss = svi.step(batch_x, batch_y)
            epoch_train_loss += loss / batch_x.shape[0]
        
        train_losses.append(epoch_train_loss / (x_train.shape[0] // batch_size))
        
        # Validation
        val_loss = 0.0
        with torch.no_grad():
            val_loss = svi.evaluate_loss(x_val, y_val) / x_val.shape[0]
        val_losses.append(val_loss)
        
        # Early stopping
        if epoch > 20 and val_losses[-1] > val_losses[-2]:
            break
    
    return train_losses, val_losses

# Train the model
train_losses, val_losses = train_model(bayesian_lstm, X_train, y_train, X_val, y_val)

# ----------------- Bayesian Predictions with VI -----------------
def predict_with_variational_inference(model, x_data, n_samples=100):
    predictive = pyro.infer.Predictive(model.model, guide=model.guide, num_samples=n_samples)
    samples = predictive(x_data)
    
    # Extract predictions
    y_pred = samples["obs"].detach().numpy()
    if y_pred.shape[-1] == 1:
        y_pred = y_pred.squeeze(-1)  # Remove last dimension if it is 1
    else:
        print(f"Warning: Last dimension of y_pred is {y_pred.shape[-1]}, not squeezing.")
    
    mean_pred = y_pred.mean(axis=0)
    std_pred = y_pred.std(axis=0)
    lower_bound = np.percentile(y_pred, 2.5, axis=0)
    upper_bound = np.percentile(y_pred, 97.5, axis=0)
    
    # Estimate aleatoric uncertainty from the model's output
    with torch.no_grad():
        mean, sigma = model.forward(x_data)
    
    return mean_pred, std_pred, lower_bound, upper_bound, sigma.numpy()

# Generate predictions
y_pred_mean, y_pred_std, y_pred_lower, y_pred_upper, y_pred_aleatoric_std = predict_with_variational_inference(
    bayesian_lstm, X_test, n_samples=100
)

# Inverse transform predictions
y_test_inv = yscaler.inverse_transform(y_test.numpy().reshape(-1, 1)).flatten()
y_pred_mean_inv = yscaler.inverse_transform(y_pred_mean.reshape(-1, 1)).flatten()
y_pred_std_inv = yscaler.scale_ * y_pred_std.flatten()
y_pred_lower_inv = yscaler.inverse_transform(y_pred_lower.reshape(-1, 1)).flatten()
y_pred_upper_inv = yscaler.inverse_transform(y_pred_upper.reshape(-1, 1)).flatten()
y_pred_aleatoric_std_inv = yscaler.scale_ * y_pred_aleatoric_std.flatten()

# Calculate metrics
r2_bayesian = r2_score(y_test_inv, y_pred_mean_inv)
mse_bayesian = mean_squared_error(y_test_inv, y_pred_mean_inv)

print(f"\n=== MODEL PERFORMANCE (Variational Inference) ===")
print(f"  R2 Score: {r2_bayesian:.4f}")
print(f"  MSE: {mse_bayesian:.4f}")
print(f"  RMSE: {np.sqrt(mse_bayesian):.4f}")
print(f"  Mean Total Prediction Uncertainty (std): {np.mean(y_pred_std_inv):.4f}")
print(f"  Mean Aleatoric Uncertainty (std from output layer): {np.mean(y_pred_aleatoric_std_inv):.4f}")

within_interval = np.sum((y_test_inv.flatten() >= y_pred_lower_inv.flatten()) & 
                        (y_test_inv.flatten() <= y_pred_upper_inv.flatten()))
coverage = within_interval / len(y_test_inv) * 100
print(f"  95% Prediction Interval Coverage: {coverage:.2f}%")

# ----------------- Enhanced Plotting with Uncertainty -----------------
plt.figure(figsize=(20, 16))

# Plot 1: Predictions with Uncertainty Bands
plt.subplot(3, 3, 1)
n_plot = min(300, len(y_test_inv))
x_axis = range(n_plot)

plt.plot(x_axis, y_test_inv[:n_plot], label='Actual', color='blue', alpha=0.8, linewidth=2)
plt.plot(x_axis, y_pred_mean_inv[:n_plot], label='Bayesian Pred (Mean)', color='red', alpha=0.8, linewidth=2)
plt.fill_between(x_axis, 
                 y_pred_lower_inv[:n_plot].flatten(), 
                 y_pred_upper_inv[:n_plot].flatten(),
                 alpha=0.3, color='red', label='95% Credible Interval')
plt.title("Bayesian LSTM (VI): Predictions with Uncertainty")
plt.xlabel("Time Steps")
plt.ylabel("CO(GT) Concentration")
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Training History
plt.subplot(3, 3, 2)
plt.plot(train_losses, label='Training Loss', alpha=0.8)
plt.plot(val_losses, label='Validation Loss', alpha=0.8)
plt.title('Model Training History')
plt.xlabel('Epochs')
plt.ylabel('Negative Log Likelihood Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Uncertainty Distribution (Total and Aleatoric)
plt.subplot(3, 3, 3)
plt.hist(y_pred_std_inv, bins=50, alpha=0.7, color='orange', edgecolor='black', label='Total Uncertainty (from sampling)')
plt.hist(y_pred_aleatoric_std_inv, bins=50, alpha=0.7, color='purple', edgecolor='black', label='Aleatoric Uncertainty (model output scale)')
plt.title('Distribution of Prediction Uncertainty')
plt.xlabel('Standard Deviation')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 4: Scatter Plot with Error Bars (Bayesian)
plt.subplot(3, 3, 4)
sample_indices = np.random.choice(len(y_test_inv), size=min(500, len(y_test_inv)), replace=False)
plt.errorbar(
    y_test_inv[sample_indices].flatten(), 
    y_pred_mean_inv[sample_indices].flatten(), 
    yerr=2 * y_pred_std_inv[sample_indices].flatten(),
    fmt='o', 
    alpha=0.5, 
    markersize=3, 
    capsize=2, 
    color='red', 
    label='Bayesian'
)
plt.plot([y_test_inv.min(), y_test_inv.max()], [y_test_inv.min(), y_test_inv.max()], 'r--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title(f'Bayesian Scatter Plot (R² = {r2_bayesian:.4f})')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 5: Prediction Interval Coverage Over Time
plt.subplot(3, 3, 5)
coverage_by_time = []
window_size = 50
for i in range(0, len(y_test_inv) - window_size, window_size):
    end_idx = i + window_size
    window_coverage = np.sum((y_test_inv[i:end_idx].flatten() >= y_pred_lower_inv[i:end_idx].flatten()) & 
                           (y_test_inv[i:end_idx].flatten() <= y_pred_upper_inv[i:end_idx].flatten()))
    coverage_by_time.append(window_coverage / window_size * 100)
plt.plot(coverage_by_time, marker='o', alpha=0.7)
plt.axhline(y=95, color='r', linestyle='--', label='Target 95%')
plt.title(f'Prediction Interval Coverage\n(Average: {coverage:.1f}%)')
plt.xlabel(f'Time Windows (size={window_size})')
plt.ylabel('Coverage (%)')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 6: Feature Importance (re-using previous RFE importance for context)
plt.subplot(3, 3, 6)
feature_importance = rf_estimator.fit(X, target).feature_importances_
feature_importance_dict = dict(zip(feature_columns, feature_importance))
sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
features, importances = zip(*sorted_features[:10])
plt.barh(range(len(features)), importances)
plt.yticks(range(len(features)), features)
plt.xlabel('Importance')
plt.title('Top 10 Feature Importances (from RFE)')
plt.grid(True, axis='x', alpha=0.3)

# Plot 7: Residuals vs Total Uncertainty
plt.subplot(3, 3, 7)
residuals = np.abs(y_test_inv.flatten() - y_pred_mean_inv.flatten())
plt.scatter(y_pred_std_inv, residuals, alpha=0.6, s=20, label='Total Uncertainty')
plt.xlabel('Total Prediction Uncertainty (std)')
plt.ylabel('Absolute Residual')
plt.title('Residuals vs Total Uncertainty')
plt.grid(True, alpha=0.3)
correlation_total_unc = np.corrcoef(y_pred_std_inv, residuals)[0, 1]
plt.text(0.05, 0.95, f'Corr (Total): {correlation_total_unc:.3f}', transform=plt.gca().transAxes, 
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

# Plot 8: Residuals vs Aleatoric Uncertainty
plt.subplot(3, 3, 8)
plt.scatter(y_pred_aleatoric_std_inv, residuals, alpha=0.6, s=20, color='purple', label='Aleatoric Uncertainty')
plt.xlabel('Aleatoric Uncertainty (std)')
plt.ylabel('Absolute Residual')
plt.title('Residuals vs Aleatoric Uncertainty')
plt.grid(True, alpha=0.3)
correlation_aleatoric_unc = np.corrcoef(y_pred_aleatoric_std_inv, residuals)[0, 1]
plt.text(0.05, 0.95, f'Corr (Aleatoric): {correlation_aleatoric_unc:.3f}', transform=plt.gca().transAxes, 
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

# Plot 9: Time Series of Uncertainty
plt.subplot(3, 3, 9)
plt.plot(y_pred_std_inv[:n_plot], alpha=0.7, color='orange', label='Total Uncertainty (from sampling)')
plt.plot(y_pred_aleatoric_std_inv[:n_plot], alpha=0.7, color='purple', label='Aleatoric Uncertainty (model output scale)')
plt.title('Uncertainty Over Time')
plt.xlabel('Time Steps')
plt.ylabel('Uncertainty (std)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('full_bayesian_lstm_vi_results_pytorch.png')
plt.close()

# ----------------- Model Summary -----------------
print("\n=== FULL BAYESIAN MODEL SUMMARY (PyTorch) ===")
print(f"Data points removed as outliers: {n_outliers} ({outlier_percentage:.2f}%)")
print(f"Final dataset size: {df.shape[0]} rows")
print(f"Original features: {len(feature_columns)}")
print(f"Selected features: {len(selected_features)}")
print(f"Features used: {selected_features}")
print(f"Model R² Score: {r2_bayesian:.4f}")
print(f"Model RMSE: {np.sqrt(mse_bayesian):.4f}")
print(f"Mean Total Prediction Uncertainty: {np.mean(y_pred_std_inv):.4f}")
print(f"Mean Aleatoric Uncertainty: {np.mean(y_pred_aleatoric_std_inv):.4f}")
print(f"95% Prediction Interval Coverage: {coverage:.2f}%")

# Uncertainty analysis
high_uncertainty_threshold = np.percentile(y_pred_std_inv, 90)
high_uncertainty_points = np.sum(y_pred_std_inv > high_uncertainty_threshold)
print(f"\nUncertainty Analysis:")
print(f"High total uncertainty threshold (90th percentile): {high_uncertainty_threshold:.4f}")
print(f"Points with high total uncertainty: {high_uncertainty_points} ({high_uncertainty_points/len(y_pred_std_inv)*100:.2f}%)")

print(f"\nFirst few outliers detected:")
print(outlier_info.head())


import joblib
import os

# Create a directory to save the model and scalers if it doesn't exist
save_dir = 'model_and_scalers'
os.makedirs(save_dir, exist_ok=True)

# Save the Bayesian LSTM model
model_path = os.path.join(save_dir, 'bayesian_lstm_model.pth')
torch.save(bayesian_lstm.state_dict(), model_path)
print(f"\nBayesian LSTM model saved to {model_path}")

# Save the xscaler
xscaler_path = os.path.join(save_dir, 'xscaler.joblib')
joblib.dump(xscaler, xscaler_path)
print(f"XScaler saved to {xscaler_path}")

# Save the yscaler
yscaler_path = os.path.join(save_dir, 'yscaler.joblib')
joblib.dump(yscaler, yscaler_path)
print(f"YScaler saved to {yscaler_path}")