import pandas as pd
import numpy as np
# Matplotlib is not directly used for Plotly, but kept as it was in your original code
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

# Import Plotly and Dash libraries for interactive dashboard creation
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc
from dash import html

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
pyro.set_rng_seed(42)

# --- Hampel Filter Function ---
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

# --- Load and Clean Data ---
# Ensure 'AirQualityUCI.csv' is in the same directory as this script.
try:
    df = pd.read_csv('AirQualityUCI.csv', delimiter=';')
except FileNotFoundError:
    print("Error: 'AirQualityUCI.csv' not found. Please ensure the file is in the same directory.")
    exit()

# Drop 'NMHC(GT)' column as it's often sparse or problematic
df = df.drop('NMHC(GT)', axis=1)

# Combine 'Date' and 'Time' columns into a single 'Datetime' index
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H.%M.%S')
df.set_index('Datetime', inplace=True)
df = df.drop(['Date', 'Time'], axis=1)

# Drop the last two columns, which are often unnamed and empty in this dataset
df = df.drop(df.columns[[12, 13]], axis=1)

# Convert all columns to numeric, handling comma decimals and missing values
for col in df.columns:
    df[col] = df[col].astype(str).str.replace(',', '.', regex=False).str.strip()
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna()
df.replace(-200, np.nan, inplace=True) # Replace specific placeholder for missing data
df.interpolate(method='linear', inplace=True) # Interpolate remaining NaNs

print("Available columns:", df.columns.tolist())
print(f"Data shape before outlier removal: {df.shape}")

# --- Apply Hampel Filter to Target Variable ---
target_col = 'CO(GT)'
print(f"\nApplying Hampel filter to target variable: {target_col}")

window_size = 24  # 24 hours for hourly data
n_sigma = 2.5     # 2.5 standard deviations for outlier detection

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
df_clean = df[~outlier_mask].copy() # Create a clean DataFrame by removing outlier rows
print(f"Data shape after outlier removal: {df_clean.shape}")

df = df_clean.copy() # Use the cleaned DataFrame for further processing

# --- Prepare Features and Target ---
target = df[target_col].copy()
feature_columns = [col for col in df.columns if col != target_col]
X = df[feature_columns].copy()

# Add cyclic time-based features (hour, day of week, month)
X['hour'] = X.index.hour
X['day_of_week'] = X.index.dayofweek
X['month'] = X.index.month

X['hour_sin'] = np.sin(2 * np.pi * X['hour'] / 24)
X['hour_cos'] = np.cos(2 * np.pi * X['hour'] / 24)
X['day_sin'] = np.sin(2 * np.pi * X['day_of_week'] / 7)
X['day_cos'] = np.cos(2 * np.pi * X['day_of_week'] / 7)
X['month_sin'] = np.sin(2 * np.pi * X['month'] / 12)
X['month_cos'] = np.cos(2 * np.pi * X['month'] / 12)

X = X.drop(['hour', 'day_of_week', 'month'], axis=1) # Drop original time columns
feature_columns = X.columns.tolist() # Update feature_columns list

print(f"\nOriginal number of features (including cyclic): {len(feature_columns)}")
print("Features (including cyclic):", feature_columns)

# --- Apply Basic Preprocessing (Differencing and Rolling Mean) ---
# Perform differencing with a lag of 24 (hourly data)
X = X.diff(24).dropna()
target = target.diff(24).dropna()

# Apply a 3-period rolling mean to smooth data
target = target.rolling(window=3, center=True).mean().dropna()
X = X.rolling(window=3, center=True).mean().dropna()

# Ensure X and target have a common index after all transformations
common_index = X.index.intersection(target.index)
X = X.loc[common_index]
target = target.loc[common_index]

print(f"Data points after differencing and rolling mean: {len(X)}")

# --- Recursive Feature Elimination (RFE) ---
rf_estimator = RandomForestRegressor(n_estimators=100, random_state=42)
# Select up to 10 features, or fewer if less than 10 are available
n_features_to_select = min(10, len(feature_columns))

rfe = RFE(estimator=rf_estimator, n_features_to_select=n_features_to_select, step=1)
# Fit RFE on the non-scaled data to identify important features
rfe.fit(X, target)
# Select only the features identified by RFE
X_rfe = X[X.columns[rfe.support_]]

selected_features = X_rfe.columns.tolist() # Get the names of selected features
print(f"\nSelected {len(selected_features)} features using RFE:")
print("Selected features:", selected_features)
print("Feature ranking:", dict(zip(feature_columns, rfe.ranking_)))

# --- Scale Data ---
xscaler = StandardScaler()
yscaler = StandardScaler()

X_scaled = xscaler.fit_transform(X_rfe)
y_scaled = yscaler.fit_transform(target.values.reshape(-1, 1)).flatten()

# --- Sequence Creation for LSTM ---
def create_sequences(X, y, seq_len):
    """
    Creates sequences for LSTM input from time series data.
    """
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(Xs), np.array(ys)

SEQ_LEN = 24 # Sequence length for LSTM (24 hours)
X_seq, y_seq = create_sequences(X_scaled, y_scaled, SEQ_LEN)

print(f"\nSequence shape: {X_seq.shape}")
print(f"Target shape: {y_seq.shape}")

# --- Split Data into Training, Validation, and Test Sets ---
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

# --- Bayesian LSTM Model Definition ---
class BayesianLSTM(pyro.nn.PyroModule):
    def __init__(self, input_dim, hidden_dim1=352, hidden_dim2=128, hidden_dim3=64, dense_dim1=72, dense_dim2=32):
        super(BayesianLSTM, self).__init__()
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.hidden_dim3 = hidden_dim3
        
        # LSTM layers with Batch Normalization and Bidirectional processing
        self.lstm1 = pyro.nn.PyroModule[nn.LSTM](input_dim, hidden_dim1, batch_first=True, bidirectional=True)
        self.bn1 = pyro.nn.PyroModule[nn.BatchNorm1d](hidden_dim1 * 2) # *2 for bidirectional
        self.lstm2 = pyro.nn.PyroModule[nn.LSTM](hidden_dim1 * 2, hidden_dim2, batch_first=True, bidirectional=True)
        self.bn2 = pyro.nn.PyroModule[nn.BatchNorm1d](hidden_dim2 * 2)
        self.lstm3 = pyro.nn.PyroModule[nn.LSTM](hidden_dim2 * 2, hidden_dim3, batch_first=True, bidirectional=True)
        self.bn3 = pyro.nn.PyroModule[nn.BatchNorm1d](hidden_dim3 * 2)
        
        # Dense layers for final prediction
        self.dense1 = pyro.nn.PyroModule[nn.Linear](hidden_dim3 * 2, dense_dim1)
        self.dense2 = pyro.nn.PyroModule[nn.Linear](dense_dim1, dense_dim2)
        self.dense_out = pyro.nn.PyroModule[nn.Linear](dense_dim2, 2)  # Predict mean and log-variance
    
    def forward(self, x, y=None):
        # Pass input through LSTM layers
        x, _ = self.lstm1(x)
        x = x.permute(0, 2, 1)  # Reshape for BatchNorm (batch, features, seq_len)
        x = self.bn1(x).permute(0, 2, 1) # Reshape back
        
        x, _ = self.lstm2(x)
        x = x.permute(0, 2, 1)
        x = self.bn2(x).permute(0, 2, 1)
        
        x, _ = self.lstm3(x)
        x = x[:, -1, :]  # Take the output of the last time step
        x = self.bn3(x) # Apply BatchNorm to the last time step's output
        
        # Pass through dense layers with ReLU activation
        x = torch.relu(self.dense1(x))
        x = torch.relu(self.dense2(x))
        output = self.dense_out(x)  # Outputs [mean, log-variance]
            
        mean, log_var = output[:, 0], output[:, 1]
        sigma = torch.exp(0.5 * log_var) + 1e-5  # Convert log-variance to standard deviation, ensuring positivity
                
        # Define the observation distribution (likelihood) using Pyro's plate for batching
        with pyro.plate("data", x.shape[0]):
            pyro.sample("obs", dist.Normal(mean, sigma), obs=y)
                
        return mean, sigma

    def model(self, x, y=None):
        # Define priors for all model parameters. This is the generative model.
        for name, param in self.named_parameters():
            # Spherical Gaussian prior for each parameter
            prior = dist.Normal(0.0, 1.0).expand(param.shape).to_event(len(param.shape))
            pyro.sample(f"prior_{name}", prior)
        
        # Forward pass through the network to get mean and sigma for the likelihood
        mean, sigma = self.forward(x, y)
        return mean, sigma

    def guide(self, x, y=None):
        # Define variational distributions (approximate posteriors) for each parameter.
        # These are learned during SVI.
        for name, param in self.named_parameters():
            # Learnable mean (loc) and standard deviation (scale) for each parameter's distribution
            loc = pyro.param(f"{name}_loc", torch.randn_like(param))
            scale = pyro.param(f"{name}_scale", torch.ones_like(param) * 0.1, constraint=dist.constraints.positive)
            pyro.sample(f"prior_{name}", dist.Normal(loc, scale).to_event(len(param.shape)))

# --- Initialize Model and Optimizer ---
bayesian_lstm = BayesianLSTM(input_dim=X_train.shape[2]) # Input dimension is number of features
svi = SVI(bayesian_lstm.model, bayesian_lstm.guide, pyro.optim.Adam({"lr": 0.001}), loss=Trace_ELBO())

# --- Training Loop Function ---
def train_model(model, x_train, y_train, x_val, y_val, epochs=100, batch_size=32):
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training phase: Iterate over minibatches
        epoch_train_loss = 0.0
        permutation = torch.randperm(x_train.shape[0])
        for i in range(0, x_train.shape[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = x_train[indices], y_train[indices]
            loss = svi.step(batch_x, batch_y) # Perform one SVI step (gradient update)
            epoch_train_loss += loss / batch_x.shape[0] # Accumulate loss
        
        train_losses.append(epoch_train_loss / (x_train.shape[0] // batch_size))
        
        # Validation phase: Evaluate loss on validation set
        val_loss = 0.0
        with torch.no_grad(): # No gradient calculation needed for validation
            val_loss = svi.evaluate_loss(x_val, y_val) / x_val.shape[0]
        val_losses.append(val_loss)
        
        # Simple early stopping: Stop if validation loss increases for 20 epochs
        if epoch > 20 and val_losses[-1] > val_losses[-2]:
            print(f"Early stopping at epoch {epoch} due to increasing validation loss.")
            break
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")
    
    return train_losses, val_losses

# Train the model
print("\n--- Starting Model Training ---")
train_losses, val_losses = train_model(bayesian_lstm, X_train, y_train, X_val, y_val, epochs=100)
print("--- Model Training Complete ---")

# --- Bayesian Predictions with Variational Inference ---
def predict_with_variational_inference(model, x_data, n_samples=100):
    """
    Generates predictions and uncertainty estimates from a Bayesian model using VI.
    """
    # Create a Predictive object to draw samples from the posterior predictive distribution
    predictive = pyro.infer.Predictive(model.model, guide=model.guide, num_samples=n_samples)
    samples = predictive(x_data)
    
    # Extract predicted observations (y_pred)
    y_pred = samples["obs"].detach().numpy()
    if y_pred.shape[-1] == 1:
        y_pred = y_pred.squeeze(-1) # Remove last dimension if it's singleton
    
    # Calculate mean, standard deviation, and credible intervals from the samples
    mean_pred = y_pred.mean(axis=0)
    std_pred = y_pred.std(axis=0) # Total uncertainty (epistemic + aleatoric)
    lower_bound = np.percentile(y_pred, 2.5, axis=0) # 2.5th percentile for 95% CI
    upper_bound = np.percentile(y_pred, 97.5, axis=0) # 97.5th percentile for 95% CI
    
    # Estimate aleatoric uncertainty directly from the model's output (sigma)
    with torch.no_grad():
        _, sigma = model.forward(x_data)
    
    return mean_pred, std_pred, lower_bound, upper_bound, sigma.numpy()

# Generate predictions on the test set
print("\n--- Generating Predictions on Test Set ---")
y_pred_mean, y_pred_std, y_pred_lower, y_pred_upper, y_pred_aleatoric_std = predict_with_variational_inference(
    bayesian_lstm, X_test, n_samples=100
)

# Inverse transform predictions and actual values back to original scale
y_test_inv = yscaler.inverse_transform(y_test.numpy().reshape(-1, 1)).flatten()
y_pred_mean_inv = yscaler.inverse_transform(y_pred_mean.reshape(-1, 1)).flatten()
# Standard deviations are scaled by the scaler's scale factor
y_pred_std_inv = yscaler.scale_ * y_pred_std.flatten()
y_pred_lower_inv = yscaler.inverse_transform(y_pred_lower.reshape(-1, 1)).flatten()
y_pred_upper_inv = yscaler.inverse_transform(y_pred_upper.reshape(-1, 1)).flatten()
y_pred_aleatoric_std_inv = yscaler.scale_ * y_pred_aleatoric_std.flatten()

# Calculate performance metrics
r2_bayesian = r2_score(y_test_inv, y_pred_mean_inv)
mse_bayesian = mean_squared_error(y_test_inv, y_pred_mean_inv)
rmse_bayesian = np.sqrt(mse_bayesian)

# Calculate 95% prediction interval coverage
within_interval = np.sum((y_test_inv.flatten() >= y_pred_lower_inv.flatten()) &
                        (y_test_inv.flatten() <= y_pred_upper_inv.flatten()))
coverage = within_interval / len(y_test_inv) * 100

print(f"\n=== MODEL PERFORMANCE (Variational Inference) ===")
print(f"  R2 Score: {r2_bayesian:.4f}")
print(f"  MSE: {mse_bayesian:.4f}")
print(f"  RMSE: {rmse_bayesian:.4f}")
print(f"  Mean Total Prediction Uncertainty (std): {np.mean(y_pred_std_inv):.4f}")
print(f"  Mean Aleatoric Uncertainty (std from output layer): {np.mean(y_pred_aleatoric_std_inv):.4f}")
print(f"  95% Prediction Interval Coverage: {coverage:.2f}%")

# Additional calculations for specific plots (Residuals and Feature Importance)
residuals = np.abs(y_test_inv.flatten() - y_pred_mean_inv.flatten())
correlation_total_unc = np.corrcoef(y_pred_std_inv, residuals)[0, 1]
correlation_aleatoric_unc = np.corrcoef(y_pred_aleatoric_std_inv, residuals)[0, 1]

# Re-calculate feature importance using the original X and target (before sequencing/scaling)
# This aligns with the original matplotlib plot's approach.
rf_estimator_for_importance = RandomForestRegressor(n_estimators=100, random_state=42)
rf_estimator_for_importance.fit(X, target) # X here is after differencing/rolling mean, with cyclic features
feature_importance = rf_estimator_for_importance.feature_importances_
feature_importance_dict = dict(zip(X.columns.tolist(), feature_importance))
sorted_features_for_plot = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
features_plot, importances_plot = zip(*sorted_features_for_plot[:10]) # Top 10 features


# --- Create Plotly Figures for Dash Dashboard ---

# Figure 1: Predictions with Uncertainty Bands (Time Series Plot)
fig_predictions = go.Figure()
n_plot = min(500, len(y_test_inv)) # Limit points for clearer visualization in dashboard

fig_predictions.add_trace(go.Scatter(
    x=np.arange(n_plot),
    y=y_test_inv[:n_plot],
    mode='lines',
    name='Actual',
    line=dict(color='#3498DB', width=2)
))

fig_predictions.add_trace(go.Scatter(
    x=np.arange(n_plot),
    y=y_pred_mean_inv[:n_plot],
    mode='lines',
    name='Bayesian Pred (Mean)',
    line=dict(color='#E74C3C', width=2)
))

# Upper bound for fill_between
fig_predictions.add_trace(go.Scatter(
    x=np.arange(n_plot),
    y=y_pred_upper_inv[:n_plot].flatten(),
    mode='lines',
    line=dict(width=0),
    marker=dict(color="#444"),
    showlegend=False,
    hoverinfo='skip'
))
# Lower bound with fill to next trace (upper bound) for credible interval
fig_predictions.add_trace(go.Scatter(
    x=np.arange(n_plot),
    y=y_pred_lower_inv[:n_plot].flatten(),
    line=dict(width=0),
    mode='lines',
    fillcolor='rgba(231,76,60,0.2)', # Semi-transparent red
    fill='tonexty',
    name='95% Credible Interval',
    hoverinfo='skip'
))

fig_predictions.update_layout(
    title_text="Bayesian LSTM Predictions with Uncertainty",
    xaxis_title="Time Steps",
    yaxis_title="CO(GT) Concentration",
    hovermode="x unified", # Shows all trace values at hovered x-coordinate
    template="plotly_white",
    height=450,
    margin=dict(l=40, r=40, t=60, b=40),
    legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.7)', bordercolor='rgba(0,0,0,0.1)', borderwidth=1),
    uirevision='constant' # Keeps zoom/pan state stable on updates
)

# Figure 2: Training History (Loss over Epochs)
fig_training_history = go.Figure()
fig_training_history.add_trace(go.Scatter(
    x=np.arange(len(train_losses)),
    y=train_losses,
    mode='lines',
    name='Training Loss',
    line=dict(color='#28B463') # Green
))
fig_training_history.add_trace(go.Scatter(
    x=np.arange(len(val_losses)),
    y=val_losses,
    mode='lines',
    name='Validation Loss',
    line=dict(color='#F39C12') # Orange
))
fig_training_history.update_layout(
    title_text="Model Training History",
    xaxis_title="Epochs",
    yaxis_title="Negative Log Likelihood Loss",
    template="plotly_white",
    height=450,
    margin=dict(l=40, r=40, t=60, b=40),
    legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.7)', bordercolor='rgba(0,0,0,0.1)', borderwidth=1)
)

# Figure 3: Uncertainty Distribution (Histograms for Total and Aleatoric)
fig_uncertainty_dist = go.Figure()
fig_uncertainty_dist.add_trace(go.Histogram(
    x=y_pred_std_inv,
    name='Total Uncertainty',
    marker_color='#F39C12', # Orange
    opacity=0.7,
    nbinsx=50
))
fig_uncertainty_dist.add_trace(go.Histogram(
    x=y_pred_aleatoric_std_inv,
    name='Aleatoric Uncertainty',
    marker_color='#8E44AD', # Purple
    opacity=0.7,
    nbinsx=50
))
fig_uncertainty_dist.update_layout(
    barmode='overlay', # Overlap histograms
    title_text="Distribution of Prediction Uncertainty",
    xaxis_title="Standard Deviation",
    yaxis_title="Frequency",
    template="plotly_white",
    height=450,
    margin=dict(l=40, r=40, t=60, b=40),
    legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.7)', bordercolor='rgba(0,0,0,0.1)', borderwidth=1)
)

# Figure 4: Scatter Plot (Actual vs. Predicted with Error Bars)
fig_scatter = go.Figure()
# Use a subsample for better performance and clarity on scatter plot
sample_indices_scatter = np.random.choice(len(y_test_inv), size=min(1000, len(y_test_inv)), replace=False)

# Add scatter plot with error bars (2 * std for approximate 95% interval)
fig_scatter.add_trace(go.Scatter(
    x=y_test_inv[sample_indices_scatter],
    y=y_pred_mean_inv[sample_indices_scatter],
    mode='markers',
    name='Predictions',
    marker=dict(color='#E74C3C', size=5, opacity=0.6),
    error_y=dict(
        type='data',
        array=2 * y_pred_std_inv[sample_indices_scatter],
        visible=True,
        color='#E74C3C',
        thickness=1
    )
))
# Add ideal prediction line (y=x)
fig_scatter.add_trace(go.Scatter(
    x=[y_test_inv.min(), y_test_inv.max()],
    y=[y_test_inv.min(), y_test_inv.max()],
    mode='lines',
    name='Ideal Prediction',
    line=dict(color='#2C3E50', dash='dash', width=2)
))
fig_scatter.update_layout(
    title_text=f"Actual vs. Predicted (R² = {r2_bayesian:.4f})",
    xaxis_title="Actual CO(GT) Concentration",
    yaxis_title="Predicted CO(GT) Concentration",
    template="plotly_white",
    showlegend=True,
    height=450,
    margin=dict(l=40, r=40, t=60, b=40),
    legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.7)', bordercolor='rgba(0,0,0,0.1)', borderwidth=1)
)

# Figure 5: Feature Importance Bar Chart
fig_feature_importance = go.Figure(go.Bar(
    x=importances_plot,
    y=[str(f) for f in features_plot], # Ensure feature names are strings for y-axis
    orientation='h', # Horizontal bar chart
    marker_color='#1ABC9C' # Turquoise
))
fig_feature_importance.update_layout(
    title_text='Top 10 Feature Importances (from RandomForestRegressor)',
    xaxis_title='Importance',
    yaxis_title='Feature',
    yaxis=dict(autorange="reversed"), # To display highest importance at the top
    template="plotly_white",
    height=450,
    margin=dict(l=100, r=40, t=60, b=40) # More left margin for long feature names
)


# --- Dash App Layout ---
# Initialize the Dash application
app = dash.Dash(__name__)

app.layout = html.Div(style={
    'fontFamily': 'Inter, sans-serif',
    'backgroundColor': '#F8F9FA',
    'color': '#333',
    'padding': '15px'
}, children=[
    html.H1("Bayesian LSTM Air Quality Prediction Dashboard",
            style={'textAlign': 'center', 'color': '#2C3E50', 'margin-top': '20px', 'font-size': '2.5em'}),

    # Section for overall summaries (model performance and data preprocessing)
    html.Div(
        style={'display': 'flex', 'flex-wrap': 'wrap', 'justify-content': 'space-around', 'padding': '10px'},
        children=[
            # Model Performance Metrics Card
            html.Div([
                html.H3("Model Performance Overview", style={'color': '#34495E', 'margin-bottom': '15px'}),
                html.P(f"• R² Score: {r2_bayesian:.4f}", style={'font-size': '1.1em', 'margin-bottom': '5px'}),
                html.P(f"• MSE: {mse_bayesian:.4f}", style={'font-size': '1.1em', 'margin-bottom': '5px'}),
                html.P(f"• RMSE: {rmse_bayesian:.4f}", style={'font-size': '1.1em', 'margin-bottom': '5px'}),
                html.P(f"• 95% Prediction Interval Coverage: {coverage:.2f}%", style={'font-size': '1.1em', 'margin-bottom': '5px'}),
                html.P(f"• Mean Total Prediction Uncertainty (std): {np.mean(y_pred_std_inv):.4f}", style={'font-size': '1.1em', 'margin-bottom': '5px'}),
                html.P(f"• Mean Aleatoric Uncertainty (std): {np.mean(y_pred_aleatoric_std_inv):.4f}", style={'font-size': '1.1em'}),
            ], style={
                'background-color': '#ECF0F1', 'padding': '25px', 'border-radius': '12px',
                'box-shadow': '0 4px 12px rgba(0,0,0,0.1)', 'margin': '15px', 'flex': '1 1 35%',
                'min-width': '320px', 'transition': 'all 0.3s ease-in-out',
                'border': '1px solid #DCDFE4'
            }),
            # Data Preprocessing Summary Card
            html.Div([
                html.H3("Data Preprocessing Summary", style={'color': '#34495E', 'margin-bottom': '15px'}),
                html.P(f"• Data points removed as outliers: {n_outliers} ({outlier_percentage:.2f}%)", style={'font-size': '1.1em', 'margin-bottom': '5px'}),
                html.P(f"• Final dataset size for modeling: {len(X_seq)} sequences", style={'font-size': '1.1em', 'margin-bottom': '5px'}),
                html.P(f"• Number of selected features: {len(selected_features)}", style={'font-size': '1.1em', 'margin-bottom': '5px'}),
                html.P(f"• Selected features: {', '.join(selected_features)}", style={'font-size': '1.1em', 'word-break': 'break-word'}),
            ], style={
                'background-color': '#ECF0F1', 'padding': '25px', 'border-radius': '12px',
                'box-shadow': '0 4px 12px rgba(0,0,0,0.1)', 'margin': '15px', 'flex': '1 1 55%',
                'min-width': '450px', 'transition': 'all 0.3s ease-in-out',
                'border': '1px solid #DCDFE4'
            })
        ]),

    # Row 1 of graphs (Predictions and Training History)
    html.Div(
        style={'display': 'flex', 'flex-wrap': 'wrap', 'justify-content': 'space-around', 'padding': '10px'},
        children=[
            dcc.Graph(figure=fig_predictions, style={'width': '48%', 'margin': '1%', 'box-shadow': '0 4px 12px rgba(0,0,0,0.1)', 'border-radius': '12px', 'overflow': 'hidden'}),
            dcc.Graph(figure=fig_training_history, style={'width': '48%', 'margin': '1%', 'box-shadow': '0 4px 12px rgba(0,0,0,0.1)', 'border-radius': '12px', 'overflow': 'hidden'})
        ]),

    # Row 2 of graphs (Uncertainty Distribution and Actual vs. Predicted Scatter)
    html.Div(
        style={'display': 'flex', 'flex-wrap': 'wrap', 'justify-content': 'space-around', 'padding': '10px'},
        children=[
            dcc.Graph(figure=fig_uncertainty_dist, style={'width': '48%', 'margin': '1%', 'box-shadow': '0 4px 12px rgba(0,0,0,0.1)', 'border-radius': '12px', 'overflow': 'hidden'}),
            dcc.Graph(figure=fig_scatter, style={'width': '48%', 'margin': '1%', 'box-shadow': '0 4px 12px rgba(0,0,0,0.1)', 'border-radius': '12px', 'overflow': 'hidden'})
        ]),
    
    # Row 3 of graphs (Feature Importance) - adjusted for a single plot
    html.Div(
        style={'display': 'flex', 'flex-wrap': 'wrap', 'justify-content': 'center', 'padding': '10px'},
        children=[
            dcc.Graph(figure=fig_feature_importance, style={'width': '48%', 'margin': '1%', 'box-shadow': '0 4px 12px rgba(0,0,0,0.1)', 'border-radius': '12px', 'overflow': 'hidden'})
        ]),
    
    html.Div(
        html.P("This dashboard visualizes the performance and uncertainty of a Bayesian LSTM model for air quality prediction.",
               style={'textAlign': 'center', 'margin-top': '40px', 'color': '#7F8C8D', 'font-size': '0.9em'}),
        style={'padding': '20px'}
    )
])

# # --- Run the Dash App ---
# if __name__ == '__main__':
#     import os
#     print("\n--- Starting Dash App ---")
#     print("Open your web browser and navigate to http://127.0.0.1:8050/ to view the dashboard.")
#     app.run(debug=False, port=int(os.environ.get("PORT", 8050)))

server = app.server  # <- This is key for deployment with gunicorn

if __name__ == '__main__':
    app.run_server(debug=True)