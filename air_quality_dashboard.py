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
import joblib
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html

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
try:
    df = pd.read_csv('AirQualityUCI.csv', delimiter=';')
except FileNotFoundError:
    print("Error: 'AirQualityUCI.csv' not found. Please ensure the file is in the same directory.")
    exit()

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
window_size = 24
n_sigma = 2.5
outlier_mask, filtered_target = hampel_filter(df[target_col], window_size=window_size, n_sigma=n_sigma)
n_outliers = outlier_mask.sum()
outlier_percentage = (n_outliers / len(df)) * 100
print(f"Detected {n_outliers} outliers ({outlier_percentage:.2f}% of data)")
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

# ----------------- Load Scalers -----------------
save_dir = 'model_and_scalers'
xscaler_path = os.path.join(save_dir, 'xscaler.joblib')
yscaler_path = os.path.join(save_dir, 'yscaler.joblib')
try:
    xscaler = joblib.load(xscaler_path)
    yscaler = joblib.load(yscaler_path)
    print(f"\nLoaded xscaler from {xscaler_path}")
    print(f"Loaded yscaler from {yscaler_path}")
except FileNotFoundError:
    print("Error: Scaler files not found. Please ensure 'xscaler.joblib' and 'yscaler.joblib' are in the 'model_and_scalers' directory.")
    exit()

X_scaled = xscaler.transform(X_rfe)
y_scaled = yscaler.transform(target.values.reshape(-1, 1)).flatten()

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
        self.lstm1 = pyro.nn.PyroModule[nn.LSTM](input_dim, hidden_dim1, batch_first=True, bidirectional=True)
        self.bn1 = pyro.nn.PyroModule[nn.BatchNorm1d](hidden_dim1 * 2)
        self.lstm2 = pyro.nn.PyroModule[nn.LSTM](hidden_dim1 * 2, hidden_dim2, batch_first=True, bidirectional=True)
        self.bn2 = pyro.nn.PyroModule[nn.BatchNorm1d](hidden_dim2 * 2)
        self.lstm3 = pyro.nn.PyroModule[nn.LSTM](hidden_dim2 * 2, hidden_dim3, batch_first=True, bidirectional=True)
        self.bn3 = pyro.nn.PyroModule[nn.BatchNorm1d](hidden_dim3 * 2)
        self.dense1 = pyro.nn.PyroModule[nn.Linear](hidden_dim3 * 2, dense_dim1)
        self.dense2 = pyro.nn.PyroModule[nn.Linear](dense_dim1, dense_dim2)
        self.dense_out = pyro.nn.PyroModule[nn.Linear](dense_dim2, 2)
    
    def forward(self, x, y=None):
        x, _ = self.lstm1(x)
        x = x.permute(0, 2, 1)
        x = self.bn1(x).permute(0, 2, 1)
        x, _ = self.lstm2(x)
        x = x.permute(0, 2, 1)
        x = self.bn2(x).permute(0, 2, 1)
        x, _ = self.lstm3(x)
        x = x[:, -1, :]
        x = self.bn3(x)
        x = torch.relu(self.dense1(x))
        x = torch.relu(self.dense2(x))
        output = self.dense_out(x)
        mean, log_var = output[:, 0], output[:, 1]
        sigma = torch.exp(0.5 * log_var) + 1e-5
        with pyro.plate("data", x.shape[0]):
            pyro.sample("obs", dist.Normal(mean, sigma), obs=y)
        return mean, sigma

    def model(self, x, y=None):
        for name, param in self.named_parameters():
            prior = dist.Normal(0.0, 1.0).expand(param.shape).to_event(len(param.shape))
            pyro.sample(f"prior_{name}", prior)
        mean, sigma = self.forward(x, y)
        return mean, sigma

    def guide(self, x, y=None):
        for name, param in self.named_parameters():
            loc = pyro.param(f"{name}_loc", torch.randn_like(param))
            scale = pyro.param(f"{name}_scale", torch.ones_like(param) * 0.1, constraint=dist.constraints.positive)
            pyro.sample(f"prior_{name}", dist.Normal(loc, scale).to_event(len(param.shape)))

# ----------------- Load Model -----------------
bayesian_lstm = BayesianLSTM(input_dim=len(selected_features))
model_path = os.path.join(save_dir, 'bayesian_lstm_model.pth')
try:
    bayesian_lstm.load_state_dict(torch.load(model_path))
    print(f"Loaded Bayesian LSTM model from {model_path}")
except FileNotFoundError:
    print(f"Error: Model file not found at {model_path}. Please ensure the file exists.")
    exit()
bayesian_lstm.eval()

# ----------------- Bayesian Predictions with VI -----------------
def predict_with_variational_inference(model, x_data, n_samples=100):
    predictive = pyro.infer.Predictive(model.model, guide=model.guide, num_samples=n_samples)
    samples = predictive(x_data)
    y_pred = samples["obs"].detach().numpy()
    if y_pred.shape[-1] == 1:
        y_pred = y_pred.squeeze(-1)
    mean_pred = y_pred.mean(axis=0)
    std_pred = y_pred.std(axis=0)
    lower_bound = np.percentile(y_pred, 2.5, axis=0)
    upper_bound = np.percentile(y_pred, 97.5, axis=0)
    with torch.no_grad():
        mean, sigma = model.forward(x_data)
    return mean_pred, std_pred, lower_bound, upper_bound, sigma.numpy()

# Generate predictions
print("\n--- Generating Predictions on Test Set ---")
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
rmse_bayesian = np.sqrt(mse_bayesian)
within_interval = np.sum((y_test_inv.flatten() >= y_pred_lower_inv.flatten()) & 
                        (y_test_inv.flatten() <= y_pred_upper_inv.flatten()))
coverage = within_interval / len(y_test_inv) * 100

# Additional calculations for plots
residuals = np.abs(y_test_inv.flatten() - y_pred_mean_inv.flatten())
correlation_total_unc = np.corrcoef(y_pred_std_inv, residuals)[0, 1]
correlation_aleatoric_unc = np.corrcoef(y_pred_aleatoric_std_inv, residuals)[0, 1]

# Feature importance for visualization
rf_estimator_for_importance = RandomForestRegressor(n_estimators=100, random_state=42)
rf_estimator_for_importance.fit(X, target)
feature_importance = rf_estimator_for_importance.feature_importances_
feature_importance_dict = dict(zip(X.columns.tolist(), feature_importance))
sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
features_plot, importances_plot = zip(*sorted_features[:10])

# ----------------- Plotly Figures for Dash Dashboard -----------------
# Figure 1: Predictions with Uncertainty Bands
fig_predictions = go.Figure()
n_plot = min(500, len(y_test_inv))
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
fig_predictions.add_trace(go.Scatter(
    x=np.arange(n_plot),
    y=y_pred_upper_inv[:n_plot].flatten(),
    mode='lines',
    line=dict(width=0),
    marker=dict(color="#444"),
    showlegend=False,
    hoverinfo='skip'
))
fig_predictions.add_trace(go.Scatter(
    x=np.arange(n_plot),
    y=y_pred_lower_inv[:n_plot].flatten(),
    line=dict(width=0),
    mode='lines',
    fillcolor='rgba(231,76,60,0.2)',
    fill='tonexty',
    name='95% Credible Interval',
    hoverinfo='skip'
))
fig_predictions.update_layout(
    title_text="Bayesian LSTM Predictions with Uncertainty",
    xaxis_title="Time Steps",
    yaxis_title="CO(GT) Concentration",
    hovermode="x unified",
    template="plotly_white",
    height=450,
    margin=dict(l=40, r=40, t=60, b=40),
    legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.7)', bordercolor='rgba(0,0,0,0.1)', borderwidth=1)
)

# Figure 2: Uncertainty Distribution
fig_uncertainty_dist = go.Figure()
fig_uncertainty_dist.add_trace(go.Histogram(
    x=y_pred_std_inv,
    name='Total Uncertainty',
    marker_color='#F39C12',
    opacity=0.7,
    nbinsx=50
))
fig_uncertainty_dist.add_trace(go.Histogram(
    x=y_pred_aleatoric_std_inv,
    name='Aleatoric Uncertainty',
    marker_color='#8E44AD',
    opacity=0.7,
    nbinsx=50
))
fig_uncertainty_dist.update_layout(
    barmode='overlay',
    title_text="Distribution of Prediction Uncertainty",
    xaxis_title="Standard Deviation",
    yaxis_title="Frequency",
    template="plotly_white",
    height=450,
    margin=dict(l=40, r=40, t=60, b=40),
    legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.7)', bordercolor='rgba(0,0,0,0.1)', borderwidth=1)
)

# Figure 3: Scatter Plot with Error Bars
fig_scatter = go.Figure()
sample_indices_scatter = np.random.choice(len(y_test_inv), size=min(1000, len(y_test_inv)), replace=False)
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
    height=450,
    margin=dict(l=40, r=40, t=60, b=40),
    legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.7)', bordercolor='rgba(0,0,0,0.1)', borderwidth=1)
)

# Figure 4: Feature Importance
fig_feature_importance = go.Figure(go.Bar(
    x=importances_plot,
    y=[str(f) for f in features_plot],
    orientation='h',
    marker_color='#1ABC9C'
))
fig_feature_importance.update_layout(
    title_text='Top 10 Feature Importances (from RandomForestRegressor)',
    xaxis_title='Importance',
    yaxis_title='Feature',
    yaxis=dict(autorange="reversed"),
    template="plotly_white",
    height=450,
    margin=dict(l=100, r=40, t=60, b=40)
)

# Figure 5: Prediction Interval Coverage Over Time
coverage_by_time = []
window_size = 50
for i in range(0, len(y_test_inv) - window_size, window_size):
    end_idx = i + window_size
    window_coverage = np.sum((y_test_inv[i:end_idx].flatten() >= y_pred_lower_inv[i:end_idx].flatten()) & 
                           (y_test_inv[i:end_idx].flatten() <= y_pred_upper_inv[i:end_idx].flatten()))
    coverage_by_time.append(window_coverage / window_size * 100)
fig_coverage = go.Figure()
fig_coverage.add_trace(go.Scatter(
    x=np.arange(len(coverage_by_time)),
    y=coverage_by_time,
    mode='lines+markers',
    name='Coverage',
    line=dict(color='#3498DB')
))
fig_coverage.add_trace(go.Scatter(
    x=[0, len(coverage_by_time)-1],
    y=[95, 95],
    mode='lines',
    name='Target 95%',
    line=dict(color='#E74C3C', dash='dash')
))
fig_coverage.update_layout(
    title_text=f"Prediction Interval Coverage (Average: {coverage:.1f}%)",
    xaxis_title=f"Time Windows (size={window_size})",
    yaxis_title="Coverage (%)",
    template="plotly_white",
    height=450,
    margin=dict(l=40, r=40, t=60, b=40),
    legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.7)', bordercolor='rgba(0,0,0,0.1)', borderwidth=1)
)

# Figure 6: Residuals vs Total Uncertainty
fig_residuals_total = go.Figure()
fig_residuals_total.add_trace(go.Scatter(
    x=y_pred_std_inv,
    y=residuals,
    mode='markers',
    name='Total Uncertainty',
    marker=dict(color='#F39C12', size=5, opacity=0.6)
))
fig_residuals_total.update_layout(
    title_text=f"Residuals vs Total Uncertainty (Corr: {correlation_total_unc:.3f})",
    xaxis_title="Total Prediction Uncertainty (std)",
    yaxis_title="Absolute Residual",
    template="plotly_white",
    height=450,
    margin=dict(l=40, r=40, t=60, b=40)
)

# Figure 7: Residuals vs Aleatoric Uncertainty
fig_residuals_aleatoric = go.Figure()
fig_residuals_aleatoric.add_trace(go.Scatter(
    x=y_pred_aleatoric_std_inv,
    y=residuals,
    mode='markers',
    name='Aleatoric Uncertainty',
    marker=dict(color='#8E44AD', size=5, opacity=0.6)
))
fig_residuals_aleatoric.update_layout(
    title_text=f"Residuals vs Aleatoric Uncertainty (Corr: {correlation_aleatoric_unc:.3f})",
    xaxis_title="Aleatoric Uncertainty (std)",
    yaxis_title="Absolute Residual",
    template="plotly_white",
    height=450,
    margin=dict(l=40, r=40, t=60, b=40)
)

# Figure 8: Uncertainty Over Time
fig_uncertainty_time = go.Figure()
fig_uncertainty_time.add_trace(go.Scatter(
    x=np.arange(n_plot),
    y=y_pred_std_inv[:n_plot],
    mode='lines',
    name='Total Uncertainty',
    line=dict(color='#F39C12')
))
fig_uncertainty_time.add_trace(go.Scatter(
    x=np.arange(n_plot),
    y=y_pred_aleatoric_std_inv[:n_plot],
    mode='lines',
    name='Aleatoric Uncertainty',
    line=dict(color='#8E44AD')
))
fig_uncertainty_time.update_layout(
    title_text="Uncertainty Over Time",
    xaxis_title="Time Steps",
    yaxis_title="Uncertainty (std)",
    template="plotly_white",
    height=450,
    margin=dict(l=40, r=40, t=60, b=40),
    legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.7)', bordercolor='rgba(0,0,0,0.1)', borderwidth=1)
)

# ----------------- Dash App Layout -----------------
app = dash.Dash(__name__)
server = app.server

app.layout = html.Div(style={
    'fontFamily': 'Inter, sans-serif',
    'backgroundColor': '#F8F9FA',
    'color': '#333',
    'padding': '15px'
}, children=[
    html.H1("Bayesian LSTM Air Quality Prediction Dashboard",
            style={'textAlign': 'center', 'color': '#2C3E50', 'margin-top': '20px', 'font-size': '2.5em'}),
    html.Div(
        style={'display': 'flex', 'flex-wrap': 'wrap', 'justify-content': 'space-around', 'padding': '10px'},
        children=[
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
    html.Div(
        style={'display': 'flex', 'flex-wrap': 'wrap', 'justify-content': 'space-around', 'padding': '10px'},
        children=[
            dcc.Graph(figure=fig_predictions, style={'width': '48%', 'margin': '1%', 'box-shadow': '0 4px 12px rgba(0,0,0,0.1)', 'border-radius': '12px', 'overflow': 'hidden'}),
            dcc.Graph(figure=fig_uncertainty_dist, style={'width': '48%', 'margin': '1%', 'box-shadow': '0 4px 12px rgba(0,0,0,0.1)', 'border-radius': '12px', 'overflow': 'hidden'})
        ]),
    html.Div(
        style={'display': 'flex', 'flex-wrap': 'wrap', 'justify-content': 'space-around', 'padding': '10px'},
        children=[
            dcc.Graph(figure=fig_scatter, style={'width': '48%', 'margin': '1%', 'box-shadow': '0 4px 12px rgba(0,0,0,0.1)', 'border-radius': '12px', 'overflow': 'hidden'}),
            dcc.Graph(figure=fig_feature_importance, style={'width': '48%', 'margin': '1%', 'box-shadow': '0 4px 12px rgba(0,0,0,0.1)', 'border-radius': '12px', 'overflow': 'hidden'})
        ]),
    html.Div(
        style={'display': 'flex', 'flex-wrap': 'wrap', 'justify-content': 'space-around', 'padding': '10px'},
        children=[
            dcc.Graph(figure=fig_coverage, style={'width': '48%', 'margin': '1%', 'box-shadow': '0 4px 12px rgba(0,0,0,0.1)', 'border-radius': '12px', 'overflow': 'hidden'}),
            dcc.Graph(figure=fig_residuals_total, style={'width': '48%', 'margin': '1%', 'box-shadow': '0 4px 12px rgba(0,0,0,0.1)', 'border-radius': '12px', 'overflow': 'hidden'})
        ]),
    html.Div(
        style={'display': 'flex', 'flex-wrap': 'wrap', 'justify-content': 'center', 'padding': '10px'},
        children=[
            dcc.Graph(figure=fig_residuals_aleatoric, style={'width': '48%', 'margin': '1%', 'box-shadow': '0 4px 12px rgba(0,0,0,0.1)', 'border-radius': '12px', 'overflow': 'hidden'}),
            dcc.Graph(figure=fig_uncertainty_time, style={'width': '48%', 'margin': '1%', 'box-shadow': '0 4px 12px rgba(0,0,0,0.1)', 'border-radius': '12px', 'overflow': 'hidden'})
        ]),
    html.Div(
        html.P("This dashboard visualizes the performance and uncertainty of a pre-trained Bayesian LSTM model for air quality prediction.",
               style={'textAlign': 'center', 'margin-top': '40px', 'color': '#7F8C8D', 'font-size': '0.9em'}),
        style={'padding': '20px'}
    )
])

# ----------------- Run Dash App -----------------
port = int(os.environ.get("PORT", 10000))
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=False)