import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import argparse
import io

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.linear(x)


def preprocess_smoking_data(df_path):
    try:
        df = pd.read_csv(df_path)
    except FileNotFoundError:
        print(f"No file exists at given path.")
        return None, None, None
    
    if 'ID' in df.columns:
        df = df.drop('ID', axis=1)
        print("Dropped ID column.")

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.strip()
            df[col] = df[col].replace('?', np.nan)
            df[col] = df[col].replace('', np.nan)

    df_cleaned = df.dropna()

    if df_cleaned.empty:
        print('Error: All entries have NaN entries.')
        return None, None, None
    
    target_column_name = 'smoking'

    if target_column_name not in df_cleaned.columns:
        print(f"Error: target column ('{target_column_name}') not found.")

    X = df_cleaned.drop(target_column_name, axis=1)
    y = df_cleaned[target_column_name]

    categorical_features = X.select_dtypes(include=['object']).columns
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

    X_processed = pd.get_dummies(X, columns=categorical_features, drop_first=True)

    numerical_features_in_processed = [col for col in X_processed.columns if col in numerical_features]

    scaler = StandardScaler()
    X_processed[numerical_features_in_processed] = scaler.fit_transform(X_processed[numerical_features_in_processed])

    print("Numerical features scaled using StandardScaler.\n")
    print(f"Input dimension for model: {X_processed.shape[1]}")
    
    globals()['ADULT_MODEL_INPUT_SIZE'] = X_processed.shape[1]

    X_processed_numeric = X_processed.astype(np.float32)

    X_tensor = torch.tensor(X_processed_numeric.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)

    actual_input_dim = X_tensor.shape[1]

    return X_tensor, y_tensor, actual_input_dim


def local_train(model, X_data, y_data, local_epochs, learning_rate):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    for epoch in range(local_epochs):
        optimizer.zero_grad()
        outputs = model(X_data)
        loss = criterion(outputs, y_data)
        loss.backward()
        optimizer.step()

        print(f"  Local Epoch [{epoch+1}/{local_epochs}], Loss: {loss.item():.4f}")
    return model.state_dict()

def add_noise_to_state_dict(state_dict, noise_scale):
    if not noise_scale or noise_scale <= 0:
        return state_dict

    noisy_state_dict = {}
    for key, param_tensor in state_dict.items():
        if param_tensor.dtype.is_floating_point:
            noise = torch.randn_like(param_tensor) * noise_scale
            noisy_state_dict[key] = param_tensor + noise
        else: 
            noisy_state_dict[key] = param_tensor 
    return noisy_state_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Federated Learning Client")
    parser.add_argument('--model_weights_path', type=str, required=True, help="Path to .pth file for initial weights.")
    parser.add_argument('--data_path', type=str, required=True, help="Path to local CSV.")
    parser.add_argument('--output_path', type=str, required=True, help="Path for saving output.")
    parser.add_argument('--local_epochs', default=5, help="Number of local training epochs.")
    parser.add_argument('--lr', default=0.01, help="Learning rate for local training.")
    parser.add_argument('--noise_scale', type=float, default=0.0, help="Scale of Gaussian noise to add to weights.")
    
    args = parser.parse_args()
    
    print("--- Client Started (Smoking Biosignatures Dataset) ---")
    print(f"Loading data from: {args.data_path}")
    X_local, y_local, actual_input_dim = preprocess_smoking_data(args.data_path)

    if X_local is None or y_local is None:
        print("Exiting due to data processing error.")
        exit(1)
        
    print(f"Data loaded. Features shape: {X_local.shape}, Labels shape: {y_local.shape}")
    print(f"Actual input dimension from processed data: {actual_input_dim}")
    
    client_model = LogisticRegressionModel(input_dim=actual_input_dim, output_dim=1)

    try:
        state_dict = torch.load(args.model_weights_path, map_location=torch.device('cpu'))
        client_model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading model weights: {e}")
        print("Ensure the model architecture in this script matches the saved weights.")
        exit(1)

    print(f"Starting local training for {args.local_epochs} epochs with LR={args.lr}...")
    updated_weights = local_train(client_model, X_local, y_local, args.local_epochs, args.lr)
    print("Local training complete.")

    if args.noise_scale > 0:
        weights_to_save = add_noise_to_state_dict(updated_weights, args.noise_scale)
        print(f"Noise added with scale: {args.noise_scale}")
    else:
        weights_to_save = updated_weights
        print("No noise added.")

    print(f"Saving updated local model weights to: {args.output_path}")
    torch.save(updated_weights, args.output_path)
    print("Updated weights saved successfully.")
    print("--- Local Client Training Finished ---")