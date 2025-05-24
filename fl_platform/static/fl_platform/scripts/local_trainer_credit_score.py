import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler 
import argparse
import re 

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes) 

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def preprocess_credit_score_data(df_path):
    print(f"Starting preprocessing for: {df_path}")
    try:
        df = pd.read_csv(df_path)
    except Exception as e:
        print(f"Error reading CSV file {df_path}: {e}")
        return None, None, None, None 

    df_clean = df.copy()
    print(f"Original credit score data shape: {df_clean.shape}")

    general_placeholders_to_nan = ['_', 'NA', '!@9#%8', '#F%$D@*&8', '_______', 'Not Specified', 'nan']
    for col in df_clean.select_dtypes(include=['object']).columns:
        for ph in general_placeholders_to_nan:
            df_clean[col] = df_clean[col].astype(str).str.replace(ph, str(np.nan), regex=False)
        df_clean[col] = df_clean[col].replace(r'^\s*$', np.nan, regex=True) 

    cols_to_drop_initial = ['ID', 'Customer_ID', 'Name', 'SSN', 'Type_of_Loan'] 
    df_clean = df_clean.drop(columns=[col for col in cols_to_drop_initial if col in df_clean.columns], errors='ignore')
    print(f"Dropped initial columns. Shape: {df_clean.shape}")

    if 'Age' in df_clean.columns:
        df_clean['Age'] = df_clean['Age'].astype(str).str.extract(r'(\d+)').iloc[:,0]
        df_clean['Age'] = pd.to_numeric(df_clean['Age'], errors='coerce')
        df_clean.loc[df_clean['Age'] <= 0, 'Age'] = np.nan

    def parse_credit_history_age_robust(age_str):
        if pd.isna(age_str): return np.nan
        age_str = str(age_str)
        years, months = 0, 0
        year_match = re.search(r'(\d+)\s*(?:Year|Years)', age_str)
        month_match = re.search(r'(\d+)\s*(?:Month|Months)', age_str)
        if year_match: years = int(year_match.group(1))
        if month_match: months = int(month_match.group(1))
        if not year_match and not month_match: return np.nan
        return (years * 12) + months
    if 'Credit_History_Age' in df_clean.columns:
        df_clean['Credit_History_Age_in_months'] = df_clean['Credit_History_Age'].apply(parse_credit_history_age_robust)
        df_clean = df_clean.drop('Credit_History_Age', axis=1, errors='ignore')

    if 'Payment_of_Min_Amount' in df_clean.columns:
        df_clean["Payment_of_Min_Amount"] = df_clean['Payment_of_Min_Amount'].replace({"NM": "No"})

    if 'Type_of_Loan' in df_clean.columns:
         df_clean['Num_Loan_Types'] = df_clean['Type_of_Loan'].apply(lambda x: len(x.split(',')) if pd.notna(x) and isinstance(x, str) else 0)
         df_clean = df_clean.drop('Type_of_Loan', axis=1, errors='ignore')


    cols_to_force_numeric = [
        'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 
        'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date',
        'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 'Num_Credit_Inquiries',
        'Outstanding_Debt', 'Credit_Utilization_Ratio', 'Total_EMI_per_month',
        'Amount_invested_monthly', 'Monthly_Balance'
    ]
    for col in cols_to_force_numeric:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    target_column_name = 'Credit_Score'
    if target_column_name not in df_clean.columns:
        print(f"Error: Target column '{target_column_name}' not found.")
        return None, None, None, None

    df_clean = df_clean.dropna(subset=[target_column_name]) 
    if df_clean.empty:
        print(f"Error: DataFrame empty after dropping NaNs in target '{target_column_name}'.")
        return None, None, None, None

    le_target = LabelEncoder()
    df_clean['Credit_Score_Encoded'] = le_target.fit_transform(df_clean[target_column_name])
    y = df_clean['Credit_Score_Encoded']
    X = df_clean.drop(columns=[target_column_name, 'Credit_Score_Encoded'], errors='ignore')
    print(f"Target encoded. Classes: {le_target.classes_}")

    print(f"Missing values in X before imputation: {X.isnull().sum().sum()}")
    from pandas.api.types import CategoricalDtype 
    for col in X.columns:
        if X[col].isnull().any():
            if X[col].dtype == 'object' or isinstance(X[col].dtype, CategoricalDtype):
                mode_val = X[col].mode()
                fill_val = mode_val[0] if not mode_val.empty else "Unknown_Imputed"
                X[col] = X[col].fillna(fill_val)
            else: 
                X[col] = X[col].fillna(X[col].median())
    print(f"Missing values in X after imputation: {X.isnull().sum().sum()}")

    categorical_cols_in_X = X.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols_in_X) > 0:
        print(f"One-hot encoding: {list(categorical_cols_in_X)}")
        X = pd.get_dummies(X, columns=categorical_cols_in_X, drop_first=True, dummy_na=False)

    numerical_cols_for_scaling = X.select_dtypes(include=np.number).columns
    numerical_cols_for_scaling = [col for col in numerical_cols_for_scaling if X[col].dtype != 'uint8'] 
    if len(numerical_cols_for_scaling) > 0:
        print(f"Scaling numerical features: {list(numerical_cols_for_scaling)}")
        scaler = StandardScaler()
        X[numerical_cols_for_scaling] = scaler.fit_transform(X[numerical_cols_for_scaling])

    actual_input_dim = X.shape[1]
    print(f"Processed X shape: {X.shape}. Input dimension for model: {actual_input_dim}")


    X_tensor = torch.tensor(X.values.astype(np.float32), dtype=torch.float32)
    y_tensor = torch.tensor(y.values.astype(np.float32), dtype=torch.long) 
    print(f"!!! CRITICAL (Credit Score): Update input_dim for model to: {actual_input_dim} and output_dim to: {len(le_target.classes_)} !!!")

    return X_tensor, y_tensor, actual_input_dim, le_target.classes_

def local_train_multiclass(model, X_data, y_data, local_epochs, learning_rate):
    criterion = nn.CrossEntropyLoss()
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
    print("--- Client Started (Credit Score Dataset) ---")

    X_local, y_local, actual_input_dim, target_classes = preprocess_credit_score_data(args.data_path)

    if X_local is None or y_local is None:
        print("Exiting due to data processing error.")
        exit(1)

    num_classes = len(target_classes)
    print(f"Data loaded. Features: {X_local.shape}, Labels: {y_local.shape}, Num Classes: {num_classes}")

    hidden_size = 64

    print(f"Instantiating SimpleNN with input_dim={actual_input_dim}, hidden_size={hidden_size}, num_classes={num_classes}")
    client_model = SimpleNN(input_size=actual_input_dim, hidden_size=hidden_size, num_classes=num_classes)

    print(f"Loading initial model weights from: {args.model_weights_path}")
    try:
        state_dict = torch.load(args.model_weights_path, map_location=torch.device('cpu'))
        client_model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading model weights: {e}")
        print("Ensure the model architecture in this script matches the saved weights.")
        exit(1)
    print("Initial model weights loaded.")

    updated_weights = local_train_multiclass(client_model, X_local, y_local, args.local_epochs, args.lr)

    if args.noise_scale > 0:
        weights_to_save = add_noise_to_state_dict(updated_weights, args.noise_scale)
        print(f"Noise added with scale: {args.noise_scale}")
    else:
        weights_to_save = updated_weights
        print("No noise added.")

    torch.save(weights_to_save, args.output_path)
    print(f"Updated local model weights saved to: {args.output_path}")
    print("--- Local Client Training Finished ---")