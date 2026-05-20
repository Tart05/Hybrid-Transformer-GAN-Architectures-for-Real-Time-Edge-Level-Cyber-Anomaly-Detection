import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

def load_and_clean_data(file_path, batch_size=64, is_train=True):
    df = pd.read_csv(file_path)
    
    # 1. Drop non-numeric columns that aren't useful for math (like hex strings)
    # We keep only float and int columns
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    # 2. Handle labels
    if 'target' in df.columns:
        # Convert 'legitimate' to 0 and everything else to 1 (anomaly)
        labels = (df['target'] != 'legitimate').astype(int).values
    else:
        labels = [0] * len(numeric_df)

    # 3. Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)
    
    # 4. Convert to Tensors
    X_tensor = torch.tensor(scaled_data, dtype=torch.float32)
    y_tensor = torch.tensor(labels, dtype=torch.long)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train), numeric_df.shape[1]