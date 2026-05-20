import torch
import numpy as np
import pandas as pd
from models import TransformerGenerator
from data_loader import load_and_clean_data
from sklearn.metrics import classification_report, f1_score

# 1. Setup and Load Data
# We use a batch size of 1 for the demo to simulate real-time packet processing
test_loader, input_dim = load_and_clean_data("data/test30_reduced.csv", batch_size=1, is_train=False)

# 2. Load the Trained Teacher Model
model = TransformerGenerator(input_dim)
model.load_state_dict(torch.load("teacher_baseline.pt"))
model.eval()

print("--- Starting Baseline Demo (Teacher Model) ---")

all_labels = []
all_preds = []
reconstruction_errors = []

# 3. Run Inference
with torch.no_grad():
    for i, (data, label) in enumerate(test_loader):
        # The Generator tries to reconstruct the packet
        reconstructed = model(data)
        
        # Calculate Mean Squared Error (Reconstruction Error)
        loss = torch.mean((data - reconstructed) ** 2).item()
        reconstruction_errors.append(loss)
        all_labels.append(label.item())
        
        # Stop after 1000 samples for the demo printout
        if i >= 1000:
            break

# 4. Determine Anomaly Threshold
# In Chapter 3, anomalies are defined as packets with high reconstruction error.
# We'll set the threshold at the 90th percentile of errors seen in this set.
threshold = np.percentile(reconstruction_errors, 90)

# 5. Classify and Report
for error in reconstruction_errors:
    if error > threshold:
        all_preds.append(1) # Anomaly
    else:
        all_preds.append(0) # Normal

print(f"Calculated Anomaly Threshold: {threshold:.4f}")
print("\nEvaluation Report:")
print(classification_report(all_labels[:1001], all_preds))

# Highlight a few specific results
for i in range(5):
    status = "⚠️ ATTACK" if all_preds[i] == 1 else "✅ NORMAL"
    actual = "⚠️ ATTACK" if all_labels[i] == 1 else "✅ NORMAL"
    print(f"Packet {i} | Prediction: {status} | Actual: {actual} | Error: {reconstruction_errors[i]:.4f}")