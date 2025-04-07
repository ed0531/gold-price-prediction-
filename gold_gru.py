import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

# --- 1. Scarica dati e normalizza ---
df = yf.download("GC=F", start="2020-01-01", end="2024-12-31", auto_adjust=True)
series = df["Close"].dropna().values

scaler = MinMaxScaler()
series_scaled = scaler.fit_transform(series.reshape(-1, 1)).flatten()

# --- 2. Split train/test ---
split = int(len(series_scaled) * 0.8)
train_data = series_scaled[:split]
test_data = series_scaled[split:]

# --- 3. Crea sequenze ---
def create_sequences(data, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i + window])
        y.append(data[i + window])
    return np.array(X), np.array(y)

window_size = 50
X_train, y_train = create_sequences(train_data, window_size)
X_test, y_test = create_sequences(test_data, window_size)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# --- 4. GRU Model con Dropout ---
class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=3, dropout_rate=0.2):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, dropout=dropout_rate, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        gru_out, _ = self.gru(x)
        out = self.fc(gru_out[:, -1, :])
        return out.squeeze()

model = GRUModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# --- 5. Training con early stopping ---
n_epochs = 1000
batch_size = 64
patience = 20
best_loss = float("inf")
trigger_times = 0

for epoch in range(n_epochs):
    permutation = torch.randperm(X_train_tensor.size(0))
    epoch_loss = 0

    for i in range(0, X_train_tensor.size(0), batch_size):
        indices = permutation[i:i + batch_size]
        X_batch = X_train_tensor[indices]
        y_batch = y_train_tensor[indices]

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = loss_fn(outputs, y_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"[Epoch {epoch + 1}] Train Loss: {epoch_loss:.4f}")

    # --- Early stopping check ---
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        trigger_times = 0
        torch.save(model.state_dict(), "gold_gru_model.pt")  # Salva solo il migliore
    else:
        trigger_times += 1
        print(f"⚠️  No improvement for {trigger_times} epoch(s)")
        if trigger_times >= patience:
            print(f"\n🛑 Early stopping triggered at epoch {epoch + 1}")
            break

# --- 6. Test Evaluation ---
model.load_state_dict(torch.load("gold_gru_model.pt"))
model.eval()
with torch.no_grad():
    preds_test = model(X_test_tensor).numpy()

# --- 7. Denormalizza ---
preds_rescaled = scaler.inverse_transform(preds_test.reshape(-1, 1)).flatten()
true_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

# --- 8. MAE ---
mae = mean_absolute_error(true_rescaled, preds_rescaled)
print(f"\n📊 MAE (Errore medio assoluto sul test set - GRU): {mae:.2f} USD")

#------9.-------RMSE e R² Score------
from sklearn.metrics import mean_squared_error, r2_score

rmse = np.sqrt(mean_squared_error(true_rescaled, preds_rescaled))  # <-- compatibile sempre
r2 = r2_score(true_rescaled, preds_rescaled)
print(f"RMSE: {rmse:.2f}, R² Score: {r2:.4f}")

#------------CSV-------------
pd.DataFrame({
    "Prezzo Reale": true_rescaled,
    "Prezzo Predetto": preds_rescaled
}).to_csv("risultati_gru_oro.csv", index=False)



# --- 9. Plot ---
plt.figure(figsize=(12, 6))
plt.plot(true_rescaled, label="Prezzo Reale (Test)")
plt.plot(preds_rescaled, label="Prezzo Predetto (GRU)")
plt.title("Previsione Oro - GRU con Early Stopping")
plt.xlabel("Step temporali")
plt.ylabel("Prezzo (USD)")
plt.legend()
plt.grid()
plt.show()