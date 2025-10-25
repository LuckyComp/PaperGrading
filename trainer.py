import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import ast
import sys
from tqdm import tqdm, trange

# ===================================================================
# 1. PaperDataset Class
# ===================================================================
class PaperDataset(Dataset):
    def __init__(self, csv_file):
        try:
            df_full = pd.read_csv(csv_file, dtype=str)
        except FileNotFoundError:
            print(f"Error: The file '{csv_file}' was not found.")
            sys.exit(1)
        print(f"Loaded '{csv_file}'. Original size: {len(df_full)} samples.")
        if df_full.shape[1] < 6:
            print(f"Error: CSV file has {df_full.shape[1]} columns. Expected at least 6.")
            sys.exit(1)

        scalar_cols = df_full.columns[0:2]
        vector_cols = df_full.columns[2:6]
        indices_to_keep = []
        print("Filtering dataset...")
        for idx, row in tqdm(df_full.iterrows(), total=len(df_full), desc="Scanning rows"):
            is_row_good = True
            for col in scalar_cols:
                scalar_string = row[col]
                try:
                    val = float(scalar_string)
                    if val == 0.0 or np.isnan(val) or np.isinf(val): is_row_good = False; break
                except (ValueError, TypeError): is_row_good = False; break
            if not is_row_good: continue
            for col in vector_cols:
                vector_string = row[col]
                if not vector_string or vector_string.lower() == 'nan': is_row_good = False; break
                try:
                    cleaned_string = vector_string.strip('[]'); string_parts = cleaned_string.split()
                    if not string_parts or len(string_parts) != 768: is_row_good = False; break
                    vector_np = np.array([float(part) for part in string_parts])
                    if np.isnan(vector_np).any() or np.isinf(vector_np).any(): is_row_good = False; break
                except Exception: is_row_good = False; break
            if is_row_good: indices_to_keep.append(idx)
        print("Filtering complete.")
        self.df = df_full.loc[indices_to_keep].reset_index(drop=True)
        if len(self.df) == 0: print("Error: Filtering removed all rows!"); sys.exit(1)
        print(f"Filtered dataset size: {len(self.df)} samples.")
        self.scalars = self.df[list(scalar_cols)].values.astype(np.float32)
        self.vector_cols = vector_cols
        self.zero_vector_list = [0.0] * 768

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        scalar_tensor = torch.tensor(self.scalars[idx], dtype=torch.float32)
        vector_list = []
        for col in self.vector_cols:
            vector_string = self.df.loc[idx, col]
            try:
                cleaned_string = vector_string.strip('[]'); string_parts = cleaned_string.split()
                vector_list_py = [float(part) for part in string_parts]
            except Exception: vector_list_py = self.zero_vector_list
            vector_np = np.array(vector_list_py, dtype=np.float32); vector_list.append(vector_np)
        vector_features_np = np.stack(vector_list); vector_features = torch.tensor(vector_features_np, dtype=torch.float32)
        return scalar_tensor, vector_features

# ===================================================================
# 2. HybridAutoencoder Class with Dropout
# ===================================================================
class HybridAutoencoder(nn.Module):
    def __init__(self, scalar_dim=2, vector_dim=768, num_vectors=4, dropout_prob=0.1): # Added dropout_prob
        super(HybridAutoencoder, self).__init__()
        self.num_vectors = num_vectors

        self.scalar_encoder = nn.Sequential(
            nn.Linear(scalar_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout_prob), # Added Dropout
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout_prob), # Added Dropout
            nn.Linear(16, 8)
        )

        self.vector_encoder = nn.Sequential(
            nn.Linear(vector_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_prob), # Added Dropout
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_prob), # Added Dropout
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_prob), # Added Dropout
            nn.Linear(128, 64)
        )

        self.latent_dim_scalar = 8
        self.latent_dim_vector_per_vec = 64
        self.latent_dim_vector_total = num_vectors * self.latent_dim_vector_per_vec

        self.scalar_decoder = nn.Sequential(
            nn.Linear(self.latent_dim_scalar, 16),
            nn.ReLU(),
            nn.Dropout(dropout_prob), # Added Dropout
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Dropout(dropout_prob), # Added Dropout
            nn.Linear(32, scalar_dim)
        )

        self.vector_decoder = nn.Sequential(
            nn.Linear(self.latent_dim_vector_per_vec, 128),
            nn.ReLU(),
            nn.Dropout(dropout_prob), # Added Dropout
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(dropout_prob), # Added Dropout
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(dropout_prob), # Added Dropout
            nn.Linear(512, vector_dim)
        )

    def forward(self, x_scalar, x_vectors):
        s_encoded = self.scalar_encoder(x_scalar)
        v_encoded_list = []
        for i in range(self.num_vectors):
            vec = x_vectors[:, i, :]; v_encoded = self.vector_encoder(vec); v_encoded_list.append(v_encoded)
        v_encoded_combined = torch.cat(v_encoded_list, dim=1)
        latent_vector = torch.cat((s_encoded, v_encoded_combined), dim=1)
        s_latent = latent_vector[:, :self.latent_dim_scalar]; v_latent_combined = latent_vector[:, self.latent_dim_scalar:]
        s_recon = self.scalar_decoder(s_latent)
        v_latent_chunks = torch.chunk(v_latent_combined, self.num_vectors, dim=1)
        v_recon_list = []
        for chunk in v_latent_chunks:
            v_recon = self.vector_decoder(chunk); v_recon_list.append(v_recon)
        v_recon = torch.stack(v_recon_list, dim=1)
        return s_recon, v_recon

# ===================================================================
# 3. Reconstruction Error Function (Standalone)
# ===================================================================
def get_reconstruction_errors(model, criterion, s_data, v_data, device):
    model.eval() # Ensure model is in eval mode for this function
    s_data = s_data.unsqueeze(0).to(device)
    v_data = v_data.unsqueeze(0).to(device)
    with torch.no_grad():
        s_recon, v_recon = model(s_data, v_data)
        loss_s_elements = criterion(s_recon, s_data) # Shape [1, 2]
        loss_v_elements = criterion(v_recon, v_data) # Shape [1, 4, 768]
        loss_s_avg = loss_s_elements.mean()
        loss_v_avg = loss_v_elements.mean()
        overall_avg_error = (loss_s_avg + loss_v_avg).item()
        scalar_errors_detailed = loss_s_elements.squeeze(0).cpu() # Shape [2]
        vector_errors_detailed = loss_v_elements.squeeze(0).cpu() # Shape [4, 768]
    model.train() # Set back to train mode if called during training loop
    return overall_avg_error, scalar_errors_detailed.numpy(), vector_errors_detailed.numpy()

# ===================================================================
# 4. Main Training Block
# ===================================================================
if __name__ == "__main__":

    csv_filename = r'C:\Users\laksh\OneDrive\Desktop\RPP Project\model_training\dataset\new_data.csv'
    VAL_SPLIT = 0.1
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    EPOCHS = 250
    CLIP_MAX_NORM = 1.0
    DROPOUT_PROB = 0.1 # Regularization strength
    MODEL_SAVE_PATH = 'autoencoder_weights.pt'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading data from '{csv_filename}'...")
    full_dataset = PaperDataset(csv_file=csv_filename)

    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    print(f"Split dataset: {len(train_dataset)} train, {len(val_dataset)} validation.")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print("DataLoaders created.")

    model = HybridAutoencoder(dropout_prob=DROPOUT_PROB).to(device)
    criterion = nn.MSELoss() # For average loss during training/validation
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Criterion for detailed error calculation (used optionally or after training)
    # criterion_detailed = nn.MSELoss(reduction='none')

    print("Starting training...")
    epoch_iterator = trange(EPOCHS, desc="Training Epochs")
    best_val_loss = float('inf') # For saving the best model

    for epoch in epoch_iterator:
        model.train()
        train_epoch_loss = 0.0
        train_batch_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} Train", leave=False)

        for s_batch, v_batch in train_batch_iterator:
            s_batch, v_batch = s_batch.to(device), v_batch.to(device)
            s_recon, v_recon = model(s_batch, v_batch)
            loss_scalar = criterion(s_recon, s_batch)
            loss_vector = criterion(v_recon, v_batch)
            total_loss = loss_scalar + loss_vector
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CLIP_MAX_NORM)
            optimizer.step()
            train_epoch_loss += total_loss.item()
            train_batch_iterator.set_postfix(loss=f'{total_loss.item():.6f}')
        avg_train_loss = train_epoch_loss / len(train_loader)

        model.eval()
        val_epoch_loss = 0.0
        val_batch_iterator = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} Val", leave=False)
        with torch.no_grad():
            for s_batch, v_batch in val_batch_iterator:
                s_batch, v_batch = s_batch.to(device), v_batch.to(device)
                s_recon, v_recon = model(s_batch, v_batch)
                loss_scalar = criterion(s_recon, s_batch)
                loss_vector = criterion(v_recon, v_batch)
                total_loss = loss_scalar + loss_vector
                val_epoch_loss += total_loss.item()
                val_batch_iterator.set_postfix(val_loss=f'{total_loss.item():.6f}')
        avg_val_loss = val_epoch_loss / len(val_loader)

        epoch_iterator.set_postfix(avg_train_loss=f'{avg_train_loss:.6f}', avg_val_loss=f'{avg_val_loss:.6f}')

        if np.isnan(avg_train_loss) or np.isnan(avg_val_loss):
            print(f"Epoch [{epoch+1}/{EPOCHS}], NaN Loss detected. TRAINING HALTED.")
            break

        # Save the model if validation loss improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            # print(f"Epoch {epoch+1}: Validation loss improved to {avg_val_loss:.6f}. Model saved.")


    print(f"\nTraining finished. Best validation loss: {best_val_loss:.6f}")
    print(f"Model weights saved to {MODEL_SAVE_PATH}")
