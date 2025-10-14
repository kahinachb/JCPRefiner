#split into sequences without mixing subjects. training, test and val set contains each diff subjects
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import joblib
from pathlib import Path

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class JointDataset(Dataset):
    """Dataset for joint position sequences"""
    def __init__(self, hpe_sequences, mocap_sequences):
        self.hpe_sequences = torch.FloatTensor(hpe_sequences)
        self.mocap_sequences = torch.FloatTensor(mocap_sequences)
        
    def __len__(self):
        return len(self.hpe_sequences)
    
    def __getitem__(self, idx):
        return self.hpe_sequences[idx], self.mocap_sequences[idx]

class LSTMJointCorrector(nn.Module):
    """LSTM model for correcting 3D joint positions"""
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super(LSTMJointCorrector, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0,
                           bidirectional=True)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)  # *2 for bidirectional
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        
        # Residual connection
        self.residual = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Apply FC layers
        out = self.fc1(lstm_out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        # Add residual connection (learn the correction)
        residual = self.residual(x)
        out = out + residual
        
        return out

# ============================================================================
# NEW DATA LOADING FUNCTIONS FOR YOUR DATA
# ============================================================================

def load_data(mocap_path='processed_data/mocap_all_subjects.npy',
                   hpe_path='processed_data/hpe_all_subjects.npy',
                   metadata_path='processed_data/metadata.json'):
    """
    Load your processed mocap and HPE data
    
    Args:
        mocap_path: Path to mocap numpy file
        hpe_path: Path to HPE numpy file
        metadata_path: Path to metadata JSON file
    
    Returns:
        mocap_data, hpe_data, metadata
    """
    
    print("Loading data files...")
    
    # Load mocap data
    if not Path(mocap_path).exists():
        raise FileNotFoundError(f"Mocap data not found at {mocap_path}")
    mocap_data = np.load(mocap_path)
    print(f"Loaded mocap data: {mocap_data.shape}")
    
    # Load HPE data
    if not Path(hpe_path).exists():
        print(f"WARNING: HPE data not found at {hpe_path}")
        print("Generating synthetic HPE data for demonstration...")
        # Create synthetic HPE data (mocap + noise) for testing
        hpe_data = mocap_data + np.random.randn(*mocap_data.shape) * 10  # 10mm noise
    else:
        hpe_data = np.load(hpe_path)
        print(f"Loaded HPE data: {hpe_data.shape}")
    
    # Load metadata
    if Path(metadata_path).exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = None
        print("No metadata file found")
    
    # Verify data alignment
    if mocap_data.shape != hpe_data.shape:
        raise ValueError(f"Data shape mismatch! Mocap: {mocap_data.shape}, HPE: {hpe_data.shape}")
    
    return mocap_data, hpe_data, metadata

def create_sequences_per_subject(hpe_data, mocap_data, metadata, sequence_length=30, stride=1):
    """
    Create sliding-window sequences *within each subject block* so no window crosses subjects.

    Returns:
      X: (Nseq, T, J*3)   HPE sequences
      Y: (Nseq, T, J*3)   MoCap sequences
      seq_subjects: (Nseq,) list of subject names for each sequence
      seq_start_indices: (Nseq,) start frame (global index) of each sequence (useful for tracing)
    """
    subj_idx = metadata["common"]["subject_indices"]
    subjects = metadata["common"]["subjects"]
    subjects = [s for s in metadata["common"]["subjects"] if s in metadata["hpe"]["n_frames_per_subject"]]


    all_X, all_Y, seq_subjects, seq_starts = [], [], [], []
    n_frames_total, n_joints, _ = hpe_data.shape
    feat_dim = n_joints * 3

    # flatten once
    hpe_flat   = hpe_data.reshape(n_frames_total, feat_dim)
    mocap_flat = mocap_data.reshape(n_frames_total, feat_dim)

    for s in subjects:
        start, end = subj_idx[s]          # [start, end) global indices for subject s
        n_frames_s = end - start
        # slide only inside [start, end)
        for i in range(start, end - sequence_length + 1, stride):
            # ensure the window stays within the subject
            if i + sequence_length <= end:
                all_X.append(hpe_flat[i:i+sequence_length])
                all_Y.append(mocap_flat[i:i+sequence_length])
                seq_subjects.append(s)
                seq_starts.append(i)

    X = np.stack(all_X, axis=0).astype(np.float32)
    Y = np.stack(all_Y, axis=0).astype(np.float32)
    return X, Y, np.array(seq_subjects), np.array(seq_starts)

import numpy as np
from collections import defaultdict

def split_subjects_fixed_counts(seq_subjects, n_train=13, n_val=2, n_test=2, random_state=42, prefer_subjects=None):
    """
    Leave-Subjects-Out split with fixed subject counts.
    - seq_subjects: (Nseq,) array/list of subject name per sequence
    - prefer_subjects: optional list/tuple of subjects to prioritize (kept only if present)
    Returns:
      idx_train, idx_val, idx_test (numpy int arrays)
      train_subj, val_subj, test_subj (lists of subject names)
    """
    rng = np.random.default_rng(random_state)

    # Subjects actually present in sequences (handles missing 'Zoe' automatically)
    subjects = np.array(sorted(set(seq_subjects)))
    total_needed = n_train + n_val + n_test
    if len(subjects) < total_needed:
        raise ValueError(f"Not enough subjects: have {len(subjects)}, need {total_needed}")

    # Optional: bias toward using certain subjects if provided (but still random)
    if prefer_subjects:
        prefer_subjects = [s for s in prefer_subjects if s in subjects]
        others = [s for s in subjects if s not in prefer_subjects]
        rng.shuffle(others)
        ordered = np.array(prefer_subjects + others, dtype=object)
    else:
        ordered = subjects.copy()
        rng.shuffle(ordered)

    train_subj = list(ordered[:n_train])
    val_subj   = list(ordered[n_train:n_train+n_val])
    test_subj  = list(ordered[n_train+n_val:n_train+n_val+n_test])

    # Build indices per split
    seq_subjects = np.asarray(seq_subjects)
    idx_train = np.where(np.isin(seq_subjects, train_subj))[0]
    idx_val   = np.where(np.isin(seq_subjects, val_subj))[0]
    idx_test  = np.where(np.isin(seq_subjects, test_subj))[0]

    # Shuffle sequences within each split for random mixing
    rng.shuffle(idx_train)
    rng.shuffle(idx_val)
    rng.shuffle(idx_test)

    print("Fixed-count LSO subjects:")
    print("  TRAIN:", sorted(train_subj))
    print("  VAL  :", sorted(val_subj))
    print("  TEST :", sorted(test_subj))
    print(f"#sequences -> train:{len(idx_train)}  val:{len(idx_val)}  test:{len(idx_test)}")

    # Sanity: no overlap
    assert len(set(idx_train) & set(idx_val))  == 0
    assert len(set(idx_train) & set(idx_test)) == 0
    assert len(set(idx_val)  & set(idx_test))  == 0

    return idx_train, idx_val, idx_test, train_subj, val_subj, test_subj



def train_model(model, train_loader, val_loader, num_epochs=100, lr=0.001, save_dir='models'):
    """Train the LSTM model"""
    
    Path(save_dir).mkdir(exist_ok=True)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                     factor=0.5, patience=10, verbose=True)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for hpe_batch, mocap_batch in progress_bar:
            hpe_batch = hpe_batch.to(device)
            mocap_batch = mocap_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(hpe_batch)
            loss = criterion(outputs, mocap_batch)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            progress_bar.set_postfix({'train_loss': loss.item()})
        
        avg_train_loss = train_loss / train_batches
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for hpe_batch, mocap_batch in val_loader:
                hpe_batch = hpe_batch.to(device)
                mocap_batch = mocap_batch.to(device)
                
                outputs = model(hpe_batch)
                loss = criterion(outputs, mocap_batch)
                
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, f'{save_dir}/best_joint_corrector.pth')
            print(f'Best model saved with validation loss: {best_val_loss:.6f}')
    
    return train_losses, val_losses


def evaluate_model(
    model, test_loader, num_joints, scaler_hpe, scaler_mocap, return_baseline=True
):
    """
    Evaluate in physical space (e.g., meters) and report mm.
    Assumes:
      - inputs to model are normalized with scaler_hpe
      - targets are normalized with scaler_mocap
    """
    model.eval()

    # Collect all windows to inverse-transform in one go (avoids per-batch partial stats)
    preds_norm_list   = []
    gts_norm_list     = []
    inputs_norm_list  = []

    with torch.no_grad():
        for hpe_batch, mocap_batch in test_loader:
            hpe_batch   = hpe_batch.to(device)      # normalized by scaler_hpe
            mocap_batch = mocap_batch.to(device)    # normalized by scaler_mocap
            outputs     = model(hpe_batch)          # normalized (mocap space)

            preds_norm_list.append(outputs.cpu().numpy())
            gts_norm_list.append(mocap_batch.cpu().numpy())
            inputs_norm_list.append(hpe_batch.cpu().numpy())

    # Stack: (Nseq, T, J*3)
    preds_norm = np.concatenate(preds_norm_list, axis=0)
    gts_norm   = np.concatenate(gts_norm_list,   axis=0)
    hpe_norm   = np.concatenate(inputs_norm_list,axis=0)

    # Flatten to (Nseq*T, J*3) for inverse transform
    N, T, F = preds_norm.shape
    preds_phys = scaler_mocap.inverse_transform(preds_norm.reshape(-1, F)).reshape(N, T, F)
    gts_phys   = scaler_mocap.inverse_transform(gts_norm.reshape(-1, F)).reshape(N, T, F)

    # If you want to baseline against raw HPE (not the model), inverse transform with scaler_hpe
    if return_baseline:
        hpe_phys = scaler_hpe.inverse_transform(hpe_norm.reshape(-1, F)).reshape(N, T, F)

    # Reshape to (N, T, J, 3)
    J = num_joints
    preds_xyz = preds_phys.reshape(N, T, J, 3)
    gts_xyz   = gts_phys.reshape(N, T, J, 3)
    if return_baseline:
        hpe_xyz = hpe_phys.reshape(N, T, J, 3)

    # MPJPE (in meters), then convert to mm
    per_joint_err_model = np.linalg.norm(preds_xyz - gts_xyz, axis=-1)  # (N, T, J)
    per_joint_mean_mm   = per_joint_err_model.mean(axis=(0,1)) * 1000.0
    overall_mpjpe_mm    = per_joint_err_model.mean() * 1000.0

    results = {
        "per_joint_mpjpe_mm": per_joint_mean_mm,      # (J,)
        "overall_mpjpe_mm": float(overall_mpjpe_mm),  # scalar
        "all_errors_mm": per_joint_err_model * 1000.0 # (N, T, J)
    }

    # Optional: baseline against raw HPE
    if return_baseline:
        per_joint_err_hpe = np.linalg.norm(hpe_xyz - gts_xyz, axis=-1)
        baseline_overall_mpjpe_mm = per_joint_err_hpe.mean() * 1000.0
        improvement_mm = baseline_overall_mpjpe_mm - overall_mpjpe_mm
        improvement_pct = 100.0 * improvement_mm / baseline_overall_mpjpe_mm

        results.update({
            "baseline_overall_mpjpe_mm": float(baseline_overall_mpjpe_mm),
            "improvement_mm": float(improvement_mm),
            "improvement_pct": float(improvement_pct)
        })

    return results

# ============================================================================
# MAIN TRAINING SCRIPT WITH YOUR DATA
# ============================================================================

def main():
    # Configuration
    config = {
        'sequence_length': 30,      # Adjust based on your motion patterns
        'stride': 1,                # Sliding window stride (1 = max overlap)
        'batch_size': 32,
        'num_epochs': 100,
        'learning_rate': 0.001,
        'hidden_dim': 256,
        'num_layers': 3,
        'dropout': 0.2,
        'split_method': 'random',    # 'random' or 'subject' split
        'test_size': 0.15,
        'val_size': 0.15,
    }
    
    # Step 1: Load your data
    print("=" * 50)
    print("Step 1: Loading data")
    print("=" * 50)
    
    mocap_data, hpe_data, metadata = load_data(
        mocap_path='/datasets/jcp_training/processed_data/mocap_all_subjects.npy',
        hpe_path='/datasets/jcp_training/processed_data/hpe_all_subjects.npy',  
        metadata_path='/datasets/jcp_training/processed_data/metadata.json'
    )
    
    # Get dimensions
    n_frames, n_joints, _ = mocap_data.shape
    input_dim = output_dim = n_joints * 3
    
    print(f"\nData info:")
    print(f"  Total frames: {n_frames}")
    print(f"  Number of joints: {n_joints}")
    print(f"  Input/Output dimension: {input_dim}")
    
    # Step 2: Create sequences (boundary-safe)
    print("\n" + "=" * 50)
    print("Step 2: Creating sequences (no cross-subject windows)")
    print("=" * 50)

    X_all, Y_all, seq_subjects, seq_starts = create_sequences_per_subject(
        hpe_data, mocap_data, metadata,
        sequence_length=config['sequence_length'],
        stride=config['stride'],
    )

    idx_train, idx_val, idx_test, train_subj, val_subj, test_subj = split_subjects_fixed_counts(
    seq_subjects,
    n_train=13, n_val=2, n_test=2, random_state=42
    )

    print("Subjects in TRAIN:", train_subj)
    print("Subjects in VAL  :", val_subj)
    print("Subjects in TEST :", test_subj)

    X_train, y_train = X_all[idx_train], Y_all[idx_train]
    X_val,   y_val   = X_all[idx_val],   Y_all[idx_val]
    X_test,  y_test  = X_all[idx_test],  Y_all[idx_test]

    # Fit scalers on TRAIN only, then transform VAL/TEST
    F = X_all.shape[2]
    
    scaler_hpe = StandardScaler()
    scaler_mocap = StandardScaler()

    X_train = scaler_hpe.fit_transform(X_train.reshape(-1, F)).reshape(X_train.shape)
    y_train = scaler_mocap.fit_transform(y_train.reshape(-1, F)).reshape(y_train.shape)

    # Transform val/test with the same scalers
    X_val = scaler_hpe.transform(X_val.reshape(-1, F)).reshape(X_val.shape)
    y_val = scaler_mocap.transform(y_val.reshape(-1, F)).reshape(y_val.shape)

    X_test = scaler_hpe.transform(X_test.reshape(-1, F)).reshape(X_test.shape)
    y_test = scaler_mocap.transform(y_test.reshape(-1, F)).reshape(y_test.shape)
    
    
    print(f"Train set: {X_train.shape[0]} sequences")
    print(f"Validation set: {X_val.shape[0]} sequences")
    print(f"Test set: {X_test.shape[0]} sequences")
    
    train_dataset = JointDataset(X_train, y_train)
    val_dataset = JointDataset(X_val, y_val)
    test_dataset = JointDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Step 6: Initialize model
    print("\n" + "=" * 50)
    print("Step 6: Initializing model")
    print("=" * 50)
    
    model = LSTMJointCorrector(
        input_dim=input_dim,
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        output_dim=output_dim,
        dropout=config['dropout']
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Step 7: Train model
    print("\n" + "=" * 50)
    print("Step 7: Training model")
    print("=" * 50)
    
    train_losses, val_losses = train_model(
        model, train_loader, val_loader,
        num_epochs=config['num_epochs'],
        lr=config['learning_rate']
    )
    
    # Step 8: Load best model and evaluate
    print("\n" + "=" * 50)
    print("Step 8: Evaluating model")
    print("=" * 50)
    
    checkpoint = torch.load('models/best_joint_corrector.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    results = evaluate_model(
        model, test_loader, n_joints, scaler_hpe, scaler_mocap, return_baseline=True
    )    
    print("\nTest Results (physical units):")
    print(f"  Overall MPJPE: {results['overall_mpjpe_mm']:.2f} mm")
    if 'baseline_overall_mpjpe_mm' in results:
        print(f"  Baseline (raw HPE) MPJPE: {results['baseline_overall_mpjpe_mm']:.2f} mm")
        print(f"  Improvement: {results['improvement_mm']:.2f} mm "
            f"({results['improvement_pct']:.1f}%)")

    print("  Per-joint MPJPE (mm):")
    print(results['per_joint_mpjpe_mm'])
    
    # Step 9: Save everything
    print("\n" + "=" * 50)
    print("Step 9: Saving models and scalers")
    print("=" * 50)
    
    # Save scalers
    joblib.dump(scaler_hpe, 'models/scaler_hpe.pkl')
    joblib.dump(scaler_mocap, 'models/scaler_mocap.pkl')
    
    # Save config
    with open('models/training_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Models and scalers saved in 'models/' directory")
    
    # Step 10: Visualize results
    print("\n" + "=" * 50)
    print("Step 10: Creating visualizations")
    print("=" * 50)
    
    # Get joint names if available
    joint_names = metadata.get('joint_names', [f'Joint_{i}' for i in range(n_joints)]) if metadata else [f'Joint_{i}' for i in range(n_joints)]
    
    # plt.figure(figsize=(15, 5))
    
    # # Training history
    # plt.subplot(1, 3, 1)
    # plt.plot(train_losses, label='Train Loss')
    # plt.plot(val_losses, label='Validation Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Training History')
    # plt.legend()
    # plt.grid(True)
    
    # # Per-joint error
    # plt.subplot(1, 3, 2)
    # plt.bar(range(n_joints), mean_joint_errors * 1000)
    # plt.xlabel('Joint Index')
    # plt.ylabel('Mean Error (mm)')
    # plt.title('Per-Joint Error on Test Set')
    # plt.xticks(range(n_joints), joint_names[:n_joints], rotation=45, ha='right')
    # plt.grid(True, axis='y')
    
    # # Error distribution
    # plt.subplot(1, 3, 3)
    # plt.hist(all_joint_errors.flatten() * 1000, bins=50, edgecolor='black')
    # plt.xlabel('Error (mm)')
    # plt.ylabel('Frequency')
    # plt.title('Error Distribution')
    # plt.axvline(np.mean(all_joint_errors) * 1000, color='red', linestyle='--', label=f'Mean: {np.mean(all_joint_errors) * 1000:.2f} mm')
    # plt.legend()
    # plt.grid(True, axis='y')
    
    # plt.tight_layout()
    # plt.savefig('training_results.png', dpi=150)
    # print("Plots saved to 'training_results.png'")
    # plt.show()
    
    print("\n" + "=" * 50)
    print("Training complete!")
    print("=" * 50)
    
    return model, scaler_hpe, scaler_mocap



if __name__ == "__main__":
    model, scaler_hpe, scaler_mocap = main()
