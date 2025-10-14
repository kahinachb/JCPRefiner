#split into sequences without mixing subjects. but same subjects will appear in train/val/test, + loss function with sgment length
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
import torch.nn.functional as F

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- (de)standardization helpers for torch ---
def scaler_to_torch_params(scaler, device, F):
    mean  = torch.tensor(scaler.mean_,  dtype=torch.float32, device=device).view(1,1,F)
    scale = torch.tensor(scaler.scale_, dtype=torch.float32, device=device).view(1,1,F)
    return mean, scale

def inv_standardize(x_norm, mean, scale):
    return x_norm * scale + mean

# compute segments length
SEGMENT_SPECS = [
    ("upperarmR", ("RShoulder", "RElbow")),
    ("upperarmL", ("LShoulder", "LElbow")),
    ("lowerarmR", ("RElbow", "RWrist")),
    ("lowerarmL", ("LElbow", "LWrist")),
    ("upperlegR", ("RHip", "RKnee")),
    ("upperlegL", ("LHip", "LKnee")),
    ("lowerlegR", ("RKnee", "RAnkle")),
    ("lowerlegL", ("LKnee", "LAnkle")),
]

def build_joint_index_map(joint_names):
    return {name: i for i, name in enumerate(joint_names)}

def segment_lengths_batched(xyz, jidx, eps=1e-8):
    """
    xyz: (B, T, J, 3) in physical units
    returns: (B, T, E) where E = len(SEGMENT_SPECS)
    """
    lens = []
    for _, (a, b) in SEGMENT_SPECS:
        ia, ib = jidx[a], jidx[b]
        v = xyz[:, :, ib] - xyz[:, :, ia]          # (B,T,3)
        l = torch.sqrt((v**2).sum(dim=-1) + eps)   # (B,T)
        lens.append(l)
    return torch.stack(lens, dim=-1)               # (B,T,E)


def bone_length_loss_first_frame_by_names(
    preds_norm,           # (B,T,F) model outputs (mocap-normalized space)
    inputs_norm,          # (B,T,F) model inputs (hpe-normalized space)
    mean_pred, scale_pred,# (1,1,F) tensors from scaler_mocap
    mean_inp,  scale_inp, # (1,1,F) tensors from scaler_hpe
    joint_names,
    relative=True, eps=1e-8
):
    J = len(joint_names); Fdim = preds_norm.shape[-1]
    assert Fdim == J*3, "Expected features = J*3"

    # de-standardize to physical space
    preds_phys  = inv_standardize(preds_norm,  mean_pred, scale_pred)  # (B,T,F)
    inputs_phys = inv_standardize(inputs_norm, mean_inp,  scale_inp)   # (B,T,F)

    preds_xyz  = preds_phys.view(preds_phys.shape[0], preds_phys.shape[1], J, 3)
    inputs_xyz = inputs_phys.view(inputs_phys.shape[0], inputs_phys.shape[1], J, 3)

    jidx = build_joint_index_map(joint_names)

    L_pred = segment_lengths_batched(preds_xyz, jidx, eps=eps)         # (B,T,E)
    L_ref0 = segment_lengths_batched(inputs_xyz[:, 0:1], jidx, eps=eps)# (B,1,E)
    L_ref  = L_ref0.detach().expand_as(L_pred)                          # (B,T,E)

    if relative:
        # ((L_pred - L_ref)/L_ref)^2 (scale-robust)
        return F.mse_loss((L_pred - L_ref)/(L_ref + eps), torch.zeros_like(L_pred))
    else:
        # (L_pred - L_ref)^2
        return F.mse_loss(L_pred, L_ref)

import torch.nn.functional as F

def bone_length_temporal_smoothness(
    preds_norm,            # (B,T,F) model outputs in mocap-normalized space
    mean_pred, scale_pred, # (1,1,F) tensors from scaler_mocap
    joint_names,
    relative=False,        # set True for % change smoothness
    eps=1e-8
):
    """
    Penalizes changes in bone lengths between consecutive frames.
    If relative=False: minimizes (L[t+1] - L[t])^2
    If relative=True:  minimizes ((L[t+1] - L[t]) / L[t])^2
    """
    J = len(joint_names); Fdim = preds_norm.shape[-1]
    assert Fdim == J*3, "Expected features = J*3"

    # de-standardize predictions to physical units
    preds_phys = inv_standardize(preds_norm, mean_pred, scale_pred)     # (B,T,F)
    preds_xyz  = preds_phys.view(preds_phys.shape[0], preds_phys.shape[1], J, 3)

    jidx = build_joint_index_map(joint_names)

    # (B,T,E) bone lengths for your segments (E = len(SEGMENT_SPECS))
    L_pred = segment_lengths_batched(preds_xyz, jidx, eps=eps)

    # consecutive differences along time: (B,T-1,E)
    dL = L_pred[:, 1:, :] - L_pred[:, :-1, :]

    if relative:
        # % change per step — scale-robust
        base = L_pred[:, :-1, :]
        return F.mse_loss(dL / (base + eps), torch.zeros_like(dL))
    else:
        # absolute change per step
        return F.mse_loss(dL, torch.zeros_like(dL))



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

def train_model(
    model, train_loader, val_loader, num_epochs=100, lr=0.001, save_dir='models',
    # first-frame constraint::
    scaler_hpe=None, scaler_mocap=None, joint_names=None,
    lambda_bone=0.1, relative_bone=True,
    # for temporal length smoothness:
    lambda_temp=0.05, relative_temp=False
):
    Path(save_dir).mkdir(exist_ok=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    # prepare de-standardization tensors once
    use_bone = (lambda_bone and scaler_hpe and scaler_mocap and joint_names)
    use_temp = (lambda_temp and scaler_mocap and joint_names)

    if use_bone or use_temp:
        F = len(joint_names)*3
        mean_inp,  scale_inp  = scaler_to_torch_params(scaler_hpe,   device, F)
        mean_pred, scale_pred = scaler_to_torch_params(scaler_mocap, device, F)

    train_losses, val_losses, best_val_loss = [], [], float('inf')

    for epoch in range(num_epochs):
        # ---- TRAIN ----
        model.train(); train_loss=0.0; train_batches=0
        for hpe_batch, mocap_batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            hpe_batch   = hpe_batch.to(device)
            mocap_batch = mocap_batch.to(device)

            optimizer.zero_grad()
            outputs = model(hpe_batch)

            mse = criterion(outputs, mocap_batch)
            loss = mse
            if use_bone:
                bl = bone_length_loss_first_frame_by_names(
                    outputs, hpe_batch,
                    mean_pred, scale_pred, mean_inp, scale_inp,
                    joint_names, relative=relative_bone
                )
                loss = mse + lambda_bone*bl
            
            if use_temp:
                tls = bone_length_temporal_smoothness(
                    outputs, mean_pred, scale_pred,
                    joint_names, relative=relative_temp
                )
                loss = loss + lambda_temp * tls

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item(); train_batches += 1

        avg_train_loss = train_loss / max(train_batches,1)
        train_losses.append(avg_train_loss)

        # ---- VAL ----
        model.eval(); val_loss=0.0; val_batches=0
        with torch.no_grad():
            for hpe_batch, mocap_batch in val_loader:
                hpe_batch   = hpe_batch.to(device)
                mocap_batch = mocap_batch.to(device)
                outputs     = model(hpe_batch)

                mse = criterion(outputs, mocap_batch)
                if use_bone:
                    bl = bone_length_loss_first_frame_by_names(
                        outputs, hpe_batch,
                        mean_pred, scale_pred, mean_inp, scale_inp,
                        joint_names, relative=relative_bone
                    )
                    loss = mse + lambda_bone*bl
                else:
                    loss = mse

                val_loss += loss.item(); val_batches += 1

        avg_val_loss = val_loss / max(val_batches,1)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}]  Train: {avg_train_loss:.6f}  Val: {avg_val_loss:.6f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, f'{save_dir}/best_joint_corrector.pth')
            print(f'✓ Saved best (val {best_val_loss:.6f})')

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
        'stride': 5,                # Sliding window stride (1 = max overlap)
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

    n_sequences, T, F = X_all.shape
    print(f"Created {n_sequences} sequences of length {T} (features {F}) from {len(np.unique(seq_subjects))} subjects")

    
    # Step 3: Normalize data
    print("\n" + "=" * 50)
    print("Step 3: Splitting data (random, but boundary-safe) and normalizing")
    print("=" * 50)

    # Random split at sequence level (still intra-subject windows)
    X_train, X_temp, Y_train, Y_temp = train_test_split(
        X_all, Y_all,
        test_size=config['test_size'] + config['val_size'],
        random_state=42, shuffle=True,
    )

    val_size_adjusted = config['val_size'] / (config['test_size'] + config['val_size'])
    X_val, X_test, Y_val, Y_test = train_test_split(
        X_temp, Y_temp,
        test_size=1 - val_size_adjusted,
        random_state=42, shuffle=True,
    )
    
    scaler_hpe = StandardScaler()
    scaler_mocap = StandardScaler()

    X_train = scaler_hpe.fit_transform(X_train.reshape(-1, F)).reshape(X_train.shape)
    y_train = scaler_mocap.fit_transform(Y_train.reshape(-1, F)).reshape(Y_train.shape)

    # Transform val/test with the same scalers
    X_val = scaler_hpe.transform(X_val.reshape(-1, F)).reshape(X_val.shape)
    y_val = scaler_mocap.transform(Y_val.reshape(-1, F)).reshape(Y_val.shape)

    X_test = scaler_hpe.transform(X_test.reshape(-1, F)).reshape(X_test.shape)
    y_test = scaler_mocap.transform(Y_test.reshape(-1, F)).reshape(Y_test.shape)
    
    
    print(f"Train set: {X_train.shape[0]} sequences")
    print(f"Validation set: {X_val.shape[0]} sequences")
    print(f"Test set: {X_test.shape[0]} sequences")

    indices = np.arange(X_all.shape[0])
    idx_train, idx_temp = train_test_split(indices, test_size=config['test_size'] + config['val_size'], random_state=42, shuffle=True)
    idx_val, idx_test = train_test_split(idx_temp, test_size=1 - val_size_adjusted, random_state=42, shuffle=True)

    # Use these idx_* to slice X_all/Y_all instead of reusing the earlier split if you like.
    print("Subjects in TRAIN:", sorted(set(seq_subjects[idx_train])))
    print("Subjects in VAL  :", sorted(set(seq_subjects[idx_val])))
    print("Subjects in TEST :", sorted(set(seq_subjects[idx_test])))
    
    # Step 5: Create data loaders
    print("\n" + "=" * 50)
    print("Step 5: Creating data loaders")
    print("=" * 50)
    
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
    joint_names = metadata['common']['joint_names'] 

    train_losses, val_losses = train_model(
    model, train_loader, val_loader,
    num_epochs=config['num_epochs'],
    lr=config['learning_rate'],
    scaler_hpe=scaler_hpe,
    scaler_mocap=scaler_mocap,
    joint_names=metadata['common']['joint_names'],
    lambda_bone=0.0,        # first-frame constancy
    relative_bone=True,
    lambda_temp=0.05,       # temporal smoothness (tune 0.02–0.1)
    relative_temp=False     # or True for % change per step
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
