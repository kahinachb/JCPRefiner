import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import joblib
import argparse

# ==============================
# CONFIGURATION
# ==============================
SEQ_LEN = 30

# ==============================
# INVERSE SCALER LAYER
# ==============================
class InverseScalerLayer(keras.layers.Layer):
    """Convert from standardized space back to physical space."""
    def __init__(self, scaler_mean, scaler_std, **kwargs):
        super().__init__(**kwargs)
        self.mean = tf.constant(scaler_mean, dtype=tf.float32)
        self.std = tf.constant(scaler_std, dtype=tf.float32)
    
    def call(self, x):
        return x * self.std + self.mean

# ==============================
# INFERENCE FUNCTION
# ==============================
def correct_jcp_from_csv(
    input_csv_path,
    output_csv_path,
    model_dir,
    seq_len=30
):
    """
    Correct JCP data from CSV file using trained corrector model.
    
    Args:
        input_csv_path: Path to input CSV with JCP data (N_frames x J*3 columns)
        output_csv_path: Path to save corrected JCP CSV
        model_dir: Directory containing trained model artifacts
        seq_len: Sequence length (default: 30)
    """
    
    print(f"\n{'='*80}")
    print("JCP CORRECTOR INFERENCE")
    print(f"{'='*80}")
    
    # Load model artifacts
    print(f"\nLoading model from: {model_dir}")
    corrector_path = os.path.join(model_dir, "corrector.keras")
    scaler_path = os.path.join(model_dir, "scaler_hpe.pkl")
    
    if not os.path.exists(corrector_path):
        raise FileNotFoundError(f"Corrector model not found at: {corrector_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found at: {scaler_path}")
    
    corrector = keras.models.load_model(corrector_path)
    scaler = joblib.load(scaler_path)
    
    print(f"  Corrector loaded")
    print(f"  Scaler loaded (mean={scaler.mean_.mean():.4f}, std={scaler.scale_.mean():.4f})")
    
    # Load input data
    print(f"\nLoading input CSV: {input_csv_path}")
    df_input = pd.read_csv(input_csv_path)
    data = df_input.values/1000  # [N_frames, J*3]
    
    N_frames, n_features = data.shape
    N_joints = n_features // 3
    
    print(f"  Input shape: {data.shape}")
    print(f"  Number of frames: {N_frames}")
    print(f"  Number of joints: {N_joints}")
    print(f"  Data range: [{data.min():.3f}, {data.max():.3f}] meters")
    
    # Prepare sliding window buffer
    print(f"\nProcessing with sliding window (seq_len={seq_len})...")
    corrected_frames = []
    keypoints_buffer = []
    
    for i in range(N_frames):
        # Reshape frame to [J, 3]
        frame_data = data[i].reshape(N_joints, 3)
        
        # For first frame, duplicate it seq_len times (like your inference pattern)
        if i == 0:
            keypoints_buffer = [frame_data.copy() for _ in range(seq_len)]
        else:
            keypoints_buffer.append(frame_data.copy())
        
        # Keep only last seq_len frames
        if len(keypoints_buffer) > seq_len:
            keypoints_buffer.pop(0)
        
        # Process when buffer is full
        if len(keypoints_buffer) == seq_len:
            # Stack buffer into sequence [T, J, 3]
            sequence = np.array(keypoints_buffer)  # [30, J, 3]
            
            # Flatten to [T, J*3]
            sequence_flat = sequence.reshape(seq_len, -1)  # [30, J*3]
            
            # Standardize input
            sequence_std = scaler.transform(sequence_flat)  # [30, J*3]
            
            # Add batch dimension [1, T, J*3]
            sequence_batch = sequence_std[np.newaxis, :, :]
            
            # Run corrector
            corrected_std = corrector.predict(sequence_batch, verbose=0)  # [1, T, J*3]
            
            # Inverse transform to physical space
            corrected_flat = scaler.inverse_transform(
                corrected_std[0]  # [T, J*3]
            )
            
            # Extract ONLY the last frame (the current frame prediction)
            corrected_last = corrected_flat[-1, :]  # [J*3]
            
            corrected_frames.append(corrected_last)
        
        # Progress indicator
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{N_frames} frames...")
    
    # Convert to array
    corrected_array = np.array(corrected_frames)  # [N_frames, J*3]
    
    print(f"\nCorrected output shape: {corrected_array.shape}")
    print(f"Corrected range: [{corrected_array.min():.3f}, {corrected_array.max():.3f}] meters")
    
    # Save to CSV
    df_output = pd.DataFrame(
        corrected_array,
        columns=df_input.columns
    )
    df_output.to_csv(output_csv_path, index=False)
    
    print(f"\nCorrected JCP saved to: {output_csv_path}")
    
    # Compute difference statistics
    diff = corrected_array - data
    diff_per_joint = np.linalg.norm(diff.reshape(-1, N_joints, 3), axis=-1)
    mean_correction = diff_per_joint.mean() * 1000  # mm
    max_correction = diff_per_joint.max() * 1000    # mm
    
    print(f"\nCorrection statistics:")
    print(f"  Mean correction per joint: {mean_correction:.2f} mm")
    print(f"  Max correction per joint: {max_correction:.2f} mm")
    print(f"\n{'='*80}")

# ==============================
# MAIN
# ==============================
def main():
    parser = argparse.ArgumentParser(
        description="Correct JCP data using trained corrector model"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input CSV file with JCP data"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save corrected JCP CSV"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory containing trained model (corrector.keras, scaler.pkl)"
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=30,
        help="Sequence length (default: 30)"
    )
    
    args = parser.parse_args()
    
    # Run correction
    correct_jcp_from_csv(
        input_csv_path=args.input,
        output_csv_path=args.output,
        model_dir=args.model_dir,
        seq_len=args.seq_len
    )

if __name__ == "__main__":
    main()