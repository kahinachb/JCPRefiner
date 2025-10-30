import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import json
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ============================================================================
# MODEL DEFINITION (must match training)
# ============================================================================

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
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        
        # Residual connection
        self.residual = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc1(lstm_out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        residual = self.residual(x)
        out = out + residual
        return out

# ============================================================================
# SIMPLE CSV INFERENCE
# ============================================================================

class SimpleCSVInference:
    """Simple inference for CSV files"""
    
    def __init__(self, model_dir='models', metadata_file='processed_data/metadata.json'):
        """
        Initialize inference system
        
        Args:
            model_dir: Directory with model files
            metadata_file: Path to metadata.json from data processing
        """
        
        self.model_dir = Path(model_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load metadata FIRST
        print(f"Loading metadata from: {metadata_file}")
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
        
        # Extract configuration from metadata
        self.n_joints = self.metadata['common']['n_joints']
        self.joint_names = self.metadata['common'].get('joint_names', None)
        self.input_dim = self.n_joints * 3
        self.output_dim = self.n_joints * 3
        
        print(f"Configuration from metadata:")
        print(f"  Number of joints: {self.n_joints}")
        print(f"  Joint names: {self.joint_names[:5]}..." if self.joint_names else "  No joint names found")
        
        # Load training config for model parameters
        self.load_training_config()
        
        # Load model
        self.load_model()
        
        # Load scalers
        self.load_scalers()
        
        print("✓ Inference system ready!\n")
    
    def load_training_config(self):
        """Load training configuration"""
        config_path = self.model_dir / 'training_config.json'
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            self.sequence_length = config.get('sequence_length', 30)
            self.hidden_dim = config.get('hidden_dim', 256)
            self.num_layers = config.get('num_layers', 3)
            self.dropout = config.get('dropout', 0.2)
            self.stride = config.get('stride', 5)  # For overlapping windows
            
            print(f"Training configuration:")
            print(f"  Sequence length: {self.sequence_length}")
            print(f"  Hidden dim: {self.hidden_dim}")
            print(f"  Num layers: {self.num_layers}")
        else:
            # Defaults
            print("Warning: training_config.json not found, using defaults")
            self.sequence_length = 30
            self.hidden_dim = 256
            self.num_layers = 3
            self.dropout = 0.2
            self.stride = 5
    
    def load_model(self):
        """Load the trained LSTM model"""
        model_path = self.model_dir / 'best_joint_corrector.pth'
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        # Initialize model with correct dimensions
        self.model = LSTMJointCorrector(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            output_dim=self.output_dim,
            dropout=0  # No dropout during inference
        ).to(self.device)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        print(f"✓ Model loaded from {model_path}")
    
    def load_scalers(self):
        """Load normalization scalers"""
        scaler_hpe_path = self.model_dir / 'scaler_hpe.pkl'
        scaler_mocap_path = self.model_dir / 'scaler_mocap.pkl'
        
        if not scaler_hpe_path.exists() or not scaler_mocap_path.exists():
            raise FileNotFoundError("Scalers not found in model directory")
        
        self.scaler_hpe = joblib.load(scaler_hpe_path)
        self.scaler_mocap = joblib.load(scaler_mocap_path)
        
        print("✓ Scalers loaded")
    
    def process_csv(self, input_csv, output_csv=None):
        """
        Process a CSV file with HPE data
        
        Args:
            input_csv: Path to input CSV file
            output_csv: Path to save corrected CSV (optional)
        
        Returns:
            corrected_data: Numpy array of corrected positions
        """
        
        print(f"\n{'='*60}")
        print(f"Processing: {input_csv}")
        print(f"{'='*60}")
        
        # Load CSV
        df = pd.read_csv(input_csv)
        print(f"Loaded CSV: {df.shape[0]} frames, {df.shape[1]} columns")
        
        # Get only numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        hpe_data = df[numeric_columns].values
        
        n_frames = len(hpe_data)
        n_coordinates = hpe_data.shape[1]
        
        # Validate dimensions
        expected_coords = self.n_joints * 3
        if n_coordinates != expected_coords:
            print(f"Warning: Expected {expected_coords} coordinates, got {n_coordinates}")
            if n_coordinates > expected_coords:
                print(f"Using first {expected_coords} coordinates")
                hpe_data = hpe_data[:, :expected_coords]
            else:
                raise ValueError(f"Not enough coordinates! Expected {expected_coords}, got {n_coordinates}")
        
        # Reshape to 3D
        hpe_data_3d = hpe_data.reshape(n_frames, self.n_joints, 3)
        print(f"Reshaped to: {hpe_data_3d.shape} (frames, joints, xyz)")
        
        # Process with overlapping windows
        print(f"\nProcessing with sequence length {self.sequence_length}...")
        corrected_flat = self.process_with_overlap(hpe_data)
        
        # Reshape back to 3D
        corrected_3d = corrected_flat.reshape(n_frames, self.n_joints, 3)
        
        # Calculate correction statistics
        corrections = np.sqrt(np.sum((corrected_3d - hpe_data_3d)**2, axis=2))
        mean_correction = np.mean(corrections)
        max_correction = np.max(corrections)
        
        print(f"\nCorrection statistics:")
        print(f"  Mean correction: {mean_correction:.4f} units")
        print(f"  Max correction: {max_correction:.4f} units")
        print(f"  Corrected {n_frames} frames")
        
        # Save if output path provided
        if output_csv:
            self.save_to_csv(corrected_flat, output_csv, df.columns)
            print(f"\n✓ Saved corrected data to: {output_csv}")
        
        return corrected_3d
    
    def process_with_overlap(self, hpe_data_flat):
        """
        Process data with overlapping windows for better accuracy
        
        Args:
            hpe_data_flat: 2D array (n_frames, n_coordinates)
        
        Returns:
            corrected_data: 2D array (n_frames, n_coordinates)
        """
        
        n_frames = len(hpe_data_flat)
        
        # If data is shorter than sequence length, pad it
        if n_frames < self.sequence_length:
            print(f"Data has only {n_frames} frames, padding to {self.sequence_length}")
            padding = self.sequence_length - n_frames
            hpe_data_flat = np.pad(hpe_data_flat, ((0, padding), (0, 0)), mode='edge')
            padded = True
        else:
            padded = False
        
        # Create overlapping sequences
        sequences = []
        sequence_starts = []
        
        stride = min(self.stride, self.sequence_length // 2)  # Adaptive stride
        
        for i in range(0, len(hpe_data_flat) - self.sequence_length + 1, stride):
            sequences.append(hpe_data_flat[i:i + self.sequence_length])
            sequence_starts.append(i)
        
        # Add last sequence if needed
        if len(hpe_data_flat) > self.sequence_length and sequence_starts[-1] + self.sequence_length < len(hpe_data_flat):
            sequences.append(hpe_data_flat[-self.sequence_length:])
            sequence_starts.append(len(hpe_data_flat) - self.sequence_length)
        
        print(f"  Created {len(sequences)} overlapping sequences (stride={stride})")
        
        # Process all sequences
        all_corrections = []
        batch_size = 32
        
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i + batch_size]
            batch_array = np.array(batch)
            
            # Normalize
            batch_norm = self.scaler_hpe.transform(batch_array.reshape(-1, self.input_dim))
            batch_norm = batch_norm.reshape(len(batch), self.sequence_length, self.input_dim)
            
            # Convert to tensor
            batch_tensor = torch.FloatTensor(batch_norm).to(self.device)
            
            # Inference
            with torch.no_grad():
                corrected_norm = self.model(batch_tensor)
            
            # Denormalize
            corrected_norm_np = corrected_norm.cpu().numpy()
            corrected_flat = self.scaler_mocap.inverse_transform(
                corrected_norm_np.reshape(-1, self.output_dim)
            )
            corrected_batch = corrected_flat.reshape(len(batch), self.sequence_length, self.output_dim)
            
            all_corrections.extend(corrected_batch)
        
        # Blend overlapping predictions
        corrected_data = np.zeros((len(hpe_data_flat), self.output_dim))
        frame_counts = np.zeros(len(hpe_data_flat))
        
        for start_idx, correction in zip(sequence_starts, all_corrections):
            end_idx = start_idx + self.sequence_length
            corrected_data[start_idx:end_idx] += correction
            frame_counts[start_idx:end_idx] += 1
        
        # Average overlapping regions
        frame_counts[frame_counts == 0] = 1
        corrected_data = corrected_data / frame_counts[:, np.newaxis]
        
        # Remove padding if applied
        if padded:
            corrected_data = corrected_data[:n_frames]
        
        return corrected_data
    
    def save_to_csv(self, corrected_data, output_path, original_columns):
        """
        Save corrected data to CSV with proper column names
        
        Args:
            corrected_data: 2D array of corrected data
            output_path: Path to save CSV
            original_columns: Original column names from input CSV
        """
        
        # Try to preserve non-numeric columns if any
        non_numeric_cols = [col for col in original_columns 
                           if not any(x in col.lower() for x in ['_x', '_y', '_z', 'x', 'y', 'z'])]
        
        # Create column names for corrected data
        if self.joint_names:
            columns = []
            for joint in self.joint_names:
                columns.extend([f"{joint}_x", f"{joint}_y", f"{joint}_z"])
        else:
            # Try to use original column names if they match
            numeric_cols = [col for col in original_columns 
                          if any(x in col.lower() for x in ['_x', '_y', '_z', 'x', 'y', 'z'])]
            if len(numeric_cols) == corrected_data.shape[1]:
                columns = numeric_cols
            else:
                columns = [f"coord_{i}" for i in range(corrected_data.shape[1])]
        
        # Create dataframe
        df_corrected = pd.DataFrame(corrected_data, columns=columns)
        
        # Add frame numbers if not present
        if 'frame' not in [col.lower() for col in df_corrected.columns]:
            df_corrected.insert(0, 'frame', range(len(df_corrected)))
        
        # Save
        df_corrected.to_csv(output_path, index=False)
        print(f"  Saved with {len(columns)} coordinate columns")
    
    def plot_comparison(self, hpe_data, corrected_data, mocap_data=None, 
                       output_path='joint_comparison.png', max_frames=None):
        """
        Plot comparison between HPE, corrected, and mocap (if available) for each joint
        
        Args:
            hpe_data: Original HPE data (n_frames, n_joints, 3)
            corrected_data: Corrected data (n_frames, n_joints, 3)
            mocap_data: Optional mocap reference data (n_frames, n_joints, 3)
            output_path: Path to save the plot
            max_frames: Maximum frames to plot (None = all frames)
        """
        
        print(f"\nGenerating comparison plots...")
        
        # Determine number of frames to plot
        total_frames = len(hpe_data)
        if max_frames is None:
            n_frames = total_frames
            print(f"  Plotting all {n_frames} frames")
        else:
            n_frames = min(total_frames, max_frames)
            print(f"  Plotting {n_frames} of {total_frames} frames")
        
        hpe_plot = hpe_data[:n_frames]
        corrected_plot = corrected_data[:n_frames]
        if mocap_data is not None:
            mocap_plot = mocap_data[:n_frames]
        
        # Create figure with subplots for each joint
        n_joints = self.n_joints
        n_cols = 4  # Number of columns in subplot grid
        n_rows = (n_joints + n_cols - 1) // n_cols  # Calculate rows needed
        
        fig = plt.figure(figsize=(20, n_rows * 3))
        fig.suptitle('Joint Position Comparison: HPE vs Corrected vs Reference', fontsize=16)
        
        # Create time axis
        time_axis = np.arange(n_frames)
        
        # Plot each joint
        for joint_idx in range(n_joints):
            ax = plt.subplot(n_rows, n_cols, joint_idx + 1)
            
            # Get joint name
            if self.joint_names:
                joint_name = self.joint_names[joint_idx]
            else:
                joint_name = f"Joint {joint_idx}"
            
            # Calculate magnitude (Euclidean norm) for each frame
            hpe_magnitude = np.linalg.norm(hpe_plot[:, joint_idx, :], axis=1)
            corrected_magnitude = np.linalg.norm(corrected_plot[:, joint_idx, :], axis=1)
            
            # Plot HPE and corrected
            ax.plot(time_axis, hpe_magnitude, label='HPE', alpha=0.5, linewidth=1, color='green')
            ax.plot(time_axis, corrected_magnitude, label='Corrected', alpha=1, linewidth=2, color='blue')
            
            # Plot mocap if available
            if mocap_data is not None:
                mocap_magnitude = np.linalg.norm(mocap_plot[:, joint_idx, :], axis=1)
                ax.plot(time_axis, mocap_magnitude, label='Mocap (Ref)', alpha=0.9, 
                       linewidth=1, color='Black', linestyle='--')
                
                # Calculate improvements
                hpe_error = np.mean(np.abs(hpe_magnitude - mocap_magnitude))
                corrected_error = np.mean(np.abs(corrected_magnitude - mocap_magnitude))
                improvement = (1 - corrected_error / hpe_error) * 100 if hpe_error > 0 else 0
                
                # Add improvement text
                ax.text(0.02, 0.98, f'Improve: {improvement:.1f}%', 
                       transform=ax.transAxes, fontsize=8,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax.set_title(joint_name, fontsize=10)
            ax.set_xlabel('Frame', fontsize=8)
            ax.set_ylabel('Position Magnitude', fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)
            
            # Add legend only to first subplot
            if joint_idx == 0:
                ax.legend(fontsize=8, loc='upper right')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved comparison plot to: {output_path}")
        
        # Also create a detailed plot for X, Y, Z separately for first few joints
        self.plot_xyz_comparison(hpe_data, corrected_data, mocap_data, max_frames)
        
        # Show plots (optional)
        plt.show()  # Uncomment if you want to display plots interactively
    
    def plot_xyz_comparison(self, hpe_data, corrected_data, mocap_data=None, 
                           max_frames=None, joints_per_fig=6):
        """
        Plot detailed XYZ comparison for first few joints
        
        Args:
            hpe_data: Original HPE data (n_frames, n_joints, 3)
            corrected_data: Corrected data (n_frames, n_joints, 3)
            mocap_data: Optional mocap reference data (n_frames, n_joints, 3)
            max_frames: Maximum frames to plot (None = all frames)
            n_joints_to_show: Number of joints to show in detail
        """
        
            # Determine number of frames to plot
        total_frames = len(hpe_data)
        if max_frames is None:
            n_frames = total_frames
        else:
            n_frames = min(total_frames, max_frames)
        
        n_joints_plot = self.n_joints
        time_axis = np.arange(n_frames)
        coord_labels = ['X', 'Y', 'Z']
        
        # Split into chunks of joints_per_fig
        for start_joint in range(0, n_joints_plot, joints_per_fig):
            end_joint = min(start_joint + joints_per_fig, n_joints_plot)
            joints_this_fig = range(start_joint, end_joint)
            
            # Create figure for this chunk
            fig, axes = plt.subplots(len(joints_this_fig), 3, 
                                    figsize=(20, len(joints_this_fig) * 2.5))
            axes = np.atleast_2d(axes)  # ensure 2D for consistency
            
            for row_idx, joint_idx in enumerate(joints_this_fig):
                joint_name = self.joint_names[joint_idx] if self.joint_names else f"Joint {joint_idx}"
                
                for coord_idx in range(3):
                    ax = axes[row_idx, coord_idx]
                    
                    # Get coordinate data
                    hpe_coord = hpe_data[:n_frames, joint_idx, coord_idx]
                    corrected_coord = corrected_data[:n_frames, joint_idx, coord_idx]
                    
                    # Plot HPE and corrected
                    ax.plot(time_axis, hpe_coord, label='HPE', alpha=0.4, linewidth=1, color='red')
                    ax.plot(time_axis, corrected_coord, label='Corrected', alpha=0.8, linewidth=1.2, color='green')
                    
                    # Plot mocap if available
                    if mocap_data is not None:
                        mocap_coord = mocap_data[:n_frames, joint_idx, coord_idx]
                        ax.plot(time_axis, mocap_coord, label='Mocap (Ref)', alpha=0.6,
                                linewidth=1, color='black', linestyle='--')
                        
                        # Calculate RMSE
                        hpe_rmse = np.sqrt(np.mean((hpe_coord - mocap_coord)**2))
                        corrected_rmse = np.sqrt(np.mean((corrected_coord - mocap_coord)**2))
                        
                        # Add RMSE text
                        ax.text(0.02, 0.98, f'RMSE: {hpe_rmse:.4f}→{corrected_rmse:.2f}', 
                                transform=ax.transAxes, fontsize=7,
                                verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                    
                    # Set titles/labels
                    ax.set_title(f'{joint_name} - {coord_labels[coord_idx]}', fontsize=9)
                    if coord_idx == 0:
                        ax.set_ylabel('Position', fontsize=8)
                    if row_idx == len(joints_this_fig) - 1:
                        ax.set_xlabel('Frame', fontsize=8)
                    ax.tick_params(labelsize=7)
                    ax.grid(True, alpha=0.3)
                    
                    # Add legend only once
                    if start_joint == 0 and row_idx == 0 and coord_idx == 0:
                        ax.legend(fontsize=7, loc='upper right')
        
        plt.tight_layout()
        
        # Save figure
        output_path = 'joint_xyz_comparison.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved XYZ comparison plot to: {output_path}")
    
    def process_csv_with_reference(self, input_csv, reference_csv=None, output_csv=None):
        """
        Process a CSV file and compare with reference if available
        
        Args:
            input_csv: Path to input HPE CSV file
            reference_csv: Path to reference mocap CSV file (optional)
            output_csv: Path to save corrected CSV (optional)
        
        Returns:
            corrected_data: Numpy array of corrected positions
        """
        
        # Process the HPE data
        corrected_data = self.process_csv(input_csv, output_csv)
        
        # Load original HPE for comparison
        df_hpe = pd.read_csv(input_csv)
        numeric_columns = df_hpe.select_dtypes(include=[np.number]).columns
        hpe_data = df_hpe[numeric_columns].values[:, :self.n_joints*3]
        hpe_data_3d = hpe_data.reshape(-1, self.n_joints, 3)
        
        # Load reference if provided
        mocap_data_3d = None
        if reference_csv:
            print(f"\nLoading reference from: {reference_csv}")
            df_ref = pd.read_csv(reference_csv)
            ref_numeric = df_ref.select_dtypes(include=[np.number]).columns
            mocap_data = df_ref[ref_numeric].values[:, :self.n_joints*3]
            mocap_data_3d = mocap_data.reshape(-1, self.n_joints, 3)
            
            # Ensure same length
            min_len = min(len(hpe_data_3d), len(mocap_data_3d), len(corrected_data))
            hpe_data_3d = hpe_data_3d[:min_len]
            mocap_data_3d = mocap_data_3d[:min_len]
            corrected_data = corrected_data[:min_len]
            
            # Calculate overall statistics
            hpe_errors = np.sqrt(np.sum((hpe_data_3d - mocap_data_3d)**2, axis=2))
            corrected_errors = np.sqrt(np.sum((corrected_data - mocap_data_3d)**2, axis=2))
            
            print(f"\n📊 Performance Metrics:")
            print(f"  HPE Mean Error: {np.mean(hpe_errors):.4f}")
            print(f"  Corrected Mean Error: {np.mean(corrected_errors):.4f}")
            print(f"  Overall Improvement: {(1-np.mean(corrected_errors)/np.mean(hpe_errors))*100:.1f}%")
            
            # Per-joint improvements
            print(f"\n  Per-Joint Improvements:")
            for j in range(self.n_joints):  # Show first 5 joints
                joint_name = self.joint_names[j] if self.joint_names else f"Joint {j}"
                hpe_err = np.mean(hpe_errors[:, j])
                corr_err = np.mean(corrected_errors[:, j])
                impr = (1 - corr_err/hpe_err) * 100 if hpe_err > 0 else 0
                print(f"    {joint_name}: {hpe_err:.3f} → {corr_err:.3f} ({impr:+.1f}%)")
        
        # Generate plots
        self.plot_comparison(hpe_data_3d, corrected_data, mocap_data_3d)
        
        return corrected_data

def main():
    """Main function for command-line usage"""
    
    import argparse
    #trained_2/test_w_2/models2 : best model
    parser = argparse.ArgumentParser(description='Simple CSV Inference for LSTM Joint Correction')
    parser.add_argument('input_csv', type=str, help='Input CSV file with HPE data')
    parser.add_argument('-o', '--output', type=str, help='Output CSV file (default: input_corrected.csv)')
    parser.add_argument('-r', '--reference', type=str, help='Reference mocap CSV for comparison (optional)')
    
    parser.add_argument('-m', '--model-dir', default='trained_2/model_w_offsets', help='Model directory (default: models)')
    parser.add_argument('-d', '--metadata', default='DATA/jcp_npy_w_offsets/metadata.json', 
                       help='Metadata file from data processing')
    

    parser.add_argument('--no-plot', action='store_true', help='Disable plotting')
    parser.add_argument('--max-frames', type=int, default=None, 
                       help='Max frames to plot (default: all frames, -1 for all, or specify number to limit)')
    
    args = parser.parse_args()
    
    # Set default output name if not provided
    if not args.output:
        input_path = Path(args.input_csv)
        args.output = str(input_path.parent / f"{input_path.stem}_corrected.csv")
    
    try:
        # Initialize inference system
        inferencer = SimpleCSVInference(
            model_dir=args.model_dir,
            metadata_file=args.metadata
        )
        
        # Process CSV with optional reference
        if args.reference and not args.no_plot:
            corrected_data = inferencer.process_csv_with_reference(
                args.input_csv, 
                args.reference,
                args.output
            )
        else:
            corrected_data = inferencer.process_csv(args.input_csv, args.output)
            
            # If no reference but plotting requested, still generate plots
            if not args.no_plot:
                # Load HPE data for plotting
                df_hpe = pd.read_csv(args.input_csv)
                numeric_columns = df_hpe.select_dtypes(include=[np.number]).columns
                hpe_data = df_hpe[numeric_columns].values[:, :inferencer.n_joints*3]
                hpe_data_3d = hpe_data.reshape(-1, inferencer.n_joints, 3)
                
                # Handle max_frames = -1 as all frames
                max_frames_to_plot = None if args.max_frames is None or args.max_frames < 0 else args.max_frames
                
                # Generate plots without reference
                inferencer.plot_comparison(hpe_data_3d, corrected_data, 
                                         max_frames=max_frames_to_plot)
        
        print(f"\n{'='*60}")
        print("✅ SUCCESS!")
        print(f"{'='*60}")
        print(f"Input:  {args.input_csv}")
        print(f"Output: {args.output}")
        if args.reference:
            print(f"Reference: {args.reference}")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()