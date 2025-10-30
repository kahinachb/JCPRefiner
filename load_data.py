import os
import numpy as np
import pandas as pd
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

class UnifiedDataProcessor:
    """
    Unified processor for both Mocap and HPE data
    Ensures perfect alignment between both datasets
    """
    
    def __init__(self, mocap_folder='mocap_jcp', hpe_folder='hpe_jcp',
                 mocap_csv='robot_welding_joint_center_positions.csv',
                 hpe_csv='hpe_joint_positions.csv',
                 output_dir='processed_data'):
        """
        Initialize the processor with folder and file information
        
        Args:
            mocap_folder: Base folder for mocap data
            hpe_folder: Base folder for HPE data
            mocap_csv: CSV filename in each mocap subject folder
            hpe_csv: CSV filename in each HPE subject folder
            output_dir: Directory to save processed data
        """
        self.mocap_folder = Path(mocap_folder)
        self.hpe_folder = Path(hpe_folder)
        self.mocap_csv = mocap_csv
        self.hpe_csv = hpe_csv
        self.output_dir = Path(output_dir)
        
        # Data storage
        self.mocap_data = {}
        self.hpe_data = {}
        self.metadata = {
            'mocap': {},
            'hpe': {},
            'common': {
                'subjects': [],
                'n_joints': None,
                'joint_names': None,
                'total_frames': 0,
                'processing_date': datetime.now().isoformat()
            }
        }
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
    def process_all_data(self):
        """Main processing pipeline"""
        print("=" * 60)
        print("UNIFIED MOCAP AND HPE DATA PROCESSOR")
        print("=" * 60)
        
        # Step 1: Discover subjects
        print("\n[Step 1/7] Discovering subjects...")
        self.discover_subjects()
        
        # Step 2: Load mocap data
        print("\n[Step 2/7] Loading mocap data...")
        self.load_mocap_data()
        
        # Step 3: Load HPE data
        print("\n[Step 3/7] Loading HPE data...")
        self.load_hpe_data()
        
        # Step 4: Verify alignment
        print("\n[Step 4/7] Verifying data alignment...")
        if not self.verify_alignment():
            return False
        
        # Step 5: Validate data
        print("\n[Step 5/7] Validating data quality...")
        self.validate_all_data()
        
        # Step 6: Save processed data
        print("\n[Step 6/7] Saving processed data...")
        self.save_all_data()
        
        # Step 7: Generate report
        print("\n[Step 7/7] Generating analysis report...")
        self.generate_report()
        
        print("\n" + "=" * 60)
        print("✓ PROCESSING COMPLETE!")
        print("=" * 60)
        
        return True
    
    def discover_subjects(self):
        """Discover all subjects in both mocap and HPE folders"""
        
        # Check if folders exist
        if not self.mocap_folder.exists():
            raise FileNotFoundError(f"Mocap folder not found: {self.mocap_folder}")
        if not self.hpe_folder.exists():
            raise FileNotFoundError(f"HPE folder not found: {self.hpe_folder}")
        
        # Get subject folders
        mocap_subjects = {f.name for f in self.mocap_folder.iterdir() if f.is_dir()}
        hpe_subjects = {f.name for f in self.hpe_folder.iterdir() if f.is_dir()}
        
        # Find common subjects
        common_subjects = mocap_subjects & hpe_subjects
        only_mocap = mocap_subjects - hpe_subjects
        only_hpe = hpe_subjects - mocap_subjects
        
        print(f"  Found {len(mocap_subjects)} mocap subjects")
        print(f"  Found {len(hpe_subjects)} HPE subjects")
        print(f"  Common subjects: {len(common_subjects)}")
        
        if only_mocap:
            print(f"  ⚠ Only in mocap: {only_mocap}")
        if only_hpe:
            print(f"  ⚠ Only in HPE: {only_hpe}")
        
        if not common_subjects:
            raise ValueError("No common subjects found between mocap and HPE!")
        
        # Store sorted list of common subjects
        self.metadata['common']['subjects'] = sorted(list(common_subjects))
        self.metadata['mocap']['all_subjects'] = sorted(list(mocap_subjects))
        self.metadata['hpe']['all_subjects'] = sorted(list(hpe_subjects))
        
        print(f"  ✓ Will process {len(common_subjects)} common subjects")
    
    def load_mocap_data(self):
        """Load all mocap data"""
        
        subjects = self.metadata['common']['subjects']
        self.metadata['mocap']['n_frames_per_subject'] = {}
        
        for subject in tqdm(subjects, desc="  Loading mocap"):
            csv_path = self.mocap_folder / subject / "robot_welding" /self.mocap_csv
            print(csv_path)
            
            if not csv_path.exists():
                print(f"\n  ⚠ Mocap CSV not found for {subject}")
                continue
            
            try:
                # Load and process CSV
                df = pd.read_csv(csv_path)
                data_array = self.process_csv_to_3d(df, 'mocap', subject)
                
                # Store data
                self.mocap_data[subject] = data_array
                self.metadata['mocap']['n_frames_per_subject'][subject] = len(data_array)
                
                # Extract metadata from first subject
                if self.metadata['common']['n_joints'] is None:
                    self.metadata['common']['n_joints'] = data_array.shape[1]
                    self.metadata['common']['joint_names'] = self.extract_joint_names(df.columns)
                    print(f"\n  Detected {data_array.shape[1]} joints")
                    
            except Exception as e:
                print(f"\n  ❌ Error loading mocap for {subject}: {e}")
                continue
        
        print(f"  ✓ Loaded mocap data for {len(self.mocap_data)} subjects")
    
    def load_hpe_data(self):
        """Load all HPE data"""

        # Columns you want to keep
        selected_cols = [
            "RShoulder_x","RShoulder_y","RShoulder_z",
            "LShoulder_x","LShoulder_y","LShoulder_z",
            "Neck_x","Neck_y","Neck_z",
            "RElbow_x","RElbow_y","RElbow_z",
            "LElbow_x","LElbow_y","LElbow_z",
            "RWrist_x","RWrist_y","RWrist_z",
            "LWrist_x","LWrist_y","LWrist_z",
            "RHip_x","RHip_y","RHip_z",
            "LHip_x","LHip_y","LHip_z",
            "midHip_x","midHip_y","midHip_z",
            "RKnee_x","RKnee_y","RKnee_z",
            "LKnee_x","LKnee_y","LKnee_z",
            "RAnkle_x","RAnkle_y","RAnkle_z",
            "LAnkle_x","LAnkle_y","LAnkle_z",
            "RHeel_x","RHeel_y","RHeel_z",
            "LHeel_x","LHeel_y","LHeel_z",
            "RBigToe_x","RBigToe_y","RBigToe_z",
            "LBigToe_x","LBigToe_y","LBigToe_z",
            "RSmallToe_x","RSmallToe_y","RSmallToe_z",
            "LSmallToe_x","LSmallToe_y","LSmallToe_z",
            "FHD_x", "FHD_y","FHD_z",
            "LHD_x", "LHD_y","LHD_z",
            "RHD_x", "RHD_y","RHD_z"

        ]

        subjects = self.metadata['common']['subjects']
        self.metadata['hpe']['n_frames_per_subject'] = {}

        for subject in tqdm(subjects, desc="  Loading HPE"):
            csv_path = self.hpe_folder / subject / self.hpe_csv

            if not csv_path.exists():
                print(f"\n  ⚠ HPE CSV not found for {subject}")
                continue

            try:
                # Load CSV and keep only desired columns
                df = pd.read_csv(csv_path)

                # Filter columns (keep only those that exist in the file)
                df = df[[col for col in selected_cols if col in df.columns]]

                data_array = self.process_csv_to_3d(df, 'hpe', subject)

                # Store data
                self.hpe_data[subject] = data_array
                self.metadata['hpe']['n_frames_per_subject'][subject] = len(data_array)

            except Exception as e:
                print(f"\n  ❌ Error loading HPE for {subject}: {e}")
                continue

        print(f"  ✓ Loaded HPE data for {len(self.hpe_data)} subjects")
    
    def process_csv_to_3d(self, df, data_type, subject):
        """
        Process CSV DataFrame to 3D array (n_frames, n_joints, 3)
        
        Args:
            df: pandas DataFrame with joint data
            data_type: 'mocap' or 'hpe' for debugging
            subject: subject name for debugging
        
        Returns:
            numpy array of shape (n_frames, n_joints, 3)
        """
        
        # Get only numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        data = df[numeric_columns].values
        
        n_frames = len(data)
        n_coordinates = data.shape[1]
        
        # Check if divisible by 3
        if n_coordinates % 3 != 0:
            print(f"\n  ⚠ Warning: {data_type} {subject} has {n_coordinates} coordinates (not divisible by 3)")
        
        n_joints = n_coordinates // 3
        
        # Reshape to 3D
        try:
            data_3d = data[:, :n_joints*3].reshape(n_frames, n_joints, 3)
            return data_3d
        except Exception as e:
            print(f"\n  ❌ Could not reshape {data_type} {subject}: {e}")
            raise
    
    def extract_joint_names(self, columns):
        """Extract joint names from column names"""
        
        joint_names = []
        for col in columns:
            # Skip non-coordinate columns
            if any(skip in col.lower() for skip in ['frame', 'time', 'index']):
                continue
            
            # Extract joint name from patterns like 'joint1_x' or 'Head_x'
            if '_x' in col.lower():
                joint_name = col.rsplit('_x', 1)[0]
                joint_names.append(joint_name)
            elif col.endswith('_X') or col.endswith('.x'):
                joint_name = col[:-2]
                if joint_name not in joint_names:
                    joint_names.append(joint_name)
        
        return joint_names if joint_names else None
    
    def verify_alignment(self):
        """Verify that mocap and HPE data are aligned"""
        
        print("\n  Checking data alignment...")
        
        # Get subjects that have both mocap and HPE data
        valid_subjects = list(set(self.mocap_data.keys()) & set(self.hpe_data.keys()))
        
        if not valid_subjects:
            print("  ❌ No subjects have both mocap and HPE data!")
            return False
        
        # Check each subject
        misaligned_subjects = []
        frame_differences = {}
        
        for subject in valid_subjects:
            mocap_shape = self.mocap_data[subject].shape
            hpe_shape = self.hpe_data[subject].shape
            
            # Check frame count
            if mocap_shape[0] != hpe_shape[0]:
                frame_diff = abs(mocap_shape[0] - hpe_shape[0])
                frame_differences[subject] = (mocap_shape[0], hpe_shape[0], frame_diff)
                misaligned_subjects.append(subject)
                print(f"  ⚠ {subject}: Frame mismatch - Mocap: {mocap_shape[0]}, HPE: {hpe_shape[0]} (diff: {frame_diff})")
            
            # Check joint count
            if mocap_shape[1] != hpe_shape[1]:
                print(f"  ❌ {subject}: Joint count mismatch - Mocap: {mocap_shape[1]}, HPE: {hpe_shape[1]}")
                return False
        
        # Handle frame mismatches
        if misaligned_subjects:
            print(f"\n  Found {len(misaligned_subjects)} subjects with frame mismatches")
            print("  Options:")
            print("  1. Trim to minimum frame count (recommended)")
            print("  2. Skip misaligned subjects")
            print("  3. Stop processing")
            
            choice = input("  Choose option (1/2/3): ").strip()
            
            if choice == '1':
                self.trim_to_minimum_frames(misaligned_subjects)
            elif choice == '2':
                for subject in misaligned_subjects:
                    del self.mocap_data[subject]
                    del self.hpe_data[subject]
                print(f"  Removed {len(misaligned_subjects)} misaligned subjects")
            else:
                return False
        
        # Update valid subjects list
        self.metadata['common']['valid_subjects'] = list(set(self.mocap_data.keys()) & set(self.hpe_data.keys()))
        
        print(f"  ✓ Data aligned for {len(self.metadata['common']['valid_subjects'])} subjects")
        return True
    
    def trim_to_minimum_frames(self, subjects):
        """Trim data to minimum frame count for given subjects"""
        
        for subject in subjects:
            mocap_frames = self.mocap_data[subject].shape[0]
            hpe_frames = self.hpe_data[subject].shape[0]
            min_frames = min(mocap_frames, hpe_frames)
            
            # Trim both to minimum
            self.mocap_data[subject] = self.mocap_data[subject][:min_frames]
            self.hpe_data[subject] = self.hpe_data[subject][:min_frames]
            
            print(f"  Trimmed {subject} to {min_frames} frames")
    
    def validate_all_data(self):
        """Validate data quality for all subjects"""
        
        print("\n  Validating data quality...")
        
        issues_found = False
        
        for subject in self.metadata['common']['valid_subjects']:
            # Check mocap
            mocap_nan = np.sum(np.isnan(self.mocap_data[subject]))
            mocap_inf = np.sum(np.isinf(self.mocap_data[subject]))
            
            # Check HPE
            hpe_nan = np.sum(np.isnan(self.hpe_data[subject]))
            hpe_inf = np.sum(np.isinf(self.hpe_data[subject]))
            
            if mocap_nan > 0 or mocap_inf > 0 or hpe_nan > 0 or hpe_inf > 0:
                print(f"  ⚠ {subject}: Mocap (NaN:{mocap_nan}, Inf:{mocap_inf}), HPE (NaN:{hpe_nan}, Inf:{hpe_inf})")
                issues_found = True
        
        if not issues_found:
            print("  ✓ All data passed quality checks")
        else:
            response = input("\n  Data quality issues found. Continue anyway? (y/n): ")
            if response.lower() != 'y':
                return False
        
        return True
    
    def save_all_data(self):
        """Save all processed data"""
        
        valid_subjects = self.metadata['common']['valid_subjects']
        
        # Combine data in subject order
        mocap_list = []
        hpe_list = []
        subject_indices = {}
        current_idx = 0
        
        for subject in sorted(valid_subjects):
            # Add to combined lists
            mocap_list.append(self.mocap_data[subject])
            hpe_list.append(self.hpe_data[subject])
            
            # Track indices
            n_frames = len(self.mocap_data[subject])
            subject_indices[subject] = (current_idx, current_idx + n_frames)
            current_idx += n_frames
            
            # Save individual subject files
            np.save(self.output_dir / f'mocap_{subject}.npy', self.mocap_data[subject])
            np.save(self.output_dir / f'hpe_{subject}.npy', self.hpe_data[subject])
        
        # Save combined data
        mocap_combined = np.concatenate(mocap_list, axis=0)
        hpe_combined = np.concatenate(hpe_list, axis=0)
        
        np.save(self.output_dir / 'mocap_all_subjects.npy', mocap_combined)
        np.save(self.output_dir / 'hpe_all_subjects.npy', hpe_combined)
        
        print(f"  ✓ Saved combined mocap data: {mocap_combined.shape}")
        print(f"  ✓ Saved combined HPE data: {hpe_combined.shape}")
        
        # Update and save metadata
        self.metadata['common']['subject_indices'] = subject_indices
        self.metadata['common']['total_frames'] = current_idx
        self.metadata['common']['combined_shape'] = list(mocap_combined.shape)
        
        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        print(f"  ✓ Saved metadata to metadata.json")
    
    def generate_report(self):
        """Generate analysis report and visualizations"""
        
        # Load combined data
        mocap_data = np.load(self.output_dir / 'mocap_all_subjects.npy')
        hpe_data = np.load(self.output_dir / 'hpe_all_subjects.npy')
        
        # Calculate statistics
        errors = np.sqrt(np.sum((hpe_data - mocap_data)**2, axis=2))  # Euclidean distance
        mean_error_per_joint = np.mean(errors, axis=0)
        
        # Create visualizations
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Mocap vs HPE Data Analysis', fontsize=16)
        
        # 1. Per-joint error
        ax = axes[0, 0]
        joint_names = self.metadata['common']['joint_names']
        n_joints = len(mean_error_per_joint)
        x_pos = np.arange(n_joints)
        ax.bar(x_pos, mean_error_per_joint)
        ax.set_xlabel('Joint Index')
        ax.set_ylabel('Mean Error (units)')
        ax.set_title('Mean Error per Joint')
        if joint_names and len(joint_names) == n_joints:
            ax.set_xticks(x_pos[::2])
            ax.set_xticklabels(joint_names[::2], rotation=45, ha='right')
        
        # 2. Error distribution
        ax = axes[0, 1]
        ax.hist(errors.flatten(), bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(errors), color='red', linestyle='--', label=f'Mean: {np.mean(errors):.3f}')
        ax.set_xlabel('Error (units)')
        ax.set_ylabel('Frequency')
        ax.set_title('Error Distribution')
        ax.legend()
        
        # 3. Error over time (sample)
        ax = axes[0, 2]
        sample_frames = min(1000, len(errors))
        ax.plot(np.mean(errors[:sample_frames], axis=1), alpha=0.7)
        ax.set_xlabel('Frame')
        ax.set_ylabel('Mean Error')
        ax.set_title(f'Mean Error over Time (first {sample_frames} frames)')
        
        # 4. Coordinate comparison (first joint, first 100 frames)
        ax = axes[1, 0]
        frames_to_plot = min(100, len(mocap_data))
        ax.plot(mocap_data[:frames_to_plot, 0, 0], label='Mocap X', alpha=0.7)
        ax.plot(hpe_data[:frames_to_plot, 0, 0], label='HPE X', alpha=0.7)
        ax.set_xlabel('Frame')
        ax.set_ylabel('Position')
        ax.set_title('Joint 0, X-coordinate comparison')
        ax.legend()
        
        # 5. Subject-wise statistics
        ax = axes[1, 1]
        subject_errors = []
        subject_names = []
        for subject, (start, end) in self.metadata['common']['subject_indices'].items():
            subject_error = np.mean(errors[start:end])
            subject_errors.append(subject_error)
            subject_names.append(subject)
        
        ax.bar(range(len(subject_errors)), subject_errors)
        ax.set_xlabel('Subject')
        ax.set_ylabel('Mean Error')
        ax.set_title('Mean Error per Subject')
        ax.set_xticks(range(0, len(subject_names), max(1, len(subject_names)//10)))
        ax.set_xticklabels(subject_names[::max(1, len(subject_names)//10)], rotation=45, ha='right')
        
        # 6. Summary statistics
        ax = axes[1, 2]
        ax.axis('off')
        stats_text = f"""
        Dataset Statistics:
        
        Total Subjects: {len(self.metadata['common']['valid_subjects'])}
        Total Frames: {self.metadata['common']['total_frames']}
        Joints per Frame: {self.metadata['common']['n_joints']}
        
        Error Statistics:
        Mean Error: {np.mean(errors):.4f}
        Std Error: {np.std(errors):.4f}
        Min Error: {np.min(errors):.4f}
        Max Error: {np.max(errors):.4f}
        
        Data Ranges:
        Mocap: [{np.min(mocap_data):.2f}, {np.max(mocap_data):.2f}]
        HPE: [{np.min(hpe_data):.2f}, {np.max(hpe_data):.2f}]
        """
        ax.text(0.1, 0.5, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='center', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'data_analysis_report.png', dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved analysis report to data_analysis_report.png")
        
        # Save text report
        with open(self.output_dir / 'data_report.txt', 'w') as f:
            f.write("MOCAP AND HPE DATA PROCESSING REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Processing Date: {self.metadata['common']['processing_date']}\n\n")
            f.write("SUMMARY:\n")
            f.write(f"  Processed Subjects: {len(self.metadata['common']['valid_subjects'])}\n")
            f.write(f"  Total Frames: {self.metadata['common']['total_frames']}\n")
            f.write(f"  Joints per Frame: {self.metadata['common']['n_joints']}\n")
            f.write(f"  Data Shape: {self.metadata['common']['combined_shape']}\n\n")
            f.write("ERROR STATISTICS:\n")
            f.write(f"  Mean Error: {np.mean(errors):.6f} units\n")
            f.write(f"  Std Error: {np.std(errors):.6f} units\n")
            f.write(f"  Min Error: {np.min(errors):.6f} units\n")
            f.write(f"  Max Error: {np.max(errors):.6f} units\n\n")
            f.write("PER-SUBJECT ERRORS:\n")
            for subject, error in zip(subject_names, subject_errors):
                f.write(f"  {subject}: {error:.6f}\n")
        
        print(f"  ✓ Saved text report to data_report.txt")
        
        plt.show()

# Convenience function for command-line usage
def main():
    """Main function to run the processor"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Process Mocap and HPE data')
    parser.add_argument('--mocap-folder', default='mocap_jcp', help='Mocap data folder')
    parser.add_argument('--hpe-folder', default='hpe_jcp', help='HPE data folder')
    parser.add_argument('--mocap-csv', default='robot_welding_joint_center_positions.csv', 
                       help='Mocap CSV filename')
    parser.add_argument('--hpe-csv', default='hpe_joint_positions.csv', 
                       help='HPE CSV filename')
    parser.add_argument('--output-dir', default='DATA/jcp_npy_w_offsets_hpealigned', help='Output directory')
    
    args = parser.parse_args()
    
    # Create processor
    processor = UnifiedDataProcessor(
        mocap_folder="DATA/mocap_jcp_40hz",
        hpe_folder="DATA/cosmik_jcp",
        mocap_csv="joint_center_positions_with_offsets.csv",
        hpe_csv="3d_keypoints_filtered_aligned.csv",
        output_dir=args.output_dir
    )
    
    # Process data
    success = processor.process_all_data()
    
    if success:
        print("\n✅ SUCCESS! Your data is ready for LSTM training.")
        print("\nNext steps:")
        print("1. Review the generated reports in 'processed_data/'")
        print("2. Run the training script: python train_with_real_data.py")
        print("3. The LSTM will learn to map HPE → Mocap")
    else:
        print("\n❌ Processing failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
