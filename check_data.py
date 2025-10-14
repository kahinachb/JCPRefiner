import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

def load_data(mocap_path='base_hpe/mocap_all_subjects.npy',
                   hpe_path='base_hpe/hpe_all_subjects.npy',
                   metadata_path='base_hpe/metadata.json'):
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
    # if mocap_data.shape != hpe_data.shape:
    #     raise ValueError(f"Data shape mismatch! Mocap: {mocap_data.shape}, HPE: {hpe_data.shape}")
    
    return mocap_data, hpe_data, metadata

mocap_data, hpe_data, metadata = load_data(
    mocap_path='processed_data_test/mocap_all_subjects.npy',
    hpe_path='processed_data_test/hpe_all_subjects.npy',  # Update this path when you have HPE data
    metadata_path='processed_data_test/metadata.json'
)
print(hpe_data[0])
print(mocap_data[0])