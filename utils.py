import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_subject_yaml(file_path):
    """
    Lit un fichier YAML et retourne directement id, height, weight et gender.
    """
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)
    
    subject_id = data.get('id')
    height = data.get('height')
    weight = data.get('weight')
    gender = data.get('gender')
    
    return subject_id, height, weight, gender
def save_to_csv(data, output_path, header=None):
    """Save 3D keypoints to a CSV file with optional header."""
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False, header=header if header is not None else False)
    print(f"Saved {len(data)} frames to {output_path}")

def read_mks_data(data_markers, start_sample=0, converter = 1.0):
    #the mks are ordered in a csv like this : "time,r.ASIS_study_x,r.ASIS_study_y,r.ASIS_study_z...."
    """    
    Parameters:
        data_markers (pd.DataFrame): The input DataFrame containing marker data.
        start_sample (int): The index of the sample to start processing from.
        time_column (str): The name of the time column in the DataFrame.
        
    Returns:
        list: A list of dictionaries where each dictionary contains markers with 3D coordinates.
        dict: A dictionary representing the markers and their 3D coordinates for the specified start_sample.
    """
    # Extract marker column names
    marker_columns = [col[:-2] for col in data_markers.columns if col.endswith("_x")]
    
    # Initialize the result list
    result_markers = []
    
    # Iterate over each row in the DataFrame
    for _, row in data_markers.iterrows():
        frame_dict = {}
        for marker in marker_columns:
            x = row[f"{marker}_x"] / converter  #convert to m
            y = row[f"{marker}_y"]/ converter
            z = row[f"{marker}_z"]/ converter
            frame_dict[marker] = np.array([x, y, z])  # Store as a NumPy array
        result_markers.append(frame_dict)
    
    # Get the data for the specified start_sample
    start_sample_mks = result_markers[start_sample]
    
    return result_markers, start_sample_mks

def kabsch_global(P_cam_seq, P_mocap_seq, weights=None):
    """
    P_cam_seq, P_mocap_seq: arrays (T, N, 3) temporally and point-wise aligned.  
    Computes a single (R, t) that aligns everything (cam -> mocap) by minimizing the sum of errors.

    """

    assert P_cam_seq.shape == P_mocap_seq.shape and P_cam_seq.shape[-1] == 3
    T, N, _ = P_cam_seq.shape
    X = P_cam_seq.reshape(T*N, 3)
    Y = P_mocap_seq.reshape(T*N, 3)

    if weights is not None:
        w = np.asarray(weights).reshape(T, N)
        w = w / (w.sum() + 1e-12)
        w = w.reshape(T*N, 1)
        Xc = (X * w).sum(axis=0)     # weighted means
        Yc = (Y * w).sum(axis=0)
        X0 = X - Xc
        Y0 = Y - Yc
        H = (Y0 * w).T @ X0
    else:
        Xc = X.mean(axis=0)
        Yc = Y.mean(axis=0)
        X0 = X - Xc
        Y0 = Y - Yc
        H = Y0.T @ X0

    U, S, Vt = np.linalg.svd(H)
    R = U @ Vt
    if np.linalg.det(R) < 0:  # corrige réflexion
        U[:, -1] *= -1
        R = U @ Vt
    t = Yc - R @ Xc

    X_align = (R @ X.T).T + t
    rms = np.sqrt(np.mean(np.sum((X_align - Y)**2, axis=1)))
    return R, t, rms

def plot_aligned_markers(P_mocap_seq, P_hpe_seq_aligned, marker_names):

    if isinstance(P_hpe_seq_aligned, pd.DataFrame):
        P_hpe_seq_aligned = P_hpe_seq_aligned.to_numpy().reshape(P_mocap_seq.shape)

    T, N, _ = P_mocap_seq.shape
    fig, axes = plt.subplots(N, 3, figsize=(12, 3*N), sharex=True)
    if N == 1:  # cas spécial 1 marker
        axes = axes.reshape(1, 3)

    for i, mk in enumerate(marker_names):
        for j, coord in enumerate(["x", "y", "z"]):
            ax = axes[i, j]
            ax.plot(P_mocap_seq[:, i, j], "r", label="mocap" if i==0 and j==0 else "")
            ax.plot(P_hpe_seq_aligned[:, i, j], "g", label="hpe" if i==0 and j==0 else "")
            ax.set_title(f"{mk} - {coord}")
            if i == N-1:
                ax.set_xlabel("Frame")
    axes[0,0].legend()
    plt.tight_layout()
    plt.show()


def compute_mpjpe(P_pred, P_gt):
    """
    P_pred, P_gt: arrays (T, N, 3)
    Returns mean per-joint position error 
    """
    assert P_pred.shape == P_gt.shape
    errors = np.linalg.norm(P_pred - P_gt, axis=2)  # (T,N) distance per joint per frame
    mpjpe = errors.mean()                            # moyenne sur tous les joints et frames
    return mpjpe