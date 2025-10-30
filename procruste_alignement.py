import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from  utils  import read_mks_data
import pandas as pd
from utils import kabsch_global, plot_aligned_markers,compute_mpjpe

def get_marker_array(mks_list, marker_name):
    """
    mks_list: list of dicts with keys 'name' and 'value'
    marker_name: str
    returns: array (T,3)
    """
    for mk in mks_list:
        if mk["name"] == marker_name:
            return mk["value"]
    raise ValueError(f"Marker {marker_name} not found")


# === Load data ===
subject = "Zoe"
task = "robot_welding"
path_to_jcp_mocap= f"/home/kchalabi/Documents/THESE/JCP_corrector/DATA/mocap_jcp_40hz/{subject}/{task}/joint_center_positions_with_offsets.csv"
df_mocap = pd.read_csv(path_to_jcp_mocap)
mks_mocap, start_sample_mocap = read_mks_data(df_mocap, start_sample=0,converter=1.0)


path_to_jcp_hpe= f"/home/kchalabi/Documents/THESE/JCP_corrector/DATA/cosmik_jcp/{subject}/3d_keypoints_filtered.csv"
df_hpe = pd.read_csv(path_to_jcp_hpe)
mks_hpe, start_sample_hpe = read_mks_data(df_hpe, start_sample=0,converter=1.0)
# print(mks_mocap)
# print(mks_mocap[0])


#take common markers
common_mks = sorted(set(start_sample_mocap.keys()) & set(start_sample_hpe.keys()))
print("Common markers:", common_mks)

T = len(mks_mocap)
N = len(common_mks)

P_mocap_seq = np.zeros((T, N, 3))
P_hpe_seq   = np.zeros((T, N, 3))
jcp_seq = np.zeros((len(mks_hpe), len(start_sample_hpe.keys()), 3))

for i, mk in enumerate(common_mks):
    P_mocap_seq[:, i, :] = np.stack([frame[mk] for frame in mks_mocap], axis=0)
    P_hpe_seq[:, i, :]   = np.stack([frame[mk] for frame in mks_hpe], axis=0)

for j, k in enumerate(start_sample_hpe.keys()):
    print(k)
    jcp_seq[:, j, :]   = np.stack([frame[k] for frame in mks_hpe], axis=0)


R, t, rms = kabsch_global(P_hpe_seq, P_mocap_seq)
print("RMS alignment error:", rms)

jcp_seq_aligned = (jcp_seq @ R.T) + t  # (T,N,3)
jcp_common_seq_aligned = np.stack([jcp_seq_aligned[:, list(start_sample_hpe.keys()).index(mk), :] 
                                   for mk in common_mks], axis=1)

plot_aligned_markers(P_mocap_seq, jcp_common_seq_aligned, common_mks)
err_R = np.linalg.norm(jcp_common_seq_aligned - P_mocap_seq, axis=2).mean()
print(err_R)

mpjpe = compute_mpjpe(jcp_common_seq_aligned, P_mocap_seq)
print("MPJPE (m):", mpjpe)

T, N, _ = jcp_seq_aligned.shape
df_hpe_aligned = pd.DataFrame(jcp_seq_aligned.reshape(T, 3*N),
                              columns=df_hpe.columns,
                              index=df_hpe.index)

output_csv = f"/home/kchalabi/Documents/THESE/JCP_corrector/DATA/cosmik_jcp/{subject}/3d_keypoints_filtered_aligned.csv"
df_hpe_aligned.to_csv(output_csv, index=False)
print("Aligned HPE saved to:", output_csv)