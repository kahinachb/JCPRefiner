from utils import *
import pandas as pd

s = "Zoe"
start_sample = 0
jcp_w_head = f'/home/kchalabi/Documents/THESE/JCP_corrector/DATA/mocap_jcp/{s}/robot_welding_joint_center_positions_2.csv'
jcp_out_head = f'/home/kchalabi/Documents/THESE/JCP_corrector/DATA/mocap_jcp_40hz/{s}/robot_welding/joint_center_positions_with_offsets.csv'

df_w_head = pd.read_csv(jcp_w_head)
jcp_w_head, start_sample_dict = read_mks_data(df_w_head, start_sample=start_sample,converter = 1000.0)
df_out_head = pd.read_csv(jcp_out_head)
jcp_out_head, start_dict = read_mks_data(df_out_head, start_sample=start_sample,converter = 1.0)

n_frames = min(len(jcp_w_head), len(jcp_out_head))
print(f"Nombre de frames considérées : {n_frames}")

markers_to_add = ['FHD', 'LHD', 'RHD']

for i in range(n_frames):
    for m in markers_to_add:
        if m in jcp_w_head[i]:
            jcp_out_head[i][m] = jcp_w_head[i][m]

merged_df = []
for frame_dict in jcp_out_head:
    row = {}
    for mk, coords in frame_dict.items():
        row[f"{mk}_x"], row[f"{mk}_y"], row[f"{mk}_z"] = coords
    merged_df.append(row)

merged_df = pd.DataFrame(merged_df)
merged_df.to_csv(f'/home/kchalabi/Documents/THESE/JCP_corrector/DATA/mocap_jcp_40hz/{s}/robot_welding/joint_center_positions_with_offsets_with_head.csv', index=False)

