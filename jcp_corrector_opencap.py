import os, json, pathlib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# ==============================
# CONFIG
# ==============================
SEQ_LEN     = 30
STRIDE      = 1
BATCH_SIZE  = 32
EPOCHS      = 2
LR          = 1e-4
HIDDEN      = 256
LAYERS      = 3
DROPOUT     = 0.2
MIDHIP_IDX  = 9

LAMBDA_JCP   = 0.1  # Weight for joint loss
LAMBDA_DENSE = 0.9   # Weight for dense marker loss

PATH_METADATA_JSON   = "DATA/jcp_hpe_mocap_mks_wou_head/metadata.json"
PATH_HPE_NPY         = "DATA/processed_data/mocap_all_subjects.npy" #take jcp mocap as input
PATH_DENSE_MOCAP_NPY = "DATA/jcp_hpe_mocap_mks_wou_head/mocap_all_subjects.npy" #anatomical mocap 
PATH_JCP_NPY         = "DATA/processed_data/mocap_all_subjects.npy" #take jcp mocap as reference


AUGMENTER_ROOT       = "lstm_opencap"
AUGMENTER_MODELNAME  = "LSTM"
SUBMODEL_LOWER       = "v0.3_lower"
SUBMODEL_UPPER       = "v0.3_upper"
SUBJECT_YAML_DIR     = "DATA//metadata"

OUT_DIR              = "models_tf_physical_dual_mocap"

LOWER_IDX = [2, 0, 1, 7, 8, 10, 11, 12, 13, 14, 15, 18, 19, 16, 17]
UPPER_IDX = [2, 0, 1, 3, 4, 5, 6]

# ==============================
# SUBJECT METADATA
# ==============================
from utils import read_subject_yaml

class SubjectMetaCache:
    def __init__(self, metadata_dir="metadata", default_mass=70.0, default_height=1.75):
        self.metadata_dir = metadata_dir
        self.cache = {}
        self.default_mass = float(default_mass)
        self.default_height = float(default_height)

    def get_subject_metadata(self, subject: str):
        if subject not in self.cache:
            yaml_path = f"{self.metadata_dir}/{subject}.yaml"
            try:
                subject_id, subject_height, subject_mass, gender = read_subject_yaml(yaml_path)
                self.cache[subject] = {
                    "id": subject_id,
                    "height": float(subject_height),
                    "mass": float(subject_mass),
                    "gender": gender,
                }
            except Exception as e:
                print(f"[WARN] Could not load YAML for '{subject}': {e}")
                self.cache[subject] = {
                    "id": subject, "height": self.default_height,
                    "mass": self.default_mass, "gender": None
                }
        return self.cache[subject]

# ==============================
# MARKER MAPPING
# ==============================
def create_marker_mapping():
    augmenter_markers = [
        'r.ASIS_study','L.ASIS_study','r.PSIS_study','L.PSIS_study','r_knee_study',
        'r_mknee_study','r_ankle_study','r_mankle_study','r_toe_study','r_5meta_study',
        'r_calc_study','L_knee_study','L_mknee_study','L_ankle_study','L_mankle_study',
        'L_toe_study','L_calc_study','L_5meta_study','r_shoulder_study','L_shoulder_study',
        'C7_study','r_thigh1_study','r_thigh2_study','r_thigh3_study','L_thigh1_study',
        'L_thigh2_study','L_thigh3_study','r_sh1_study','r_sh2_study','r_sh3_study',
        'L_sh1_study','L_sh2_study','L_sh3_study','RHJC_study','LHJC_study','r_lelbow_study',
        'r_melbow_study','r_lwrist_study','r_mwrist_study','L_lelbow_study','L_melbow_study',
        'L_lwrist_study','L_mwrist_study'
    ]
    marker_map = {
        'r.ASIS_study': 'RASI','L.ASIS_study': 'LASI','r.PSIS_study': 'RPSI','L.PSIS_study': 'LPSI',
        'r_knee_study': 'RKNE','r_mknee_study': 'RMKNE','r_ankle_study': 'RANK','r_mankle_study':'RMANK',
        'r_toe_study':'RTOE','r_5meta_study':'R5MHD','r_calc_study':'RHEE','L_knee_study':'LKNE',
        'L_mknee_study':'LMKNE','L_ankle_study':'LANK','L_mankle_study':'LMANK','L_toe_study':'LTOE',
        'L_5meta_study':'L5MHD','L_calc_study':'LHEE','r_shoulder_study':'RSHO','L_shoulder_study':'LSHO',
        'C7_study':'C7','r_lelbow_study':'RELB','r_melbow_study':'RMELB','r_lwrist_study':'RWRI',
        'r_mwrist_study':'RMWRI','L_lelbow_study':'LELB','L_melbow_study':'LMELB',
        'L_lwrist_study':'LWRI','L_mwrist_study':'LMWRI',
    }
    return augmenter_markers, marker_map

# ==============================
# DATA SEQUENCES
# ==============================
# def create_sequences_with_subjects(array_Jx3, metadata, seq_len=30, stride=1):
#     N, J, _ = array_Jx3.shape
#     subj_idx = metadata["common"]["subject_indices"]
#     subjects_all = [s for s in metadata["common"]["subjects"]
#                     if s in metadata["hpe"]["n_frames_per_subject"]]

#     X, seq_subjects, seq_starts = [], [], []
#     for s in subjects_all:
#         if s == "Zoe":
#             continue
#         else: 
#             start, end = subj_idx[s]
#             start = max(0, int(start))
#             end   = min(N, int(end))
#             for i in range(start, end - seq_len + 1, stride):
#                 X.append(array_Jx3[i:i+seq_len].reshape(seq_len, J*3))
#                 seq_subjects.append(s)
#                 seq_starts.append(i)
#     X = np.stack(X, axis=0).astype(np.float32)
#     return X, np.array(seq_subjects), np.array(seq_starts)

def create_sequences_for_subjects(array_Jx3, metadata, subjects_keep, seq_len=30, stride=1):
    N, J, _ = array_Jx3.shape
    subj_idx = metadata["common"]["subject_indices"]

    X, seq_subjects, seq_starts = [], [], []
    for s in subjects_keep:
        if s not in subj_idx:
            continue
        start, end = subj_idx[s]
        start = max(0, int(start))
        end   = min(N, int(end))
        for i in range(start, end - seq_len + 1, stride):
            X.append(array_Jx3[i:i+seq_len].reshape(seq_len, J*3))
            seq_subjects.append(s)
            seq_starts.append(i)
    X = np.stack(X, axis=0).astype(np.float32) if len(X) else np.zeros((0, seq_len, J*3), np.float32)
    return X, np.array(seq_subjects), np.array(seq_starts)

# ==============================
# LOAD AUGMENTER
# ==============================
def load_keras_submodel(model_dir):
    with open(os.path.join(model_dir, "model.json"), "r") as f:
        model_json = f.read()
    model = keras.models.model_from_json(model_json)
    model.load_weights(os.path.join(model_dir, "weights.h5"))
    return model

def load_mean_std(model_dir):
    mean_path = os.path.join(model_dir, "mean.npy")
    std_path  = os.path.join(model_dir, "std.npy")
    mean = np.load(mean_path) if os.path.isfile(mean_path) else None
    std  = np.load(std_path)  if os.path.isfile(std_path)  else None
    mean = tf.constant(mean, dtype=tf.float32) if mean is not None else None
    std  = tf.constant(std,  dtype=tf.float32) if std  is not None else None
    return mean, std

# ==============================
# INVERSE SCALER LAYER
# ==============================
class InverseScalerLayer(layers.Layer):
    """Convert from standardized space back to physical space."""
    def __init__(self, scaler_mean, scaler_std, **kwargs):
        super().__init__(**kwargs)
        self.mean = tf.constant(scaler_mean, dtype=tf.float32)
        self.std = tf.constant(scaler_std, dtype=tf.float32)
    
    def call(self, x):
        # x_physical = x_standardized * std + mean
        return x * self.std + self.mean

# ==============================
# AUGMENTER LAYER
# ==============================
class AugmenterLayer(layers.Layer):
    def __init__(self, lower_model, upper_model,
                 lower_idx, upper_idx,
                 lower_mean=None, lower_std=None,
                 upper_mean=None, upper_std=None,
                 common_indices=None,
                 midhip_idx=9, **kwargs):
        super().__init__(**kwargs)
        self.lower_model = lower_model
        self.upper_model = upper_model
        self.lower_idx = tf.constant(lower_idx, dtype=tf.int32)
        self.upper_idx = tf.constant(upper_idx, dtype=tf.int32)
        self.lower_mean = lower_mean
        self.lower_std  = lower_std
        self.upper_mean = upper_mean
        self.upper_std  = upper_std
        self.common_indices = tf.constant(common_indices, dtype=tf.int32) if common_indices is not None else None
        self.midhip_idx = int(midhip_idx)
        self.lower_model.trainable = False
        self.upper_model.trainable = False

    def call(self, inputs):
        """
        inputs: [joints_corr, height, mass]
          - joints_corr: [B,T,J*3] in PHYSICAL SPACE (meters)
          - height: [B,1] in meters
          - mass: [B,1] in kg
        returns: dense markers [B,T,K*3] in PHYSICAL SPACE (meters)
        """
        joints_corr, height, mass = inputs
        joints_corr = tf.cast(joints_corr, tf.float32)
        height = tf.cast(height, tf.float32)
        mass   = tf.cast(mass,   tf.float32)

        B = tf.shape(joints_corr)[0]
        T = tf.shape(joints_corr)[1]
        J = tf.shape(joints_corr)[2] // 3

        joints = tf.reshape(joints_corr, [B, T, J, 3])
        midhip = joints[:, :, self.midhip_idx:self.midhip_idx+1, :]
        j_norm = (joints - midhip) / tf.reshape(height, [B,1,1,1])

        feat_lower = tf.gather(j_norm, self.lower_idx, axis=2)
        feat_upper = tf.gather(j_norm, self.upper_idx, axis=2)

        def add_hm(x):
            x = tf.reshape(x, [B, T, -1])
            h = tf.tile(tf.reshape(height, [B,1,1]), [1,T,1])
            m = tf.tile(tf.reshape(mass,   [B,1,1]), [1,T,1])
            return tf.concat([x, h, m], axis=-1)

        in_lower = add_hm(feat_lower)
        in_upper = add_hm(feat_upper)

        if self.lower_mean is not None:
            in_lower = in_lower - self.lower_mean
        if self.lower_std is not None:
            in_lower = in_lower / self.lower_std
        if self.upper_mean is not None:
            in_upper = in_upper - self.upper_mean
        if self.upper_std is not None:
            in_upper = in_upper / self.upper_std

        out_lower = self.lower_model(in_lower, training=False)
        out_upper = self.upper_model(in_upper, training=False)

        def unnorm(out):
            out = out * tf.reshape(height, [B,1,1])
            out = tf.reshape(out, [B, T, -1, 3])
            return out + midhip

        dense_lower = unnorm(out_lower)
        dense_upper = unnorm(out_upper)
        dense_full  = tf.concat([dense_lower, dense_upper], axis=2)

        if self.common_indices is not None:
            dense_full = tf.gather(dense_full, self.common_indices, axis=2)

        K = tf.shape(dense_full)[2]
        dense_flat = tf.reshape(dense_full, [B, T, K*3])
        return dense_flat

# ==============================
# CORRECTOR MODEL
# ==============================
def build_joint_model(input_dim, hidden=256, layers_n=3, dropout=0.2, 
                      output_dim=None):
    """Corrects the ENTIRE sequence with temporal context"""
    inp = keras.Input(shape=(SEQ_LEN, input_dim), name="hpe_in")
    x = inp

    for _ in range(layers_n):
        x = layers.Bidirectional(
            layers.LSTM(hidden, return_sequences=True, dropout=dropout),
            merge_mode="concat"
        )(x)

    # Process each timestep
    x = layers.TimeDistributed(layers.Dense(hidden, activation="relu"))(x)
    x = layers.TimeDistributed(layers.Dropout(dropout))(x)
    
    out_dim = input_dim if output_dim is None else output_dim
    # Predict correction for ALL frames
    pred_seq = layers.TimeDistributed(
        layers.Dense(out_dim, activation=None)
    )(x)  # [B, T, J*3]
    
    return keras.Model(inp, pred_seq, name="joint_corrector_sequence")

def build_joint_model_lstm(input_dim, hidden=256, layers_n=3, dropout=0.2, output_dim=None):
    """Corrects the ENTIRE sequence using past-only (causal) context via LSTMs."""
    inp = keras.Input(shape=(SEQ_LEN, input_dim), name="hpe_in")
    x = inp

    for _ in range(layers_n):
        x = layers.LSTM(hidden, return_sequences=True, dropout=dropout)(x)

    # Per-timestep head
    x = layers.TimeDistributed(layers.Dense(hidden, activation="relu"))(x)
    x = layers.TimeDistributed(layers.Dropout(dropout))(x)

    out_dim = input_dim if output_dim is None else output_dim
    pred_seq = layers.TimeDistributed(layers.Dense(out_dim, activation=None))(x)
    return keras.Model(inp, pred_seq, name="joint_corrector_sequence_lstm")
# ==============================
# MAIN
# ==============================
def main():
    pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    # Load metadata
    with open(PATH_METADATA_JSON, "r") as f:
        meta_json = json.load(f)
    mocap_marker_names = meta_json["common"]["mocap_marker_names"]

    # Load arrays - all in PHYSICAL space
    hpe   = np.load(PATH_HPE_NPY)
    dense = np.load(PATH_DENSE_MOCAP_NPY)
    jcp   = np.load(PATH_JCP_NPY)
    
    N, J, _ = hpe.shape
    M = dense.shape[1]
    J_jcp = jcp.shape[1]
    
    print(f"HPE shape: {hpe.shape}, Dense shape: {dense.shape}, JCP shape: {jcp.shape}")
    
    # Verify dimensions
    assert hpe.shape[0] == dense.shape[0] == jcp.shape[0], "Frame count mismatch!"
    assert J == J_jcp, f"Joint count mismatch! HPE has {J} joints, JCP has {J_jcp}"

    # Distribution check
    print(f"\nDistribution Check:")
    print(f"HPE: mean={hpe.mean():.4f}, std={hpe.std():.4f}, range=[{hpe.min():.3f}, {hpe.max():.3f}]")
    print(f"JCP: mean={jcp.mean():.4f}, std={jcp.std():.4f}, range=[{jcp.min():.3f}, {jcp.max():.3f}]")
    print(f"Difference: {abs(hpe.mean() - jcp.mean()):.4f} meters")

    # Build sequences
    # X_all, seq_subjects, seq_starts = create_sequences_with_subjects(hpe, meta_json, SEQ_LEN, STRIDE)
    # Y_all, seq_subjects_y, _        = create_sequences_with_subjects(dense, meta_json, SEQ_LEN, STRIDE)
    # J_all, seq_subjects_j, _        = create_sequences_with_subjects(jcp, meta_json, SEQ_LEN, STRIDE)
    
    # S = X_all.shape[0]
    # print(f"Total sequences: {S}")
    # assert Y_all.shape[0] == S and J_all.shape[0] == S, "Sequence count mismatch!"

    # # Train/val/test split
    # idx = np.arange(S)
    # forced_subject = "Alessandro"
    # idx_forced_test = np.where(seq_subjects == forced_subject)[0]
    # idx_remaining = np.setdiff1d(idx, idx_forced_test)
    # idx_train, idx_val = train_test_split(idx_remaining, test_size=0.30, random_state=42)
    # idx_test = idx_forced_test

    # X_train, X_val, X_test = X_all[idx_train], X_all[idx_val], X_all[idx_test]
    # Y_train, Y_val, Y_test = Y_all[idx_train], Y_all[idx_val], Y_all[idx_test]
    # J_train, J_val, J_test = J_all[idx_train], J_all[idx_val], J_all[idx_test]
    # subj_train, subj_val, subj_test = seq_subjects[idx_train], seq_subjects[idx_val], seq_subjects[idx_test]

    # --- Define your fixed test subjects ---
    test_subjects = ["Alessandro"]

    # --- List all available subjects ---
    all_subjects = [s for s in meta_json["common"]["subjects"]
                    if s in meta_json["hpe"]["n_frames_per_subject"]]

    # --- Safety check ---
    for s in test_subjects:
        assert s in all_subjects, f"Subject {s} not found in dataset!"

    # --- Remove them from the pool ---
    remaining_subjects = [s for s in all_subjects if s not in test_subjects]

    # --- Randomly pick 2 validation subjects ---
    rng = np.random.default_rng(42)
    rng.shuffle(remaining_subjects)

    val_subjects = remaining_subjects[:2]
    train_subjects = remaining_subjects[2:]

    print("Train subjects:", train_subjects)
    print("Val subjects:", val_subjects)
    print("Test subjects:", test_subjects)

    # HPE inputs
    X_train, subj_train, _ = create_sequences_for_subjects(hpe, meta_json, train_subjects, SEQ_LEN, STRIDE)
    X_val,   subj_val,  _  = create_sequences_for_subjects(hpe, meta_json, val_subjects,   SEQ_LEN, STRIDE)
    X_test,  subj_test, _  = create_sequences_for_subjects(hpe, meta_json, test_subjects,  SEQ_LEN, STRIDE)

    # Corresponding targets (dense + jcp)
    Y_train, _, _ = create_sequences_for_subjects(dense, meta_json, train_subjects, SEQ_LEN, STRIDE)
    Y_val,   _, _ = create_sequences_for_subjects(dense, meta_json, val_subjects,   SEQ_LEN, STRIDE)
    Y_test,  _, _ = create_sequences_for_subjects(dense, meta_json, test_subjects,  SEQ_LEN, STRIDE)

    J_train, _, _ = create_sequences_for_subjects(jcp,   meta_json, train_subjects, SEQ_LEN, STRIDE)
    J_val,   _, _ = create_sequences_for_subjects(jcp,   meta_json, val_subjects,   SEQ_LEN, STRIDE)
    J_test,  _, _ = create_sequences_for_subjects(jcp,   meta_json, test_subjects,  SEQ_LEN, STRIDE)


    from collections import Counter
    print("\n" + "="*80)
    print("TRAIN/VAL/TEST SPLIT")
    print("="*80)
    for split_name, subjects in [("TRAIN", subj_train), ("VAL", subj_val), ("TEST", subj_test)]:
        counts = Counter(subjects)
        print(f"\n{split_name} ({len(subjects)} sequences):")
        for subj, cnt in sorted(counts.items()):
            print(f"  {subj:20s}: {cnt:5d}")
    print("="*80)

    # Marker alignment
    augmenter_markers, marker_map = create_marker_mapping()
    aug_idx, mocap_idx, common_names = [], [], []
    for i, aug_m in enumerate(augmenter_markers):
        mc = marker_map.get(aug_m)
        if mc in mocap_marker_names:
            aug_idx.append(i)
            mocap_idx.append(mocap_marker_names.index(mc))
            common_names.append(mc)
    K = len(aug_idx)
    print(f"\nUsing {K} common markers: {common_names}")

    # Slice dense markers to common ones
    def slice_common(Y):
        S_, T_, _ = Y.shape
        return Y.reshape(S_, T_, -1, 3)[:, :, mocap_idx, :].reshape(S_, T_, K*3)

    Y_train_c = slice_common(Y_train)
    Y_val_c   = slice_common(Y_val)
    Y_test_c  = slice_common(Y_test)

    # Keep targets in PHYSICAL space (all frames)
    J_train_gt = J_train.astype(np.float32)  # [S, T, J*3] PHYSICAL
    J_val_gt = J_val.astype(np.float32)
    J_test_gt = J_test.astype(np.float32)
    
    Y_train_c_gt = Y_train_c.astype(np.float32)  # [S, T, K*3] PHYSICAL
    Y_val_c_gt = Y_val_c.astype(np.float32)
    Y_test_c_gt = Y_test_c.astype(np.float32)

    print(f"\nTargets prepared (PHYSICAL SPACE, all frames):")
    print(f"   J_train_gt shape: {J_train_gt.shape}, range: [{J_train_gt.min():.3f}, {J_train_gt.max():.3f}]")
    print(f"   Y_train_c_gt shape: {Y_train_c_gt.shape}, range: [{Y_train_c_gt.min():.3f}, {Y_train_c_gt.max():.3f}]")

    # Standardize ONLY inputs
    sx_hpe = StandardScaler()
    
    Sh, T, Dx = X_train.shape

    # Scale HPE inputs
    X_train = sx_hpe.fit_transform(X_train.reshape(-1, Dx)).reshape(Sh, T, Dx)
    X_val = sx_hpe.transform(X_val.reshape(-1, Dx)).reshape(X_val.shape[0], T, Dx)
    X_test = sx_hpe.transform(X_test.reshape(-1, Dx)).reshape(X_test.shape[0], T, Dx)


    print("\nData scaling:")
    print(f"   X_train: standardized, mean={X_train.mean():.4f}, std={X_train.std():.4f}")
    print(f"   J_train_gt: PHYSICAL, range [{J_train_gt.min():.3f}, {J_train_gt.max():.3f}]")
    print(f"   Y_train_c_gt: PHYSICAL, range [{Y_train_c_gt.min():.3f}, {Y_train_c_gt.max():.3f}]")
    
    # Verify standardization
    assert abs(X_train.mean()) < 0.1, "X_train not properly centered!"
    assert abs(X_train.std() - 1.0) < 0.2, "X_train std should be ~1.0!"

    # Subject metadata
    meta_cache = SubjectMetaCache(metadata_dir=SUBJECT_YAML_DIR)
    def subjects_to_mass_height(subjects):
        mass = np.zeros((len(subjects), 1), dtype=np.float32)
        height = np.zeros((len(subjects), 1), dtype=np.float32)
        for i, s in enumerate(subjects):
            m = meta_cache.get_subject_metadata(s)
            mass[i, 0] = m["mass"]
            height[i, 0] = m["height"]
        return height, mass

    hgt_train, mass_train = subjects_to_mass_height(subj_train)
    hgt_val, mass_val = subjects_to_mass_height(subj_val)
    hgt_test, mass_test = subjects_to_mass_height(subj_test)

    # Load augmenter
    lower_dir = os.path.join(AUGMENTER_ROOT, AUGMENTER_MODELNAME, SUBMODEL_LOWER)
    upper_dir = os.path.join(AUGMENTER_ROOT, AUGMENTER_MODELNAME, SUBMODEL_UPPER)
    mdl_lower = load_keras_submodel(lower_dir)
    mdl_upper = load_keras_submodel(upper_dir)
    lower_mean, lower_std = load_mean_std(lower_dir)
    upper_mean, upper_std = load_mean_std(upper_dir)

    # Build corrector - outputs full sequence
    corrector = build_joint_model(
        input_dim=J*3, hidden=HIDDEN, layers_n=LAYERS,
        dropout=DROPOUT, output_dim=J*3
    )

    # corrector = build_joint_model_lstm(
    #     input_dim=J*3, hidden=HIDDEN, layers_n=LAYERS,
    #     dropout=DROPOUT, output_dim=J*3
    # )

    # Build augmenter layer
    aug_layer = AugmenterLayer(
        lower_model=mdl_lower, upper_model=mdl_upper,
        lower_idx=LOWER_IDX, upper_idx=UPPER_IDX,
        lower_mean=lower_mean, lower_std=lower_std,
        upper_mean=upper_mean, upper_std=upper_std,
        common_indices=aug_idx,
        midhip_idx=MIDHIP_IDX
    )

    # Build cascade - Both losses in PHYSICAL space
    print("\nBuilding cascade model (both losses in physical space)...")
    inp_hpe = keras.Input(shape=(SEQ_LEN, J*3), name="hpe_in")
    inp_height = keras.Input(shape=(1,), name="height_in")
    inp_mass = keras.Input(shape=(1,), name="mass_in")

    # Corrector outputs full corrected sequence in standardized space
    jcp_seq_std = corrector(inp_hpe)  # [B, T, J*3] standardized

    # Convert to physical space
    inverse_scaler = InverseScalerLayer(
        scaler_mean=sx_hpe.mean_, 
        scaler_std=sx_hpe.scale_
    )
    jcp_seq_phys = inverse_scaler(jcp_seq_std)  # [B, T, J*3] physical

    # Output 1: JCP sequence in PHYSICAL space (for loss)
    jcp_output = layers.Lambda(lambda x: x, name="jcp_corrected_phys")(jcp_seq_phys)
    
    # Augmenter receives corrected sequence with temporal variation
    dense_pred = aug_layer([jcp_seq_phys, inp_height, inp_mass])  # [B, T, K*3]
    
    # Output 2: Dense markers sequence in PHYSICAL space (for loss)
    dense_output = layers.Lambda(lambda x: x, name="dense_markers")(dense_pred)

    model = keras.Model(
        inputs=[inp_hpe, inp_height, inp_mass], 
        outputs=[jcp_output, dense_output],
        name="cascade_physical_loss"
    )

    # Compile: Both losses in PHYSICAL space on ALL frames
    model.compile(
        optimizer=keras.optimizers.Adam(LR),
        loss={
            "jcp_corrected_phys": "mse",  # MSE in meters² on all frames
            "dense_markers": "mse"        # MSE in meters² on all frames
        },
        loss_weights={
            "jcp_corrected_phys": LAMBDA_JCP,
            "dense_markers": LAMBDA_DENSE
        }
    )
    
    model.summary()

    # Callback
    best_corrector_path = os.path.join(OUT_DIR, "corrector.keras")
    
    class SaveBestCorrectorCallback(keras.callbacks.Callback):
        def __init__(self, corrector_model, save_path):
            super().__init__()
            self.corrector_model = corrector_model
            self.save_path = save_path
            self.best_val_loss = float('inf')
        
        def on_epoch_end(self, epoch, logs=None):
            current_val_loss = logs.get('val_loss')
            val_jcp_loss = logs.get('val_jcp_corrected_phys_loss')
            val_dense_loss = logs.get('val_dense_markers_loss')
            
            if current_val_loss is not None and current_val_loss < self.best_val_loss:
                self.best_val_loss = current_val_loss
                self.corrector_model.save(self.save_path)
                
                # Both in physical space (meters²)
                jcp_rmse_mm = np.sqrt(val_jcp_loss) * 1000
                dense_rmse_mm = np.sqrt(val_dense_loss) * 1000
                
                print(f"\nEpoch {epoch+1}:")
                print(f"   Total val_loss={current_val_loss:.6f}")
                print(f"   JCP RMSE: {jcp_rmse_mm:.2f} mm (all frames)")
                print(f"   Dense RMSE: {dense_rmse_mm:.2f} mm (all frames)")
                print(f"   Saved corrector")

    save_best_callback = SaveBestCorrectorCallback(corrector, best_corrector_path)

    # Train
    print("\nStarting training...")
    print(f"Both losses in PHYSICAL space (meters²) on ALL frames")
    print(f"JCP weight: {LAMBDA_JCP}, Dense weight: {LAMBDA_DENSE}")
    
    history = model.fit(
        x={"hpe_in": X_train, "height_in": hgt_train, "mass_in": mass_train},
        y={
            "jcp_corrected_phys": J_train_gt,      # [S, T, J*3] Physical
            "dense_markers": Y_train_c_gt          # [S, T, K*3] Physical
        },
        validation_data=(
            {"hpe_in": X_val, "height_in": hgt_val, "mass_in": mass_val},
            {
                "jcp_corrected_phys": J_val_gt,
                "dense_markers": Y_val_c_gt
            }
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2,
        callbacks=[save_best_callback]
    )
    
    # Plot loss
    plt.figure(figsize=(8,6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Evolution')
    plt.legend()
    out_fig_path = os.path.join(OUT_DIR, "loss_curve.png")
    plt.savefig(out_fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nLoss curve saved at {out_fig_path}")

    # Evaluation
    print("\nEvaluating best model...")
    best_corrector = keras.models.load_model(best_corrector_path)
    
    # Rebuild eval model
    eval_jcp_seq_std = best_corrector(inp_hpe)
    eval_jcp_seq_phys = inverse_scaler(eval_jcp_seq_std)
    eval_dense = aug_layer([eval_jcp_seq_phys, inp_height, inp_mass])
    
    eval_model = keras.Model(
        [inp_hpe, inp_height, inp_mass], 
        [eval_jcp_seq_phys, eval_dense]
    )

    # Predict
    jcp_pred_phys, dense_pred = eval_model.predict({
        "hpe_in": X_test, "height_in": hgt_test, "mass_in": mass_test
    }, verbose=2)

    # Evaluate on LAST frame (what matters for inference)
    jcp_pred_last = jcp_pred_phys[:, -1, :].reshape(-1, J, 3)
    jcp_gt_last = J_test_gt[:, -1, :].reshape(-1, J, 3)
    
    dense_pred_last = dense_pred[:, -1, :].reshape(-1, K, 3)
    dense_gt_last = Y_test_c_gt[:, -1, :].reshape(-1, K, 3)

    # Compute metrics (physical space)
    err_jcp = np.linalg.norm(jcp_pred_last - jcp_gt_last, axis=-1)
    mpjpe_jcp_mm = float(err_jcp.mean() * 1000.0)
    rmse_jcp_mm = float(np.sqrt(np.mean((jcp_pred_last - jcp_gt_last)**2)) * 1000.0)

    err_dense = np.linalg.norm(dense_pred_last - dense_gt_last, axis=-1)
    mpjpe_dense_mm = float(err_dense.mean() * 1000.0)
    rmse_dense_mm = float(np.sqrt(np.mean((dense_pred_last - dense_gt_last)**2)) * 1000.0)

    print(f"\n" + "="*80)
    print("TEST RESULTS (Last Frame Only)")
    print("="*80)
    print(f"\nJoint Centers (JCP):")
    print(f"   MPJPE: {mpjpe_jcp_mm:.2f} mm")
    print(f"   RMSE:  {rmse_jcp_mm:.2f} mm")
    print(f"\nDense Markers:")
    print(f"   MPJPE: {mpjpe_dense_mm:.2f} mm")
    print(f"   RMSE:  {rmse_dense_mm:.2f} mm")
    # Error per sequence (averaged over joints)
    per_sequence_error = err_jcp.mean(axis=1)  # [N_test,]
    print(f"Best sequence: {per_sequence_error.min():.2f} mm")
    print(f"Worst sequence: {per_sequence_error.max():.2f} mm")

    # Error per joint (averaged over sequences)
    per_joint_error = err_jcp.mean(axis=0)  # [J,]
    for j in range(J):
        print(f"Joint {j}: {per_joint_error[j]*1000:.2f} mm")

    # Save predictions (last frame in physical space)
    df_pred_jcp = pd.DataFrame(
        jcp_pred_last.reshape(-1, J*3),
        columns=[f"joint_{j}_{c}" for j in range(J) for c in ['x','y','z']]
    )
    df_pred_jcp.to_csv(os.path.join(OUT_DIR, "predictions_jcp.csv"), index=False)

    df_pred_dense = pd.DataFrame(
        dense_pred_last.reshape(-1, K*3),
        columns=[f"{common_names[k]}_{c}" for k in range(K) for c in ['x','y','z']]
    )
    df_pred_dense.to_csv(os.path.join(OUT_DIR, "predictions_dense.csv"), index=False)

    df_gt_dense = pd.DataFrame(
        dense_gt_last.reshape(-1, K*3),
        columns=[f"{common_names[k]}_{c}" for k in range(K) for c in ['x','y','z']]
    )
    df_gt_dense.to_csv(os.path.join(OUT_DIR, "ground_truth_dense.csv"), index=False)

    # Save artifacts
    joblib.dump(sx_hpe, os.path.join(OUT_DIR, "scaler_hpe.pkl"))

    with open(os.path.join(OUT_DIR, "common_markers.json"), "w") as f:
        json.dump({
            "common_markers": common_names,
            "aug_indices": aug_idx,
            "mocap_indices": mocap_idx
        }, f, indent=2)

    with open(os.path.join(OUT_DIR, "training_config.json"), "w") as f:
        json.dump({
            "seq_len": SEQ_LEN, "stride": STRIDE, "batch_size": BATCH_SIZE,
            "epochs": EPOCHS, "lr": LR, "hidden": HIDDEN, "layers": LAYERS,
            "dropout": DROPOUT, "midhip_idx": MIDHIP_IDX,
            "lower_idx": LOWER_IDX, "upper_idx": UPPER_IDX,
            "n_joints": J, "n_common_markers": K,
            "loss_space": "physical",
            "loss_on_all_frames": True,
            "dual_loss": True,
            "lambda_jcp": LAMBDA_JCP,
            "lambda_dense": LAMBDA_DENSE,
            "best_val_loss_total": float(save_best_callback.best_val_loss),
            "test_jcp_mpjpe_mm": mpjpe_jcp_mm,
            "test_jcp_rmse_mm": rmse_jcp_mm,
            "test_dense_mpjpe_mm": mpjpe_dense_mm,
            "test_dense_rmse_mm": rmse_dense_mm
        }, f, indent=2)

    print(f"\nTraining complete. Artifacts in: {OUT_DIR}")
if __name__ == "__main__":
    main()