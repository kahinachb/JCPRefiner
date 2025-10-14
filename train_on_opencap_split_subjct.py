# tf_cascade_train.py
import os, json, pathlib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# ==============================
# 0) CONFIG (EDIT PATHS IF NEEDED)
# ==============================
SEQ_LEN     = 30
STRIDE      = 1
BATCH_SIZE  = 32
EPOCHS      = 400
LR          = 1e-4
HIDDEN      = 256
LAYERS      = 3
DROPOUT     = 0.2
MIDHIP_IDX  = 9      # check this is mid-hip in your 26-keypoint layout

PATH_METADATA_JSON   = "/datasets/jcp_training/jcp_hpe_mocap_mks/metadata.json"
PATH_HPE_NPY         = "/datasets/jcp_training/jcp_hpe_mocap_mks/hpe_all_subjects.npy"       # [N, J, 3]
PATH_DENSE_MOCAP_NPY = "/datasets/jcp_training/jcp_hpe_mocap_mks/mocap_all_subjects.npy"     # [N, M, 3] (dense markers)

AUGMENTER_ROOT       = "lstm_opencap"                                 # root dir
AUGMENTER_MODELNAME  = "LSTM"
SUBMODEL_LOWER       = "v0.3_lower"
SUBMODEL_UPPER       = "v0.3_upper"
SUBJECT_YAML_DIR     = "/datasets/jcp_training/metadata"                                     # *.yaml per subject

OUT_DIR              = "models_tf"

# Lower/upper feature marker indices (as in your augmenter)
LOWER_IDX = [2, 0, 1, 7, 8, 10, 11, 12, 13, 14, 15, 18, 19, 16, 17]
UPPER_IDX = [2, 0, 1, 3, 4, 5, 6]


# ==============================
# 1) SUBJECT METADATA (mass/height)
# ==============================
from utils import read_subject_yaml  # you already have this

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
                print(f"[WARN] Could not load YAML for '{subject}' at {yaml_path}: {e}")
                self.cache[subject] = {
                    "id": subject, "height": self.default_height, "mass": self.default_mass, "gender": None
                }
        return self.cache[subject]


# ==============================
# 2) MARKER NAME MAPPING (augmenter -> mocap)
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
# 3) DATA: build sequences WITH subject labels (for metadata)
# ==============================
def create_sequences_with_subjects(array_Jx3, metadata, seq_len=30, stride=1):
    """
    array_Jx3: [N, J, 3]
    Returns:
      X: [S, T, J*3]
      seq_subjects: np.array (S,)
      seq_starts:   np.array (S,)
    """
    N, J, _ = array_Jx3.shape
    subj_idx = metadata["common"]["subject_indices"]          # dict: subj -> [start, end)
    subjects_all = [s for s in metadata["common"]["subjects"]
                    if s in metadata["hpe"]["n_frames_per_subject"]]

    X, seq_subjects, seq_starts = [], [], []
    for s in subjects_all:
        # if s== "Alessandro" or s == "Zoe":
        #     continue
        start, end = subj_idx[s]
        start = max(0, int(start))
        end   = min(N, int(end))
        for i in range(start, end - seq_len + 1, stride):
            X.append(array_Jx3[i:i+seq_len].reshape(seq_len, J*3))
            seq_subjects.append(s)
            seq_starts.append(i)
    X = np.stack(X, axis=0).astype(np.float32)
    return X, np.array(seq_subjects), np.array(seq_starts)


# ==============================
# 4) LOAD AUGMENTER SUBMODELS + NORM
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
# 5) KERAS LAYER: AugmenterLayer (TF-graph friendly)
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

        # Freeze submodels by default
        self.lower_model.trainable = False
        self.upper_model.trainable = False

    def call(self, inputs):
        """
        inputs: [joints_corr, height, mass]
          - joints_corr: [B,T,J*3]
          - height:      [B,1]
          - mass:        [B,1]
        returns: dense markers aligned [B,T,K*3]
        """
        joints_corr, height, mass = inputs
        joints_corr = tf.cast(joints_corr, tf.float32)
        height = tf.cast(height, tf.float32)
        mass   = tf.cast(mass,   tf.float32)

        B = tf.shape(joints_corr)[0]
        T = tf.shape(joints_corr)[1]
        J = tf.shape(joints_corr)[2] // 3

        joints = tf.reshape(joints_corr, [B, T, J, 3])     # [B,T,J,3]
        midhip = joints[:, :, self.midhip_idx:self.midhip_idx+1, :]  # [B,T,1,3]
        j_norm = (joints - midhip) / tf.reshape(height, [B,1,1,1])   # center & scale

        # Select features
        feat_lower = tf.gather(j_norm, self.lower_idx, axis=2)  # [B,T,15,3]
        feat_upper = tf.gather(j_norm, self.upper_idx, axis=2)  # [B,T, 7,3]

        # Flatten + append height/mass per frame
        def add_hm(x):
            x = tf.reshape(x, [B, T, -1])                        # [B,T,F]
            h = tf.tile(tf.reshape(height, [B,1,1]), [1,T,1])    # [B,T,1]
            m = tf.tile(tf.reshape(mass,   [B,1,1]), [1,T,1])    # [B,T,1]
            return tf.concat([x, h, m], axis=-1)                 # [B,T,F+2]

        in_lower = add_hm(feat_lower)
        in_upper = add_hm(feat_upper)

        # Feature normalization (if mean/std provided)
        if self.lower_mean is not None:
            in_lower = in_lower - self.lower_mean
        if self.lower_std is not None:
            in_lower = in_lower / self.lower_std
        if self.upper_mean is not None:
            in_upper = in_upper - self.upper_mean
        if self.upper_std is not None:
            in_upper = in_upper / self.upper_std

        # Call submodels: each should output [B,T,D] (D = markers*3)
        out_lower = self.lower_model(in_lower, training=False)
        out_upper = self.upper_model(in_upper, training=False)

        # Unnormalize back to metric and add reference
        def unnorm(out):
            out = out * tf.reshape(height, [B,1,1])              # [B,T,D]
            out = tf.reshape(out, [B, T, -1, 3])                 # [B,T,M,3]
            return out + midhip

        dense_lower = unnorm(out_lower)                          # [B,T,L,3]
        dense_upper = unnorm(out_upper)                          # [B,T,U,3]
        dense_full  = tf.concat([dense_lower, dense_upper], axis=2)  # [B,T,L+U,3]

        # Align to mocap order if indices provided
        if self.common_indices is not None:
            dense_full = tf.gather(dense_full, self.common_indices, axis=2)  # [B,T,K,3]

        K = tf.shape(dense_full)[2]
        dense_flat = tf.reshape(dense_full, [B, T, K*3])         # [B,T,K*3]
        return dense_flat


# ==============================
# 6) CORRECTOR (BiLSTM + residual)
# ==============================
def build_corrector(input_dim, hidden=256, layers_n=3, dropout=0.2, output_dim=None):
    inp = keras.Input(shape=(SEQ_LEN, input_dim), name="hpe_in")
    x = inp
    for _ in range(layers_n):
        x = layers.Bidirectional(
            layers.LSTM(hidden, return_sequences=True, dropout=dropout),
            merge_mode="concat"
        )(x)
    x = layers.Dense(hidden, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    out_dim = input_dim if output_dim is None else output_dim
    corr = layers.Dense(out_dim, name="joint_delta")(x)
    jcp_corrected = layers.Add(name="residual_add")([corr, inp])
    return keras.Model(inp, jcp_corrected, name="joint_corrector")


# ==============================
# 7) MAIN
# ==============================
def main():
    pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    # ---- Load metadata JSON ----
    with open(PATH_METADATA_JSON, "r") as f:
        meta_json = json.load(f)
    mocap_marker_names = meta_json["common"]["mocap_marker_names"]

    # ---- Load arrays ----
    hpe   = np.load(PATH_HPE_NPY)         # [N, J, 3]
    dense = np.load(PATH_DENSE_MOCAP_NPY) # [N, M, 3]
    N, J, _ = hpe.shape
    M = dense.shape[1]
    print(hpe.shape)

    # ---- Build sequences WITH subjects ----
    X_all, seq_subjects, seq_starts = create_sequences_with_subjects(hpe,   meta_json, SEQ_LEN, STRIDE)
    Y_all, seq_subjects_y, _        = create_sequences_with_subjects(dense, meta_json, SEQ_LEN, STRIDE)

    assert X_all.shape[0] == Y_all.shape[0] == len(seq_subjects), "Sequence count mismatch."

    S = X_all.shape[0]
    print(f"Total sequences: {S}")

    # ---- Train/Val/Test split BY SUBJECT (not random sequences) ----
    from collections import Counter
    
    # Count sequences per subject
    subject_counts = Counter(seq_subjects)
    all_subjects = sorted(subject_counts.keys())
    
    print("\n" + "="*80)
    print("SEQUENCES PER SUBJECT")
    print("="*80)
    for subj in all_subjects:
        print(f"  {subj:25s}: {subject_counts[subj]:6d} sequences")
    print("="*80 + "\n")
    
    # Shuffle subjects for random selection (keeps sequences from same subject together)
    np.random.seed(42)  # For reproducibility
    shuffled_subjects = np.array(all_subjects.copy())
    np.random.shuffle(shuffled_subjects)
    
    # Decide how many subjects for each split
    n_subjects = len(shuffled_subjects)
    n_val = max(1, int(n_subjects * 0.15))   # 15% of subjects
    n_test = max(1, int(n_subjects * 0.15))  # 15% of subjects
    
    # Assign subjects to each split
    VAL_SUBJECTS = shuffled_subjects[:n_val].tolist()
    TEST_SUBJECTS = shuffled_subjects[n_val:n_val+n_test].tolist()
    TRAIN_SUBJECTS = shuffled_subjects[n_val+n_test:].tolist()
    
    print(f"Subject assignment (seed=42):")
    print(f"  TRAIN ({len(TRAIN_SUBJECTS)} subjects): {TRAIN_SUBJECTS}")
    print(f"  VAL   ({len(VAL_SUBJECTS)} subjects):   {VAL_SUBJECTS}")
    print(f"  TEST  ({len(TEST_SUBJECTS)} subjects):  {TEST_SUBJECTS}")
    print()
    
    
    # Create masks for each split
    train_mask = np.array([s in TRAIN_SUBJECTS for s in seq_subjects])
    val_mask = np.array([s in VAL_SUBJECTS for s in seq_subjects])
    test_mask = np.array([s in TEST_SUBJECTS for s in seq_subjects])
    
    # Get indices
    idx_train = np.where(train_mask)[0]
    idx_val = np.where(val_mask)[0]
    idx_test = np.where(test_mask)[0]
    
    # Split data
    X_train, X_val, X_test = X_all[idx_train], X_all[idx_val], X_all[idx_test]
    Y_train, Y_val, Y_test = Y_all[idx_train], Y_all[idx_val], Y_all[idx_test]
    subj_train, subj_val, subj_test = seq_subjects[idx_train], seq_subjects[idx_val], seq_subjects[idx_test]
    
    # Print distribution
    print("\n" + "="*80)
    print("TRAIN/VAL/TEST SPLIT BY SUBJECT")
    print("="*80)

    def print_subject_distribution(subjects, split_name):
        counts = Counter(subjects)
        total = len(subjects)
        print(f"\n{split_name} SET ({total} sequences, {len(counts)} subjects):")
        for subject, count in sorted(counts.items()):
            pct = 100 * count / total if total > 0 else 0
            print(f"  {subject:25s}: {count:6d} sequences ({pct:5.1f}%)")

    print_subject_distribution(subj_train, "TRAIN")
    print_subject_distribution(subj_val, "VALIDATION")
    print_subject_distribution(subj_test, "TEST")
    print("="*80 + "\n")
    print("\nValidation subjects represent {:.1f}% of total sequences".format(
    100 * len(idx_val) / S))
    print("Test subjects represent {:.1f}% of total sequences".format(
        100 * len(idx_test) / S))
    

    # ---- Build marker alignment indices (augmenter order -> mocap names) ----
    augmenter_markers, marker_map = create_marker_mapping()
    
    aug_idx   = []  # indices into augmenter output (prediction)
    mocap_idx = []  # indices into mocap labels (ground truth)
    common_names = []

    for i, aug_m in enumerate(augmenter_markers):
        mc = marker_map.get(aug_m)
        if mc in mocap_marker_names:
            aug_idx.append(i)                               # position in augmenter order
            mocap_idx.append(mocap_marker_names.index(mc))  # position in mocap order
            common_names.append(mc)

    K = len(aug_idx)
    print(f"Using {K} common markers:", common_names)

    # Slice dense labels to common order ONCE
    def slice_common(Y):
        # Y: [S,T,M*3] -> [S,T,K*3] using mocap indices
        S_, T_, _ = Y.shape
        return Y.reshape(S_, T_, -1, 3)[:, :, mocap_idx, :].reshape(S_, T_, K*3)

    Y_train_c = slice_common(Y_train)
    Y_val_c   = slice_common(Y_val)
    Y_test_c  = slice_common(Y_test)

    # ---- Standardize inputs/labels (like your PyTorch code) ----
    sx = StandardScaler()
    sy = StandardScaler()

    Sh, T, Dx = X_train.shape
    Sd, _, Dy = Y_train_c.shape

    X_train = sx.fit_transform(X_train.reshape(-1, Dx)).reshape(Sh, T, Dx)
    Y_train_c = sy.fit_transform(Y_train_c.reshape(-1, Dy)).reshape(Sd, T, Dy)

    X_val  = sx.transform(X_val.reshape(-1, Dx)).reshape(X_val.shape[0], T, Dx)
    Y_val_c= sy.transform(Y_val_c.reshape(-1, Dy)).reshape(Y_val_c.shape[0], T, Dy)

    X_test = sx.transform(X_test.reshape(-1, Dx)).reshape(X_test.shape[0], T, Dx)
    Y_test_c=sy.transform(Y_test_c.reshape(-1, Dy)).reshape(Y_test_c.shape[0], T, Dy)

    # ---- Subject metadata arrays (height/mass per sequence) ----
    meta_cache = SubjectMetaCache(metadata_dir=SUBJECT_YAML_DIR, default_mass=70.0, default_height=1.75)

    def subjects_to_mass_height(subjects):
        mass = np.zeros((len(subjects), 1), dtype=np.float32)
        height = np.zeros((len(subjects), 1), dtype=np.float32)
        for i, s in enumerate(subjects):
            m = meta_cache.get_subject_metadata(s)
            mass[i, 0]   = m["mass"]
            height[i, 0] = m["height"]
        return height, mass

    hgt_train, mass_train = subjects_to_mass_height(subj_train)
    hgt_val,   mass_val   = subjects_to_mass_height(subj_val)
    hgt_test,  mass_test  = subjects_to_mass_height(subj_test)

    print(f"Example subjects: {subj_train[:5]}")
    print(f"Heights (first 5): {hgt_train[:5,0]}")
    print(f"Masses  (first 5): {mass_train[:5,0]}")

    # ---- Load augmenter submodels + mean/std ----
    lower_dir = os.path.join(AUGMENTER_ROOT, AUGMENTER_MODELNAME, SUBMODEL_LOWER)
    upper_dir = os.path.join(AUGMENTER_ROOT, AUGMENTER_MODELNAME, SUBMODEL_UPPER)
    mdl_lower = load_keras_submodel(lower_dir)
    mdl_upper = load_keras_submodel(upper_dir)
    lower_mean, lower_std = load_mean_std(lower_dir)
    upper_mean, upper_std = load_mean_std(upper_dir)

    # ---- Build Corrector + Augmenter layer ----
    corrector = build_corrector(J*3, hidden=HIDDEN, layers_n=LAYERS, dropout=DROPOUT, output_dim=J*3)

    aug_layer = AugmenterLayer(
        lower_model=mdl_lower, upper_model=mdl_upper,
        lower_idx=LOWER_IDX, upper_idx=UPPER_IDX,
        lower_mean=lower_mean, lower_std=lower_std,
        upper_mean=upper_mean, upper_std=upper_std,
        common_indices=aug_idx,             # <-- use AUGMENTER indices here
        midhip_idx=MIDHIP_IDX
    )

    # ---- Assemble full model: HPE -> Corrector -> Augmenter -> Dense(common) ----
    inp_hpe    = keras.Input(shape=(SEQ_LEN, J*3), name="hpe_in")
    inp_height = keras.Input(shape=(1,),          name="height_in")
    inp_mass   = keras.Input(shape=(1,),          name="mass_in")

    jcp_corr   = corrector(inp_hpe)
    dense_pred = aug_layer([jcp_corr, inp_height, inp_mass])  # [B,T,K*3]

    model = keras.Model([inp_hpe, inp_height, inp_mass], dense_pred, name="cascade_tf")
    model.compile(optimizer=keras.optimizers.Adam(LR), loss="mse")

    model.summary()

    # ---- Create callback to save best corrector ----
    pathlib.Path(OUT_DIR).mkdir(exist_ok=True, parents=True)
    
    best_corrector_path = os.path.join(OUT_DIR, "corrector.keras")
    
    # Custom callback to save only the corrector when val_loss improves
    class SaveBestCorrectorCallback(keras.callbacks.Callback):
        def __init__(self, corrector_model, save_path):
            super().__init__()
            self.corrector_model = corrector_model
            self.save_path = save_path
            self.best_val_loss = float('inf')
        
        def on_epoch_end(self, epoch, logs=None):
            current_val_loss = logs.get('val_loss')
            if current_val_loss is not None:
                if current_val_loss < self.best_val_loss:
                    self.best_val_loss = current_val_loss
                    self.corrector_model.save(self.save_path)
                    print(f"\n💾 Epoch {epoch+1}: val_loss improved to {current_val_loss:.6f}, saving corrector to {self.save_path}")
                else:
                    print(f"\n   Epoch {epoch+1}: val_loss={current_val_loss:.6f} (no improvement from {self.best_val_loss:.6f})")
    
    save_best_callback = SaveBestCorrectorCallback(corrector, best_corrector_path)

    # ---- Train ----
    history = model.fit(
        x={"hpe_in": X_train, "height_in": hgt_train, "mass_in": mass_train},
        y=Y_train_c,
        validation_data=(
            {"hpe_in": X_val, "height_in": hgt_val, "mass_in": mass_val},
            Y_val_c
        ),
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE, 
        verbose=2,
        callbacks=[save_best_callback]  # Add callback here
    )

    # ---- Load best corrector for evaluation ----
    print(f"\n📂 Loading best corrector from: {best_corrector_path}")
    best_corrector = keras.models.load_model(best_corrector_path)
    
    # Rebuild model with best corrector for evaluation
    inp_hpe_eval = keras.Input(shape=(SEQ_LEN, J*3), name="hpe_in")
    inp_height_eval = keras.Input(shape=(1,), name="height_in")
    inp_mass_eval = keras.Input(shape=(1,), name="mass_in")
    
    jcp_corr_eval = best_corrector(inp_hpe_eval)
    dense_pred_eval = aug_layer([jcp_corr_eval, inp_height_eval, inp_mass_eval])
    
    eval_model = keras.Model([inp_hpe_eval, inp_height_eval, inp_mass_eval], dense_pred_eval)

    # ---- Evaluate MPJPE (mm) in physical space ----
    print("\n🧪 Evaluating best model on test set...")
    Yp = eval_model.predict({"hpe_in": X_test, "height_in": hgt_test, "mass_in": mass_test}, verbose=2)

    # inverse-transform
    Yp_phys  = sy.inverse_transform(Yp.reshape(-1, K*3)).reshape(Yp.shape).reshape(-1, K, 3)
    Ygt_phys = sy.inverse_transform(Y_test_c.reshape(-1, K*3)).reshape(Y_test_c.shape).reshape(-1, K, 3)

    err = np.linalg.norm(Yp_phys - Ygt_phys, axis=-1)   # [N*T, K]
    overall_mpjpe_mm = float(err.mean() * 1000.0)
    print(f"\nOverall MPJPE: {overall_mpjpe_mm:.2f} mm over {K} common markers")

    # ---- Save other artifacts ----
    # Note: corrector.keras was already saved by callback
    
    # Save scalers
    joblib.dump(sx, os.path.join(OUT_DIR, "scaler_hpe.pkl"))
    joblib.dump(sy, os.path.join(OUT_DIR, "scaler_dense_common.pkl"))

    # Save marker mapping info
    with open(os.path.join(OUT_DIR, "common_markers.json"), "w") as f:
        json.dump({
            "common_markers": common_names,
            "aug_indices": aug_idx,        # augmenter output indices
            "mocap_indices": mocap_idx     # ground truth indices
        }, f, indent=2)

    # Save all config needed for reconstruction
    with open(os.path.join(OUT_DIR, "training_config.json"), "w") as f:
        json.dump({
            "seq_len": SEQ_LEN, 
            "stride": STRIDE, 
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS, 
            "lr": LR, 
            "hidden": HIDDEN, 
            "layers": LAYERS, 
            "dropout": DROPOUT,
            "midhip_idx": MIDHIP_IDX,
            "lower_idx": LOWER_IDX,
            "upper_idx": UPPER_IDX,
            "n_joints": J,
            "n_common_markers": K,
            # Augmenter paths (unchanged, just reference)
            "augmenter_root": AUGMENTER_ROOT,
            "augmenter_model": AUGMENTER_MODELNAME,
            "submodel_lower": SUBMODEL_LOWER,
            "submodel_upper": SUBMODEL_UPPER,
            # Save best val_loss achieved
            "best_val_loss": float(save_best_callback.best_val_loss),
            "final_mpjpe_mm": overall_mpjpe_mm
        }, f, indent=2)

    print("\n✅ Training complete. Artifacts saved in:", OUT_DIR)
    print(f"   - Best corrector model: corrector.keras (val_loss={save_best_callback.best_val_loss:.6f})")
    print(f"   - Scalers: scaler_hpe.pkl, scaler_dense_common.pkl")
    print(f"   - Config: training_config.json, common_markers.json")
    print(f"   - Test MPJPE: {overall_mpjpe_mm:.2f} mm")

if __name__ == "__main__":
    main()
