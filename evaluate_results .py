import os
import numpy as np
import h5py
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split
from tqdm.keras import TqdmCallback

# ---------------------------
# 1. Configuration
# ---------------------------
DATA_PATH = r"D:\FINAL_GP\GOLD_XYZ_OSC.0001_1024.hdf5"
CHECKPOINT_DIR = r"D:\FINAL_GP\checkpoints_v2"
PLOT_DIR = r"D:\FINAL_GP\plots_v2"

AWGN_STD = 0.708
BATCH_SIZE = 2048 
EPOCHS = 32       

# ---------------------------
# 2. GPU Setup
# ---------------------------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU Detected: {gpus}")
    except RuntimeError as e:
        print(e)
else:
    print("⚠️ No GPU detected.")

# ---------------------------
# 3. Data Generator
# ---------------------------
class HDF5Generator(Sequence):
    def __init__(self, file_path, indices, batch_size=BATCH_SIZE, shuffle=True, add_noise=True):
        super().__init__()
        self.file_path = file_path
        self.indices = indices
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.add_noise = add_noise
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.indices)/self.batch_size))

    def __getitem__(self, idx):
        start = idx*self.batch_size
        end = min((idx+1)*self.batch_size, len(self.indices))
        batch_indices = self.indices[start:end]
        
        X_signals = np.zeros((len(batch_indices), 1024, 2), dtype='float32')
        with h5py.File(self.file_path, 'r') as f:
            Xh = f['X']
            # Sorting indices improves HDD read speed
            sorted_idx = np.sort(batch_indices)
            for i, real_idx in enumerate(sorted_idx):
                X_signals[i] = Xh[real_idx]
            
        y_signals = np.ones(len(batch_indices), dtype='float32')

        if self.add_noise:
            X_noise = np.random.normal(0, AWGN_STD, X_signals.shape).astype('float32')
            y_noise = np.zeros(len(batch_indices), dtype='float32')
            X_batch = np.concatenate([X_signals, X_noise], axis=0)
            y_batch = np.concatenate([y_signals, y_noise], axis=0)
            p = np.random.permutation(len(X_batch))
            return X_batch[p], y_batch[p]
        else:
            return X_signals, y_signals

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

# ---------------------------
# 4. GAP Model 
# ---------------------------
def create_gap_model(input_shape=(1024, 2)):
    inp = layers.Input(shape=input_shape)
    x = layers.Conv1D(32, 3, padding='valid', activation='relu')(inp)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(64, 3, padding='valid', activation='relu')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.GlobalAveragePooling1D()(x) # GAP Layer
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    return models.Model(inp, out, name="GAP_Model")

# ---------------------------
# 5. Plotting Functions
# ---------------------------
def plot_results(history, model, file_path, test_indices, snr_vector):
    os.makedirs(PLOT_DIR, exist_ok=True)

    # 1. Accuracy & Loss Curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'Training_History.png'))
    plt.show()

    # 2. SNR Performance Curve
    print("\n--- Generating SNR Plot ---")
    plot_snrs = np.arange(-20, 32, 2)
    snr_accuracies = []
    test_snrs = snr_vector[test_indices]
    
    for snr in plot_snrs:
        subset_mask = (test_snrs == snr)
        if not np.any(subset_mask):
            snr_accuracies.append(0)
            continue
        specific_indices = test_indices[subset_mask]
        gen = HDF5Generator(file_path, specific_indices, batch_size=2048, shuffle=False, add_noise=True)
        scores = model.evaluate(gen, verbose=0)
        snr_accuracies.append(scores[1])
        print(f"SNR {snr}dB: Accuracy = {scores[1]:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(plot_snrs, snr_accuracies, marker='o', linewidth=2, color='blue')
    plt.title('Detection Performance vs SNR')
    plt.xlabel('SNR (dB)'); plt.ylabel('Probability of Detection')
    plt.grid(True); plt.xticks(np.arange(-20, 32, 4))
    plt.savefig(os.path.join(PLOT_DIR, 'Final_SNR_Curve.png'))
    plt.show()

# ---------------------------
# 6. Main Execution
# ---------------------------
def run_final_experiment():
    print("--- 1. Reading TRUE Metadata (Z) ---")
    with h5py.File(DATA_PATH, 'r') as f:
        # Read the true SNR column (column 0 of Z)
        snr_vector = f['Z'][:, 0] 
        total_samples = snr_vector.shape[0]

    print(f"Total samples found: {total_samples}")
    
   # --- filtering: SNR >= -8 dB ---
    mask = (snr_vector >= -8)
    all_indices = np.arange(total_samples)
    filtered_indices = all_indices[mask]
    
    print(f"Training on clean samples (>= -8dB): {len(filtered_indices)}")
    
    # Split
    idx_train, idx_temp = train_test_split(filtered_indices, test_size=0.2, random_state=42, shuffle=True)
    idx_val, idx_test = train_test_split(idx_temp, test_size=0.5, random_state=42, shuffle=True)

    # Generators
    train_gen = HDF5Generator(DATA_PATH, idx_train)
    val_gen   = HDF5Generator(DATA_PATH, idx_val, shuffle=False)
    
    print("\n--- 2. Building Model (GAP + Adam) ---")
    model = create_gap_model()
    opt = optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    ## --- save model at every epoch ---
    cp = ModelCheckpoint(
        filepath=os.path.join(CHECKPOINT_DIR, "GAP_epoch_{epoch:02d}.h5"), 
        save_best_only=False, 
        save_weights_only=False,
        monitor='val_loss'
    )
    
    print("\n--- 3. Starting Training ---")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=[cp, ReduceLROnPlateau(factor=0.5, patience=3), TqdmCallback(verbose=1)],
        workers=1, use_multiprocessing=False
    )
    
    print("\n--- 4. Evaluating & Plotting ---")
    # Use the latest model weights
    plot_results(history, model, DATA_PATH, idx_test, snr_vector)

if __name__ == "__main__":
    run_final_experiment()
