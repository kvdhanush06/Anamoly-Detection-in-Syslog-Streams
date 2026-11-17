import os
import sys
import copy
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import KFold
import pickle

# -------------------------------------------------------------------------
# STANDALONE HDFS IMPROVEMENT SCRIPT (PCA + POT)
# 
# INSTRUCTIONS FOR RUNNING ON A NEW LAPTOP:
# 1. Ensure you have the required libraries installed (numpy, scikit-learn).
# 2. Make sure 'features.npz' is located in the exact same folder as this script.
#    (If the data file is somewhere else, change the PATH_TO_DATA variable below)
# 3. Just run: python run_hdfs_pot.py
# -------------------------------------------------------------------------

# Import from the same directory since we copied the files here
from pca_rpca import PCAModel
from thresholding import compute_threshold_pot

# Update this path if you move the features.npz file somewhere else
PATH_TO_DATA = "features.npz"

# We save the models locally in a subfolder so it doesn't clutter
os.makedirs("saved_models", exist_ok=True)

def l2_normalize(X):
    # This is the undocumented trick that makes PCA work perfectly on HDFS!
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms

def run_hdfs_pot_improvement():
    print(f"\n{'='*80}")
    print(f"HDFS PCA [POT Threshold Improvement] - Standalone Run")
    print(f"{'='*80}")
    
    # 1. Load the data
    if not os.path.exists(PATH_TO_DATA):
        print(f"ERROR: Cannot find '{PATH_TO_DATA}'!")
        print("Please ensure the preprocessed HDFS features file is in this folder.")
        sys.exit(1)
        
    print(f"Loading data from {PATH_TO_DATA}...")
    data = np.load(PATH_TO_DATA)
    X = l2_normalize(data["X_counts"].astype(np.float32))
    y = data["y"]
    
    # 2. Identify normal vs abnormal logs
    normal_idx = np.where(y == 0)[0]
    abnormal_idx = np.where(y == 1)[0]
    
    # 3. Sample 25,000 normal and 1,000 abnormal logs (as per the base paper's protocol)
    rng = np.random.default_rng(42)
    sampled_normal_idx = rng.choice(normal_idx, 25000, replace=False)
    sampled_abnormal_idx = rng.choice(abnormal_idx, 1000, replace=False)
    
    # 4. Set up 5-Fold Cross Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    metrics = {"p": [], "r": [], "f1": []}
    
    model = PCAModel(k=6)
    fold = 1
    
    for fit_n_idx, test_n_idx in kf.split(sampled_normal_idx):
        # 5. Split Train (fit) and Test data
        fit_idx = sampled_normal_idx[test_n_idx]
        test_idx = np.concatenate([sampled_normal_idx[fit_n_idx], sampled_abnormal_idx])
        
        Xf, Xt, y_test = X[fit_idx], X[test_idx], y[test_idx]
        
        model_path = f"saved_models/hdfs_pot_pca_fold_{fold}.pkl"
        
        # 6. Load saved model if it exists, otherwise train a new one
        if os.path.exists(model_path):
            print(f"Fold {fold}: Loading saved model from {model_path}...")
            with open(model_path, "rb") as f:
                fold_model = pickle.load(f)
        else:
            fold_model = copy.deepcopy(model)
            print(f"Fold {fold}: Training model on {len(Xf)} normal logs...")
            fold_model.fit(Xf) # <-- Purely training on normal data (NO LEAKAGE)
            
            with open(model_path, "wb") as f:
                pickle.dump(fold_model, f)
            print(f"  -> Model saved to {model_path}")
        
        # 7. Predict errors
        ef = fold_model.predict_errors(Xf)
        et = fold_model.predict_errors(Xt)
        
        # 8. Apply Dynamic Extreme Value Theory (POT)
        # We calculate the threshold purely from the training errors (ef)
        theta = compute_threshold_pot(ef, q=0.90, target_fpr=1e-3)
        
        # 9. Flag anomalies if test error is greater than threshold
        y_pred = (et > theta).astype(int)
        
        p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
        print(f"Fold {fold}: P: {p*100:.2f}%, R: {r*100:.2f}%, F1: {f1*100:.2f}% (Threshold: {theta:.4f})")
        
        metrics["p"].append(p); metrics["r"].append(r); metrics["f1"].append(f1)
        fold += 1
        
    print(f"\nFINAL HDFS POT RESULT:")
    print(f"Precision: {np.mean(metrics['p'])*100:.2f}%")
    print(f"Recall:    {np.mean(metrics['r'])*100:.2f}%")
    print(f"F1 Score:  {np.mean(metrics['f1'])*100:.2f}%\n")

if __name__ == "__main__":
    run_hdfs_pot_improvement()
