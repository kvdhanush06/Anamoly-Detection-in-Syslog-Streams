"""
Pipeline to download, parse, and extract L2-normalized feature vectors 
from the BGL (Blue Gene/L) dataset.

This script demonstrates generalizing beyond HDFS by testing a dataset
with fundamentally different system behavior and message interleaving.
"""
import os
import sys
import tarfile
import urllib.request
import pandas as pd
import numpy as np
import warnings
import ssl
from logparser.Drain import LogParser

# Suppress warnings and fix SSL for macOS
warnings.filterwarnings('ignore')
ssl._create_default_https_context = ssl._create_unverified_context

DATA_DIR = "data/BGL"
RAW_FILE = os.path.join(DATA_DIR, "BGL.log")
ARCHIVE_FILE = os.path.join(DATA_DIR, "BGL.tar.gz")
OUTPUT_DIR = os.path.join(DATA_DIR, "preprocessed")
FEATURES_FILE = os.path.join(OUTPUT_DIR, "features.npz")

BGL_URL = "https://zenodo.org/record/3227177/files/BGL.tar.gz"

# Drain parameters for BGL
# BGL logs look like: - 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.50.363779 R02-M1-N0-C:J12-U11 RAS KERNEL INFO instruction cache parity error corrected
log_format = '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>'
st = 0.5  # Similarity threshold
depth = 4 # Max depth of parse tree

def download_and_extract():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(RAW_FILE):
        if not os.path.exists(ARCHIVE_FILE):
            print(f"Downloading BGL dataset from Zenodo... (this is ~733MB)")
            urllib.request.urlretrieve(BGL_URL, ARCHIVE_FILE)
            print("Download complete.")
        
        print("Extracting BGL.tar.gz...")
        with tarfile.open(ARCHIVE_FILE, "r:gz") as tar:
            tar.extractall(path=DATA_DIR)
        print("Extraction complete.")
    else:
        print("BGL.log already exists.")

def parse_logs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    structured_file = os.path.join(OUTPUT_DIR, "BGL.log_structured.csv")
    if not os.path.exists(structured_file):
        print("Parsing BGL logs using Drain... (this may take ~10 minutes for 4.7M logs)")
        # logparser outputs to outdir
        parser = LogParser(log_format, indir=DATA_DIR, outdir=OUTPUT_DIR, depth=depth, st=st)
        parser.parse("BGL.log")
        print("Parsing complete.")
    else:
        print("Parsed logs already exist.")
    return structured_file

def extract_features(structured_file):
    if os.path.exists(FEATURES_FILE):
        print("Features already extracted.")
        return
        
    print("Loading structured logs...")
    # Read the parsed CSV
    df = pd.read_csv(structured_file)
    
    # BGL has native labels per log: '-' means normal, others are anomalous
    df['Label'] = df['Label'].apply(lambda x: 0 if x == '-' else 1)
    
    # Extract timestamp
    # We will group by sliding window or fixed time window. 
    # Loghub commonly uses a 5-minute fixed window for BGL.
    # BGL timestamps are in the Timestamp column (unix seconds)
    print("Grouping logs into 5-minute time windows...")
    window_size_seconds = 5 * 60
    
    df['Window_ID'] = df['Timestamp'] // window_size_seconds
    
    # Create Event Count Matrix
    # We group by Window_ID and count occurrences of each EventId
    print("Building event count matrix...")
    grouped = df.groupby(['Window_ID', 'EventId']).size().unstack(fill_value=0)
    
    # Determine labels for each window (1 if any log in the window is anomalous)
    window_labels = df.groupby('Window_ID')['Label'].max()
    
    # Align counts and labels
    X_counts = grouped.values.astype(np.float32)
    y = window_labels.values
    
    print(f"Generated {X_counts.shape[0]} sequences (windows) with {X_counts.shape[1]} unique events.")
    
    # L2 Normalization (Our breakthrough finding)
    print("Applying L2 Normalization...")
    norms = np.linalg.norm(X_counts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X_l2 = X_counts / norms
    
    # Save features
    np.savez(FEATURES_FILE, X_counts=X_counts, X_l2=X_l2, y=y)
    print(f"Saved features to {FEATURES_FILE}")

if __name__ == "__main__":
    download_and_extract()
    parse_logs()
    extract_features(os.path.join(OUTPUT_DIR, "BGL.log_structured.csv"))
