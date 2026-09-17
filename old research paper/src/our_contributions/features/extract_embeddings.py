import os
import pandas as pd
import numpy as np
from gensim.models import Word2Vec

HDFS_TRACES = "data/HDFS_v1/preprocessed/Event_traces.csv"
HDFS_OUTPUT_NPZ = "data/HDFS_v1/preprocessed/features_w2v.npz"

BGL_STRUCTURED = "data/BGL/preprocessed/BGL.log_structured.csv"
BGL_OUTPUT_NPZ = "data/BGL/preprocessed/features_w2v.npz"

def train_and_save_w2v(sequences, y, output_npz, vector_size=32):
    print(f"Training Word2Vec on {len(sequences)} sequences...")
    model = Word2Vec(sentences=sequences, vector_size=vector_size, window=5, min_count=1, workers=4, sg=1)
    
    print("Computing sequence embeddings via mean pooling...")
    X_w2v = np.zeros((len(sequences), model.vector_size), dtype=np.float32)
    
    for i, seq in enumerate(sequences):
        if len(seq) == 0:
            continue
        vecs = [model.wv[token] for token in seq if token in model.wv]
        if vecs:
            X_w2v[i] = np.mean(vecs, axis=0)
            
    print("L2 normalizing sequence embeddings...")
    norms = np.linalg.norm(X_w2v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X_l2 = X_w2v / norms
    
    print(f"Saving to {output_npz}...")
    np.savez(output_npz, X_l2=X_l2, y=y)

def extract_hdfs_embeddings():
    print(f"\n--- Processing HDFS ---")
    df = pd.read_csv(HDFS_TRACES)
    
    sequences = []
    for seq_str in df['Features']:
        seq_str = seq_str.strip('[]').replace(' ', '')
        tokens = seq_str.split(',') if seq_str else []
        sequences.append(tokens)
        
    y = (df['Label'] != 'Success').astype(int).values
    train_and_save_w2v(sequences, y, HDFS_OUTPUT_NPZ)

def extract_bgl_embeddings():
    print(f"\n--- Processing BGL ---")
    if not os.path.exists(BGL_STRUCTURED):
        print(f"{BGL_STRUCTURED} not found. Skipping BGL.")
        return
        
    df = pd.read_csv(BGL_STRUCTURED)
    
    # Group by Node + 6-hour sliding window
    window_size_seconds = 6 * 60 * 60
    df['Window_ID'] = df['Timestamp'] // window_size_seconds
    df['Label_Bin'] = df['Label'].apply(lambda x: 0 if x == '-' else 1)
    
    # Group by Node and Window_ID and collect sequence of EventIds
    print("Grouping logs into sequences by Node and 6-hour windows...")
    grouped = df.groupby(['Node', 'Window_ID'])
    
    sequences = grouped['EventId'].apply(list).values
    y = grouped['Label_Bin'].max().values
    
    train_and_save_w2v(sequences, y, BGL_OUTPUT_NPZ)

if __name__ == "__main__":
    extract_hdfs_embeddings()
    extract_bgl_embeddings()
