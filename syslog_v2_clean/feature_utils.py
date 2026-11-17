import numpy as np
from gensim.models import Word2Vec

def train_w2v(train_sequences, vector_size=32, window=5, min_count=1, workers=1, sg=1, seed=42):
    """
    Trains Word2Vec strictly on training sequences.
    Using workers=1 and a fixed seed guarantees exact reproducibility.
    """
    model = Word2Vec(
        sentences=train_sequences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg,
        seed=seed
    )
    return model

def compute_oov_stats(sequences, w2v_model):
    """
    Computes fraction of tokens that are Out-Of-Vocabulary (OOV).
    """
    wv = w2v_model.wv
    total_tokens = 0
    oov_tokens = 0
    for s in sequences:
        total_tokens += len(s)
        for w in s:
            if w not in wv:
                oov_tokens += 1
    oov_rate = (oov_tokens / total_tokens) if total_tokens > 0 else 0.0
    return total_tokens, oov_tokens, oov_rate

def l2_normalize(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms

def get_mean_w2v(sequences, w2v_model):
    """
    V0 & V1: Mean-pooled Word2Vec embeddings with L2-normalization (32-d).
    OOV Rule: OOV tokens are skipped. If all tokens are OOV or sequence empty,
    returns a zero vector.
    """
    dim = w2v_model.vector_size
    wv = w2v_model.wv
    X = np.zeros((len(sequences), dim), dtype=np.float32)
    for i, s in enumerate(sequences):
        if not s:
            continue
        vecs = [wv[w] for w in s if w in wv]
        if vecs:
            X[i] = np.mean(vecs, axis=0)
    return l2_normalize(X)

def get_mean_max_w2v(sequences, w2v_model):
    """
    V2: Concatenated Mean + Max-pooled Word2Vec embeddings with L2-normalization (64-d).
    OOV Rule: OOV tokens are skipped. If all tokens are OOV or sequence empty,
    returns a zero vector.
    """
    dim = w2v_model.vector_size
    wv = w2v_model.wv
    X = np.zeros((len(sequences), dim * 2), dtype=np.float32)
    for i, s in enumerate(sequences):
        if not s:
            continue
        vecs = [wv[w] for w in s if w in wv]
        if vecs:
            mean_vec = np.mean(vecs, axis=0)
            max_vec = np.max(vecs, axis=0)
            X[i] = np.concatenate([mean_vec, max_vec])
    return l2_normalize(X)

def compute_idf_weights(train_sequences):
    """
    Computes IDF weights strictly on training sequences.
    idf(w) = log((N + 1) / (df(w) + 1)) + 1.0
    """
    N = len(train_sequences)
    df = {}
    for s in train_sequences:
        for w in set(s):
            df[w] = df.get(w, 0) + 1
            
    idf = {}
    for w, count in df.items():
        idf[w] = float(np.log((N + 1.0) / (count + 1.0)) + 1.0)
    return idf

def get_idf_w2v(sequences, w2v_model, idf_dict):
    """
    V3: IDF-weighted mean pooled Word2Vec embeddings with L2-normalization (32-d).
    OOV Rule: OOV tokens are skipped. If all tokens are OOV or sequence empty,
    returns a zero vector.
    """
    dim = w2v_model.vector_size
    wv = w2v_model.wv
    X = np.zeros((len(sequences), dim), dtype=np.float32)
    default_idf = 1.0
    
    for i, s in enumerate(sequences):
        if not s:
            continue
        valid = [(wv[w], idf_dict.get(w, default_idf)) for w in s if w in wv]
        if valid:
            vecs, weights = zip(*valid)
            weights = np.array(weights, dtype=np.float32)
            total_weight = np.sum(weights)
            if total_weight > 0:
                X[i] = np.sum(np.array(vecs) * weights[:, None], axis=0) / total_weight
            else:
                X[i] = np.mean(vecs, axis=0)
    return l2_normalize(X)

