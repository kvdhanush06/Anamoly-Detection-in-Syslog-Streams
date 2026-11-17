import os
import time
import pickle
import platform
import subprocess
import numpy as np
import pandas as pd
from gensim.models import Word2Vec

from deep_learning import AutoencoderModel
from feature_utils import get_mean_w2v, get_mean_max_w2v

def count_ae_parameters(model):
    params = 0
    if hasattr(model.clf, 'coefs_'):
        for coef, intercept in zip(model.clf.coefs_, model.clf.intercepts_):
            params += coef.size + intercept.size
    if hasattr(model, 'mean_') and model.mean_ is not None:
        params += model.mean_.size
    return params

def get_system_hardware():
    try:
        cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
    except Exception:
        cpu = platform.processor()
    try:
        mem_bytes = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode().strip())
        mem_gb = mem_bytes / (1024**3)
    except Exception:
        mem_gb = 0.0
    return {
        "os": platform.platform(),
        "cpu": cpu,
        "arch": platform.machine(),
        "ram_gb": f"{mem_gb:.1f} GB"
    }

def benchmark_efficiency(ae_model_path="syslog_v2_clean/saved_models/bgl_ae_clean_fold_1.pkl", w2v_model_path="syslog_v2_clean/saved_models/bgl_w2v_clean_fold_1.pkl", input_dim=64, hidden_sizes=(32, 16, 32)):
    print("=" * 80)
    print("EFFICIENCY & LATENCY BENCHMARK")
    hw = get_system_hardware()
    print(f"Hardware: {hw['cpu']} ({hw['arch']}) | RAM: {hw['ram_gb']} | OS: {hw['os']}")
    print("=" * 80)
    
    if ae_model_path and os.path.exists(ae_model_path):
        with open(ae_model_path, "rb") as f:
            ae = pickle.load(f)
        size_kb = os.path.getsize(ae_model_path) / 1024.0
    else:
        ae = AutoencoderModel(hidden_layer_sizes=hidden_sizes, alpha=1e-4, random_state=42)
        X_dummy = np.random.randn(500, input_dim).astype(np.float32)
        ae.fit(X_dummy)
        temp_path = "temp_ae_bench.pkl"
        with open(temp_path, "wb") as f:
            pickle.dump(ae, f)
        size_kb = os.path.getsize(temp_path) / 1024.0
        if os.path.exists(temp_path): os.remove(temp_path)
        
    num_params = count_ae_parameters(ae)
    
    # Latency benchmarking
    X_single = np.random.randn(1, input_dim).astype(np.float32)
    X_batch_256 = np.random.randn(256, input_dim).astype(np.float32)
    X_throughput_10k = np.random.randn(10000, input_dim).astype(np.float32)
    
    # Warmup
    for _ in range(50):
        _ = ae.predict_errors(X_single)
        _ = ae.predict_errors(X_batch_256)
        
    # 1. Single-sample latency (mean ± p99 ms)
    n_single = 1000
    single_times = []
    for _ in range(n_single):
        t0 = time.perf_counter()
        _ = ae.predict_errors(X_single)
        single_times.append((time.perf_counter() - t0) * 1000.0)
    single_times = np.array(single_times)
    single_mean_ms = float(np.mean(single_times))
    single_p99_ms = float(np.percentile(single_times, 99))
    
    # 2. Batch latency for batch=256 (mean ± p99 ms)
    n_batch = 500
    batch_times = []
    for _ in range(n_batch):
        t0 = time.perf_counter()
        _ = ae.predict_errors(X_batch_256)
        batch_times.append((time.perf_counter() - t0) * 1000.0)
    batch_times = np.array(batch_times)
    batch_mean_ms = float(np.mean(batch_times))
    batch_p99_ms = float(np.percentile(batch_times, 99))
    
    # 3. End-to-end throughput test (token lookup + pooling + L2 norm + AE scoring)
    sample_seqs = [["3aa50e45", "7b21a8f0", "150b1306", "3aa50e45"] * 5 for _ in range(10000)]
    if w2v_model_path and os.path.exists(w2v_model_path):
        w2v = Word2Vec.load(w2v_model_path)
    else:
        w2v = Word2Vec(sample_seqs, vector_size=input_dim, min_count=1, workers=1, sg=1, seed=42)
        
    t0 = time.perf_counter()
    X_feats = get_mean_max_w2v(sample_seqs, w2v) if input_dim == 64 else get_mean_w2v(sample_seqs, w2v)
    _ = ae.predict_errors(X_feats)
    e2e_total_s = time.perf_counter() - t0
    e2e_throughput_wps = 10000.0 / e2e_total_s
    
    print(f"| Model Parameters:          {num_params:>10d}")
    print(f"| Serialized Model Size:     {size_kb:>10.2f} KB")
    print(f"| Single Window Latency:     {single_mean_ms:>8.4f} ms (p99: {single_p99_ms:6.4f} ms)")
    print(f"| Batch (256) Latency:       {batch_mean_ms:>8.4f} ms (p99: {batch_p99_ms:6.4f} ms)")
    print(f"| Per-window in Batch 256:   {batch_mean_ms/256*1000:>8.2f} µs/window")
    print(f"| End-to-End Throughput:     {e2e_throughput_wps:>10.1f} windows/second")
    print("=" * 80)
    
    bench_data = {
        "hardware_cpu": hw["cpu"],
        "hardware_arch": hw["arch"],
        "hardware_ram": hw["ram_gb"],
        "hardware_os": hw["os"],
        "model_parameters": num_params,
        "saved_model_kb": size_kb,
        "single_sample_mean_ms": single_mean_ms,
        "single_sample_p99_ms": single_p99_ms,
        "batch_256_mean_ms": batch_mean_ms,
        "batch_256_p99_ms": batch_p99_ms,
        "per_window_batch_latency_us": (batch_mean_ms / 256.0) * 1000.0,
        "e2e_throughput_windows_per_sec": e2e_throughput_wps
    }
    pd.DataFrame([bench_data]).to_csv("syslog_v2_clean/efficiency_results.csv", index=False)
    return bench_data

if __name__ == "__main__":
    benchmark_efficiency()
