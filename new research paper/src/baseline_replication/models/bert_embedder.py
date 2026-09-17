"""
BERT Contextual Log Embedding Extractor.
Section 4.4 of Alsalmi et al. (CMC 2025/2026):
x_i = BERT(l_i)
Extracts 768-dimensional contextualized semantic embedding for each log message/event.
"""
import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict
import numpy as np
from tqdm import tqdm

class BertEmbedder:
    def __init__(self, model_name: str = "bert-base-uncased", device: str = None, batch_size: int = 64):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        print(f"[BertEmbedder] Loading {model_name} on device: {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.batch_size = batch_size
        self.cache: Dict[str, np.ndarray] = {}

    @torch.no_grad()
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Computes 768-dim mean-pooled contextual embeddings for a list of text messages.
        Uses caching for duplicate log templates/messages.
        """
        uncached_indices = [i for i, t in enumerate(texts) if t not in self.cache]
        uncached_texts = [texts[i] for i in uncached_indices]

        if uncached_texts:
            embeddings_list = []
            for start in range(0, len(uncached_texts), self.batch_size):
                batch_texts = uncached_texts[start:start + self.batch_size]
                # Default empty fallback if string is blank after cleaning
                batch_texts = [t if t.strip() else "[EMPTY]" for t in batch_texts]
                
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=64,
                    return_tensors="pt"
                ).to(self.device)

                outputs = self.model(**inputs)
                # Mean-pooling over token representations taking attention mask into account
                last_hidden_state = outputs.last_hidden_state  # [B, seq_len, 768]
                attention_mask = inputs["attention_mask"].unsqueeze(-1)  # [B, seq_len, 1]
                
                sum_embeddings = torch.sum(last_hidden_state * attention_mask, dim=1)
                sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
                mean_pooled = (sum_embeddings / sum_mask).cpu().numpy()
                embeddings_list.append(mean_pooled)

            uncached_embeddings = np.vstack(embeddings_list)
            for text, emb in zip(uncached_texts, uncached_embeddings):
                self.cache[text] = emb

        return np.array([self.cache[t] for t in texts], dtype=np.float32)
