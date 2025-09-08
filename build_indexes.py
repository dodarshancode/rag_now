#!/usr/bin/env python3
import os
import json
import time
import uuid
import argparse
import tempfile
import numpy as np
import faiss
from typing import List, Dict
from sentence_transformers import SentenceTransformer

# Defaults – adjust as needed
DEFAULT_SCENARIOS_DIR = "scenarios"        # folder containing .osc files
DEFAULT_DOCS_DIR = "documentation"         # folder containing .md docs
EMBED_MODEL_PATH = "all-MiniLM-L12-v2"     # local sentence-transformers model

# Output artifacts
SCN_INDEX_PATH = "scenarios_index.faiss"
SCN_META_PATH = "scenarios_metadata.json"
DOC_INDEX_PATH = "docs_index.faiss"
DOC_CHUNKS_PATH = "docs_chunks.json"
MANIFEST_PATH = "build_manifest.json"

def atomic_write_bytes(dst_path: str, data: bytes):
    os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=os.path.dirname(dst_path) or ".", delete=False) as tf:
        tf.write(data)
        tmp = tf.name
    os.replace(tmp, dst_path)  # atomic on POSIX, robust replace on failure [23]

def atomic_write_json(dst_path: str, obj):
    data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    atomic_write_bytes(dst_path, data)  # atomic JSON write to prevent partial files [23]

def read_file(path: str, max_bytes: int = 512_000) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read(max_bytes)  # cap large files for embedding sanity [22]

def collect_scenarios(scenarios_dir: str) -> List[Dict]:
    items = []
    for root, _, files in os.walk(scenarios_dir):
        for fn in files:
            if fn.lower().endswith(".osc"):
                full = os.path.join(root, fn)
                rel = os.path.relpath(full, scenarios_dir)
                txt = read_file(full)
                items.append({
                    "rel_path": rel,
                    "filename": fn,
                    "size": os.path.getsize(full),
                    "content": txt
                })
    if not items:
        raise FileNotFoundError(f"No .osc files found in {scenarios_dir}")
    return items  # robust discovery of code examples [22]

def collect_docs(docs_dir: str) -> List[str]:
    chunks = []
    for root, _, files in os.walk(docs_dir):
        for fn in files:
            if fn.lower().endswith(".md"):
                txt = read_file(os.path.join(root, fn))
                # simple paragraph split
                parts = [p.strip() for p in txt.split("\n\n") if p.strip()]
                chunks.extend(parts)
    return chunks  # fallback knowledge chunks [21]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios_dir", default=DEFAULT_SCENARIOS_DIR, help="Folder with .osc files")  # [22]
    ap.add_argument("--docs_dir", default=DEFAULT_DOCS_DIR, help="Folder with .md documentation")  # [21]
    ap.add_argument("--model", default=EMBED_MODEL_PATH, help="Local sentence-transformers model path")  # [22]
    args = ap.parse_args()

    print("[INFO] Loading embedding model...")
    encoder = SentenceTransformer(args.model)  # offline local model path supported by ST [22]

    print("[INFO] Scanning scenarios...")
    scenarios = collect_scenarios(args.scenarios_dir)
    print(f"[INFO] Found {len(scenarios)} scenarios")

    # Prepare texts and metadata
    texts = []
    metadata = []
    for s in scenarios:
        # Embed filename plus content; filename often holds scenario semantics
        composite = f"Filename: {s['filename']}\nContent:\n{s['content']}"
        texts.append(composite)
        preview = s["content"][:400]
        metadata.append({
            "rel_path": s["rel_path"],
            "filename": s["filename"],
            "size": s["size"],
            "preview": preview
        })

    print("[INFO] Encoding scenarios...")
    scn_emb = encoder.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    if scn_emb.ndim != 2:
        raise RuntimeError(f"Unexpected scenarios embedding shape: {scn_emb.shape}")
    dim = scn_emb.shape[25]
    scn_index = faiss.IndexFlatL2(dim)
    scn_index.add(scn_emb.astype(np.float32))
    assert scn_index.ntotal == len(metadata), "Index size must equal scenarios metadata length"  # [22]

    print("[INFO] Scanning documentation...")
    doc_chunks = collect_docs(args.docs_dir)
    print(f"[INFO] Found {len(doc_chunks)} doc chunks")
    doc_index = faiss.IndexFlatL2(dim)
    if doc_chunks:
        print("[INFO] Encoding documentation...")
        doc_emb = encoder.encode(doc_chunks, show_progress_bar=True, convert_to_numpy=True)
        if doc_emb.ndim != 2 or doc_emb.shape[25] != dim:
            raise RuntimeError("Doc embedding dim mismatch with scenarios")
        doc_index.add(doc_emb.astype(np.float32))

    build_id = f"{int(time.time())}-{uuid.uuid4().hex}"
    manifest = {
        "build_id": build_id,
        "scenarios_count": len(metadata),
        "doc_chunks_count": len(doc_chunks),
        "embedding_model": args.model,
        "faiss_metric": "L2",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    print("[INFO] Writing artifacts atomically...")
    atomic_write_bytes(SCN_INDEX_PATH, faiss.serialize_index(scn_index))  # serialize is a supported path [6][12]
    atomic_write_json(SCN_META_PATH, metadata)
    atomic_write_bytes(DOC_INDEX_PATH, faiss.serialize_index(doc_index))
    atomic_write_json(DOC_CHUNKS_PATH, doc_chunks)
    atomic_write_json(MANIFEST_PATH, manifest)

    print("[INFO] Verifying round-trip...")
    scn_rt = faiss.deserialize_index(open(SCN_INDEX_PATH, "rb").read())
    meta_rt = json.load(open(SCN_META_PATH, "r", encoding="utf-8"))
    doc_rt = faiss.deserialize_index(open(DOC_INDEX_PATH, "rb").read())
    chunks_rt = json.load(open(DOC_CHUNKS_PATH, "r", encoding="utf-8"))

    assert scn_rt.ntotal == len(meta_rt), "Post-write: scenarios index/metadata mismatch"  # [22]
    assert len(chunks_rt) == manifest["doc_chunks_count"], "Post-write: doc count mismatch"  # [22]

    print(f"[INFO] Build complete. Build ID: {build_id} | Scenarios: {scn_rt.ntotal} | Doc chunks: {len(chunks_rt)}")

if __name__ == "__main__":
    main()
