#!/usr/bin/env python3
import os
import sys
import json
import time
import argparse
import subprocess
import numpy as np
import faiss

# Ensure spawn before any CUDA work
import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)  # safe for CUDA/vLLM multiproc [7][16]
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")  # vLLM workers use spawn [7][13]

from sentence_transformers import SentenceTransformer
from vllm import LLM, SamplingParams

# Artifacts
SCN_INDEX_PATH = "scenarios_index.faiss"
SCN_META_PATH = "scenarios_metadata.json"
DOC_INDEX_PATH = "docs_index.faiss"
DOC_CHUNKS_PATH = "docs_chunks.json"
MANIFEST_PATH = "build_manifest.json"

EMBED_MODEL_PATH = "all-MiniLM-L12-v2"  # local ST model
LLM_PATH = "models/codellama-13b-finetuned"  # local weights directory

SIM_THRESHOLD = 1.5
TOPK_SCN = 3
TOPK_DOC = 3

def load_faiss(path: str) -> faiss.Index:
    with open(path, "rb") as f:
        return faiss.deserialize_index(f.read())  # supported FAISS IO [6][18]

def startup_checks():
    for p in [SCN_INDEX_PATH, SCN_META_PATH, DOC_INDEX_PATH, DOC_CHUNKS_PATH, MANIFEST_PATH]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing artifact: {p}")  # hard fail for integrity [22]
    scn_index = load_faiss(SCN_INDEX_PATH)
    metadata = json.load(open(SCN_META_PATH, "r", encoding="utf-8"))
    if scn_index.ntotal != len(metadata):
        raise RuntimeError(f"Index/metadata mismatch: {scn_index.ntotal} != {len(metadata)}")  # [22]
    doc_index = load_faiss(DOC_INDEX_PATH)
    doc_chunks = json.load(open(DOC_CHUNKS_PATH, "r", encoding="utf-8"))
    return scn_index, metadata, doc_index, doc_chunks  # validated artifacts [22]

def embed_texts(encoder, texts):
    arr = encoder.encode(texts, convert_to_numpy=True)
    if arr.ndim != 2:
        raise RuntimeError(f"Unexpected embedding shape: {arr.shape}")  # sanity [22]
    return arr.astype(np.float32)

def retrieve_scenarios(scn_index, metadata, qvec, topk=TOPK_SCN):
    D, I = scn_index.search(qvec, topk)
    hits = []
    for rank, idx in enumerate(I):
        dist = float(D[0, rank])
        if idx < 0 or idx >= len(metadata):
            continue  # guard against padding or mismatch [22]
        if dist >= SIM_THRESHOLD:
            continue
        hits.append((idx, dist))
    return hits  # robust, thresholded hits [22]

def retrieve_docs(doc_index, doc_chunks, qvec, topk=TOPK_DOC):
    if doc_index.ntotal == 0 or len(doc_chunks) == 0:
        return []
    D, I = doc_index.search(qvec, topk)
    out = []
    for r, didx in enumerate(I):
        if didx < 0 or didx >= len(doc_chunks):
            continue
        out.append(doc_chunks[didx])
    return out  # conditional fallback [21]

def build_prompt(scn_examples, docs, instruction, input_text):
    # include retrieved .osc snippets and doc lines
    context = ""
    for ex in scn_examples:
        context += (
            f"\n### Example (.osc)\n"
            f"### Filename:\n{ex['filename']}\n"
            f"### Snippet:\n``````\n"
        )
    if docs:
        context += "\n# Reference (osc2 docs):\n"
        for d in docs:
            context += f"- {d}\n"
    # Alpaca-style with an explicit requirement: output only .osc
    prompt = (
        f"{context}\n"
        f"### Instruction:\n{instruction}\n"
        f"### Input:\n{input_text}\n"
        "### Output:\n"
        "Return only a valid OpenSCENARIO 2.x .osc file content, with no explanations or markdown fences.\n"
    )
    return prompt  # adheres to Alpaca format and constrains output [24]

def write_osc(output_dir, base_name, content):
    os.makedirs(output_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    fn = f"{base_name}-{ts}.osc"
    path = os.path.join(output_dir, fn)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path  # write result artifact for validation [23]

def parse_with_py_osc2(osc_path: str):
    """
    Prefer py-osc2 CLI 'osc2parser' if installed; return (ok: bool, message: str).
    """
    try:
        # CLI advertised by py-osc2 as simple syntax checker [1]
        res = subprocess.run(
            ["osc2parser", osc_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60
        )
        ok = (res.returncode == 0)
        msg = (res.stdout + "\n" + res.stderr).strip()
        return ok, msg
    except FileNotFoundError:
        return False, "osc2parser not found. Ensure py-osc2 is installed and on PATH."  # installation hint [1]
    except Exception as e:
        return False, f"Parser error: {e}"

def generate_once(llm, prompt):
    params = SamplingParams(temperature=0.1, top_p=0.9, max_tokens=4096)
    out = llm.generate([prompt], params)
    return out.outputs.text  # vLLM single-prompt generate [7]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instruction", required=True, help="High-level task (Alpaca Instruction)")  # [24]
    ap.add_argument("--input", required=True, help="Scenario specifics (Alpaca Input)")  # [24]
    ap.add_argument("--output_dir", default="outputs", help="Where to write .osc files")  # [23]
    ap.add_argument("--tp_size", type=int, default=int(os.environ.get("TP_SIZE", "8")), help="vLLM tensor parallel size")  # [7]
    ap.add_argument("--auto_retry", action="store_true", help="Enable auto-repair on parse error (max_retries)")  # [1]
    ap.add_argument("--max_retries", type=int, default=3, help="Max repair attempts if auto_retry")  # [1]
    args = ap.parse_args()

    # Load artifacts and models after spawn setup
    scn_index, metadata, doc_index, doc_chunks = startup_checks()  # validated counts [22]
    encoder = SentenceTransformer(EMBED_MODEL_PATH)  # local embeddings [22]
    llm = LLM(model=LLM_PATH, tensor_parallel_size=args.tp_size)  # vLLM multi-GPU [7]

    # Build query vector
    query = f"Instruction: {args.instruction}\nInput: {args.input}"
    qvec = embed_texts(encoder, [query])

    # Retrieve scenarios
    hits = retrieve_scenarios(scn_index, metadata, qvec, TOPK_SCN)
    scn_examples = []
    for idx, dist in hits:
        m = metadata[idx]
        scn_examples.append({
            "filename": m["filename"],
            "snippet": m["preview"]
        })

    # Retrieve docs if no good examples
    docs = []
    if not scn_examples:
        docs = retrieve_docs(doc_index, doc_chunks, qvec, TOPK_DOC)

    # Initial prompt
    prompt = build_prompt(scn_examples, docs, args.instruction, args.input)

    # Generation + validation loop
    attempt = 0
    base_name = "scenario"
    last_error = ""
    while True:
        attempt += 1
        content = generate_once(llm, prompt)
        osc_path = write_osc(args.output_dir, base_name, content)
        ok, msg = parse_with_py_osc2(osc_path)
        if ok:
            print(f"[INFO] Valid .osc written: {osc_path}")
            break
        print(f"[WARN] Parser failed on attempt {attempt}: {msg}")

        if not args.auto_retry or attempt >= max(1, args.max_retries):
            print("[ERROR] Auto-retry disabled or max retries reached; keeping last file for inspection.")
            break

        # Repair prompt: include last output and parser message to guide fixes
        repair_context = (
            "\n# Previous attempt failed with parser errors.\n"
            "## Parser message:\n"
            f"{msg}\n"
            "## Previous output that must be corrected:\n"
            f"{content}\n"
        )
        prompt = build_prompt(scn_examples, docs, args.instruction, args.input) + repair_context  # iterative refinement [1]

if __name__ == "__main__":
    main()
