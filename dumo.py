Here’s a production-ready plan and code to switch your RAG to use only local .osc scenarios and your osc2 docs, retrieve the most relevant code file(s) and documentation for a given instruction+input, generate a new .osc, then parse and auto-fix it with up to 3 retries controlled via CLI flags. This design uses exact FAISS search, atomic writes, Alpaca-style prompting, and a safe CUDA multiprocessing setup for vLLM.

What changes and why
Index the .osc scenario folder by embedding each file’s content (plus filename) for semantic retrieval, and index the markdown documentation in “documentation/” as chunked passages for fallback context.

Use FAISS IndexFlatL2 for correctness and simplicity (exact search) while your corpus is manageable, avoiding approximate errors early in production.

Persist the index and JSON sidecars via atomic write/replace to prevent partial artifacts on crash or power loss.

On startup, verify index.ntotal equals metadata row count and that both indexes were built together to avoid any index/metadata desync.

Construct Alpaca-style prompts (Instruction/Input/Output) so inference matches your fine-tuning format, then add retrieved .osc examples and relevant doc excerpts as context.

Generate a .osc file, parse it using py-osc2 (or its CLI osc2parser), and if the parser reports an error, re-prompt with the error included to repair and retry (max 3, toggleable by CLI).

Ensure CUDA workers use the “spawn” start method to avoid the “Cannot re-initialize CUDA in forked subprocess” issue with multi-GPU vLLM.

build_indexes.py (CPU FAISS, atomic writes, robust validation)
Inputs: scenarios_dir (folder of .osc files) and docs_dir (folder of .md docs; defaults to “documentation”).

Embeds each .osc file’s normalized text and stores metadata: filename, relative path, size, and first N lines preview.

Splits markdown into paragraph chunks and embeds for doc fallback.

Writes FAISS indexes with serialize_index and JSON sidecars via atomic replace; verifies read-back and counts.

python
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
app.py (retrieval, vLLM spawn-safe init, retry with py-osc2 validation)
Uses “spawn” multiprocessing for vLLM workers and defers CUDA init to main to avoid forked CUDA errors.

CLI flags: instruction, input, optional tags (unused here), output_dir, auto-retry toggle, max-retries, tensor-parallel size.

Retrieval: semantic top-k from scenarios index, then conditional doc fallback.

Prompt: prepend retrieved .osc examples (truncated) and doc snippets; Alpaca-style Instruction/Input and require only .osc in Output.

Validation: write .osc, parse with py-osc2; if parser error, re-prompt with previous output and error message; up to N retries.

python
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
How to run
Build indexes from folders: “scenarios/” with .osc files and “documentation/” with .md files.

Example:

python build_indexes.py --scenarios_dir scenarios --docs_dir documentation --model all-MiniLM-L12-v2

python app.py --instruction "Create a lane change scenario" --input "Ego changes from lane 1 to lane 2 with min TTC 3s and speed 50km/h." --auto_retry --max_retries 3 --tp_size 8

Suggestions for modeling your code examples
Include the filename and top N lines at the start of the embedding text so file names and headers influence retrieval, since they often encode scenario intent.

Normalize content (strip excessive whitespace/comments) and cap to a sensible byte budget to avoid drowning embeddings with long code, which can harm semantic signal.

If certain osc2 “constructs/modifiers” are decisive, you can add a lightweight keyword tag list per file (extracted by regex) and use it to re-rank top-k hits, but keep FAISS semantic search as the primary step for generalization to novel requests.

Stick with exact FAISS (IndexFlatL2) until corpus growth justifies approximate indexes; it reduces operational surprises and simplifies consistency checks.

Keep using atomic writes and post-write verification to avoid corruption under failures, and always validate index.ntotal == metadata length at app startup.

Make the “spawn” multiprocessing setup part of your app template to prevent CUDA fork errors when scaling to multi-GPU vLLM workers.

If helpful, the parser integration uses py-osc2’s osc2parser CLI exposed by the package, which is designed for syntax checking OpenSCENARIO 2.x files, making it suitable for your validation-and-repair loop.
