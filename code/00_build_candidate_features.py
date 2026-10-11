#!/usr/bin/env python3
"""Generate the candidate feature matrix locally from sequences and public ProtBERT weights."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from common import package_root, read_json, sha256_file, write_json
from tmhf_repro.features import AA_SET, calculate_sequence_features

MODEL_ID = "Rostlab/prot_bert"
MODEL_REVISION = "7a894481acdc12202f0a415dd567f6cfdb698908"
WEIGHTS_SHA256 = "6ea3edd26cfefc3111176100ee2a027ed510f9c86b63e5b53a2a050b59f2af9d"
MODEL_FILES = ["config.json", "pytorch_model.bin", "special_tokens_map.json", "tokenizer_config.json", "vocab.txt"]


def assemble_features(sequences: list[str], embeddings: np.ndarray, columns: list[str]) -> np.ndarray:
    embedding_columns = [name for name in columns if name.startswith("proteinbert__emb_")]
    physical_columns = [name for name in columns if name not in embedding_columns]
    expected_embeddings = [f"proteinbert__emb_{i:03d}" for i in range(768)]
    if embedding_columns != expected_embeddings or len(physical_columns) != 61 or len(set(columns)) != 829:
        raise ValueError("Expected the published 61 + 768 feature schema")
    if embeddings.shape != (len(sequences), 768):
        raise ValueError("Embedding rows/dimensions do not match the sequence batch")
    physical = calculate_sequence_features(sequences)
    missing = set(physical_columns) - set(physical.columns)
    if missing:
        raise ValueError(f"Missing calculated features: {sorted(missing)}")
    frame = pd.concat([physical[physical_columns].reset_index(drop=True),
                       pd.DataFrame(embeddings, columns=embedding_columns)], axis=1)
    matrix = frame[columns].to_numpy(dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("Non-finite candidate features")
    return matrix


def load_encoder(model_path: Path | None, device: str):
    import torch
    from transformers import BertModel, BertTokenizer
    if model_path is None:
        from huggingface_hub import snapshot_download
        model_path = Path(snapshot_download(MODEL_ID, revision=MODEL_REVISION, allow_patterns=MODEL_FILES))
    if sha256_file(model_path / "pytorch_model.bin") != WEIGHTS_SHA256:
        raise ValueError("ProtBERT weights differ from the pinned public checkpoint")
    tokenizer = BertTokenizer.from_pretrained(str(model_path), do_lower_case=False)
    model = BertModel.from_pretrained(str(model_path), use_safetensors=False,
                                      attn_implementation="eager").float().to(device).eval()
    if model.config.hidden_size != 1024:
        raise ValueError("Expected the 1024-dimensional ProtBERT encoder")
    return tokenizer, model


def encode(sequences: list[str], tokenizer, model, device: str) -> np.ndarray:
    import torch
    inputs = tokenizer([" ".join(sequence) for sequence in sequences], return_tensors="pt",
                       add_special_tokens=True, padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        hidden = model(**inputs).last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        # Match the historical candidate cache: include CLS/SEP; exclude padding;
        # mean pool 1024 outputs and retain the first 768 (not a learned projection).
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
    return pooled[:, :768].float().cpu().numpy()


def build(input_path: Path, output: Path, columns_path: Path, *, model_path: Path | None,
          device: str, batch_size: int, full_pool: bool) -> dict:
    if batch_size <= 0:
        raise ValueError("Batch size must be positive")
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(f"Output exists; choose a new path or remove it explicitly: {output}")
    if output.suffix != ".npy":
        raise ValueError("Output must have a .npy extension")
    source_hash = sha256_file(input_path)
    frame = pd.read_csv(input_path, usecols=["candidate_id", "sequence"])
    sequences = frame["sequence"].astype(str).tolist()
    if not sequences or not frame["candidate_id"].is_unique:
        raise ValueError("Expected nonempty, uniquely identified candidates")
    if any(len(s) != 14 or not set(s).issubset(AA_SET) for s in sequences):
        raise ValueError("Candidates must be uppercase 14-mers using the 20 standard amino acids")
    if full_pool:
        expected = read_json(package_root() / "data/prepare_manifest.json")["expected_prescreen"]
        if len(frame) != expected["rows"] or source_hash != expected["sha256"]:
            raise ValueError("Full-pool sequences do not match the published prescreen checksum/order")
        if frame["candidate_id"].tolist() != [f"cand_{i:08d}" for i in range(len(frame))]:
            raise ValueError("Candidate IDs must map directly to matrix row indices")
    columns = read_json(columns_path)["feature_columns"]
    # Validate the complete schema before downloading/loading model weights.
    assemble_features(sequences[:1], np.zeros((1, 768), dtype=np.float32), columns)
    tokenizer, model = load_encoder(model_path, device)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=output.name + ".", suffix=".partial", dir=output.parent)
    os.close(fd)
    temporary = Path(temporary)
    try:
        matrix = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32,
                                          shape=(len(sequences), len(columns)))
        for start in range(0, len(sequences), batch_size):
            batch = sequences[start:start + batch_size]
            matrix[start:start + len(batch)] = assemble_features(batch, encode(batch, tokenizer, model, device), columns)
            if start == 0 or (start // batch_size) % 100 == 0 or start + len(batch) == len(sequences):
                matrix.flush()
                print(f"Computed {start + len(batch):,}/{len(sequences):,} rows", flush=True)
        matrix.flush()
        del matrix
        import torch
        import transformers
        record = {"complete": True, "full_pool": full_pool, "rows": len(sequences), "columns": len(columns),
                  "dtype": "float32", "input_sha256": source_hash,
                  "feature_columns_sha256": sha256_file(columns_path),
                  "matrix_sha256": sha256_file(temporary), "model_id": MODEL_ID,
                  "model_revision": MODEL_REVISION, "weights_sha256": WEIGHTS_SHA256,
                  "pooling": "attention-masked mean including CLS/SEP; first 768 of 1024 outputs",
                  "device": device, "batch_size": batch_size,
                  "versions": {"torch": torch.__version__, "transformers": transformers.__version__}}
        temporary.replace(output)
        write_json(output.with_suffix(".json"), record)
        return record
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Optional 14-mer candidate CSV; requires --output for a subset run")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model-path", type=Path, help="Optional local copy of the pinned public ProtBERT checkpoint")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda:0, ...")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--generation-workers", type=int, default=16)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.generation_workers <= 0:
        parser.error("Batch size and worker count must be positive")
    if args.input is not None and args.output is None:
        parser.error("Use --output with --input to keep subset matrices separate from the full pool")
    try:
        import torch
        import transformers  # Check optional dependencies before expensive enumeration.
    except ImportError as exc:
        raise SystemExit("Install requirements-features.txt before generating candidate features") from exc
    root = package_root()
    output = args.output or root / "data/prepared/candidate_feature_matrix.npy"
    if output.exists() or output.with_suffix(".json").exists():
        parser.error(f"Output already exists: {output}")
    full_pool = args.input is None
    if full_pool:
        spec = importlib.util.spec_from_file_location("screening_entry", root / "code/03_screen.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        (root / "outputs/screening").mkdir(parents=True, exist_ok=True)
        input_path, result = module.enumerate_candidates(root, args.generation_workers)
        if not result["pass"]:
            raise SystemExit("Prescreen generation failed its checksum validation")
    else:
        input_path = args.input
    device = ("cuda:0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    print(json.dumps(build(input_path, output, root / "data/feature_columns.json", model_path=args.model_path,
                           device=device, batch_size=args.batch_size, full_pool=full_pool), indent=2))


if __name__ == "__main__":
    main()
