from __future__ import annotations

import csv
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np

from .features import AA, KYTE_DOOLITTLE, charge_at_ph


@dataclass
class ScreeningConfig:
    seed_length: int = 7
    alphabet: str = AA
    modes: tuple[str, ...] = ("mirror", "repeat")
    min_seed_charge: float = 2.5
    max_seed_charge: float = 3.5
    seed_charge_prescreen: bool = False
    min_charge: float = 3.0
    max_charge: float = 6.0
    min_hydrophobicity: float = -1.0
    max_hydrophobicity: float = 1.0
    min_motif_count: int = 4
    motif_count_mode: str = "nonoverlap"
    min_seed_motif_count: int = 3
    seed_prescreen: bool = False
    max_seeds: int | None = None
    max_generated_candidates: int | None = None
    generation_progress_every: int = 1_000_000
    generation_workers: int = 1
    seed_chunk_size: int = 1_000_000


def hydrophobicity_index(sequence: str) -> float:
    values = [KYTE_DOOLITTLE[aa] for aa in sequence]
    return float(-np.mean(values))


@lru_cache(maxsize=128)
def motif_sets_by_length(motifs: tuple[str, ...]) -> tuple[tuple[int, frozenset[str]], ...]:
    lengths = sorted({len(motif) for motif in motifs}, reverse=True)
    return tuple((length, frozenset(motif for motif in motifs if len(motif) == length)) for length in lengths)


def count_motifs(sequence: str, motifs: Iterable[str], count_mode: str = "overlap") -> int:
    if count_mode not in {"overlap", "nonoverlap"}:
        raise ValueError(f"Unsupported motif count mode: {count_mode}")
    clean_motifs = tuple(motif for motif in motifs if motif)

    if count_mode == "nonoverlap":
        grouped_motifs = motif_sets_by_length(clean_motifs)
        total = 0
        start = 0
        while start < len(sequence):
            matched_length = 0
            for motif_length, motif_set in grouped_motifs:
                if sequence[start : start + motif_length] in motif_set:
                    matched_length = motif_length
                    break
            if matched_length:
                total += 1
                start += matched_length
            else:
                start += 1
        return total

    total = 0
    for motif in clean_motifs:
        start = 0
        while True:
            idx = sequence.find(motif, start)
            if idx < 0:
                break
            total += 1
            start = idx + 1
    return total


def seed_charge_filter_enabled(config: ScreeningConfig) -> bool:
    return config.seed_charge_prescreen or config.seed_prescreen


def evaluate_seed_prescreen(
    config: ScreeningConfig,
    seed: str,
    seed_motifs: Sequence[str],
) -> tuple[bool, float | None, int | None]:
    seed_charge = None
    seed_motif_hits = None

    if seed_charge_filter_enabled(config):
        seed_charge = charge_at_ph(seed)
        if not (config.min_seed_charge <= seed_charge <= config.max_seed_charge):
            return False, seed_charge, seed_motif_hits

    if seed_motifs:
        seed_motif_hits = count_motifs(seed, seed_motifs)
        if seed_motif_hits < config.min_seed_motif_count:
            return False, seed_charge, seed_motif_hits

    return True, seed_charge, seed_motif_hits


def normalize_alphabet(alphabet: str) -> str:
    clean_alphabet = "".join(dict.fromkeys(alphabet.strip().upper()))
    if not clean_alphabet:
        raise ValueError("Alphabet must not be empty")
    invalid = sorted(set(clean_alphabet) - set(AA))
    if invalid:
        raise ValueError(f"Alphabet contains non-canonical amino acids: {invalid}")
    return clean_alphabet


def seed_space_size(alphabet: str, seed_length: int) -> int:
    return len(normalize_alphabet(alphabet)) ** seed_length


def seed_digits_from_index(index: int, base: int, seed_length: int) -> list[int]:
    if index < 0:
        raise ValueError("Seed index must be non-negative")
    digits = [0] * seed_length
    value = index
    for pos in range(seed_length - 1, -1, -1):
        value, remainder = divmod(value, base)
        digits[pos] = remainder
    if value:
        raise ValueError(f"Seed index {index} exceeds search space for length {seed_length}")
    return digits


def increment_seed_digits(digits: list[int], base: int) -> None:
    for pos in range(len(digits) - 1, -1, -1):
        digits[pos] += 1
        if digits[pos] < base:
            return
        digits[pos] = 0


def iter_seed_strings_range(alphabet: str, seed_length: int, start: int, end: int) -> Iterator[tuple[int, str]]:
    if end < start:
        raise ValueError("Seed range end must be greater than or equal to start")
    base = len(alphabet)
    digits = seed_digits_from_index(start, base, seed_length)
    for index in range(start, end):
        yield index, "".join(alphabet[digit] for digit in digits)
        increment_seed_digits(digits, base)


def iter_seed_strings(alphabet: str, seed_length: int, max_seeds: int | None) -> Iterator[tuple[int, str]]:
    clean_alphabet = normalize_alphabet(alphabet)
    for index, letters in enumerate(product(clean_alphabet, repeat=seed_length)):
        if max_seeds is not None and index >= max_seeds:
            break
        yield index, "".join(letters)


def iter_seed_ranges(total_seeds: int, chunk_size: int) -> Iterator[tuple[int, int]]:
    if chunk_size <= 0:
        raise ValueError("seed_chunk_size must be positive")
    for start in range(0, total_seeds, chunk_size):
        yield start, min(start + chunk_size, total_seeds)


def expand_seed(seed: str, modes: Sequence[str]) -> Iterator[tuple[str, str]]:
    for mode in modes:
        if mode == "mirror":
            yield mode, seed + seed[::-1]
        elif mode == "repeat":
            yield mode, seed + seed
        else:
            raise ValueError(f"Unsupported generation mode: {mode}")


def _remove_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def _prescreen_seed_range(args: tuple[ScreeningConfig, tuple[str, ...], tuple[str, ...], int, int]) -> dict:
    config, motifs, seed_motifs, start, end = args
    clean_alphabet = normalize_alphabet(config.alphabet)
    rows: list[dict] = []
    seen: set[str] = set()
    seeds_scanned = 0
    seeds_passed = 0
    generated = 0
    duplicate_skipped = 0

    for seed_index, seed in iter_seed_strings_range(clean_alphabet, config.seed_length, start, end):
        seeds_scanned += 1
        seed_passed, seed_charge, seed_motif_hits = evaluate_seed_prescreen(config, seed, seed_motifs)
        if not seed_passed:
            continue
        seeds_passed += 1
        for mode, sequence in expand_seed(seed, config.modes):
            generated += 1
            if sequence in seen:
                duplicate_skipped += 1
                continue
            seen.add(sequence)
            charge = charge_at_ph(sequence)
            hydrophobicity = hydrophobicity_index(sequence)
            motif_hits = count_motifs(sequence, motifs, config.motif_count_mode)
            if (
                config.min_charge <= charge <= config.max_charge
                and config.min_hydrophobicity <= hydrophobicity <= config.max_hydrophobicity
                and motif_hits >= config.min_motif_count
            ):
                if seed_charge is None:
                    seed_charge = charge_at_ph(seed)
                if seed_motifs and seed_motif_hits is None:
                    seed_motif_hits = count_motifs(seed, seed_motifs)
                rows.append(
                    {
                        "sequence": sequence,
                        "seed": seed,
                        "mode": mode,
                        "seed_index": seed_index,
                        "seed_charge_ph7": seed_charge,
                        "seed_motif_count": seed_motif_hits,
                        "charge_ph7": charge,
                        "hydrophobicity": hydrophobicity,
                        "motif_count": motif_hits,
                    }
                )

    return {
        "start": start,
        "end": end,
        "seeds_scanned": seeds_scanned,
        "seeds_passed": seeds_passed,
        "generated_mode_sequences": generated,
        "duplicate_skipped": duplicate_skipped,
        "rows": rows,
    }


def _generation_summary(
    config: ScreeningConfig,
    output_csv: Path,
    seeds_scanned: int,
    seeds_passed: int,
    generated: int,
    accepted: int,
    duplicate_skipped: int,
    stop_reason: str,
    seed_motifs: Sequence[str],
) -> dict:
    seed_charge_enabled = seed_charge_filter_enabled(config)
    seed_motif_enabled = bool(seed_motifs)
    return {
        "output_csv": str(output_csv),
        "seeds_scanned": seeds_scanned,
        "seeds_passed": seeds_passed,
        "generated_mode_sequences": generated,
        "accepted_candidates": accepted,
        "duplicate_skipped": duplicate_skipped,
        "stop_reason": stop_reason,
        "seed_filters": {
            "enabled": seed_charge_enabled or seed_motif_enabled,
            "legacy_seed_prescreen": config.seed_prescreen,
            "charge_enabled": seed_charge_enabled,
            "seed_charge_ph7": [config.min_seed_charge, config.max_seed_charge] if seed_charge_enabled else None,
            "motif_enabled": seed_motif_enabled,
            "motif_count": len(seed_motifs),
            "min_seed_motif_count": config.min_seed_motif_count if seed_motif_enabled else None,
        },
        "candidate_filters": {
            "charge_ph7": [config.min_charge, config.max_charge],
            "hydrophobicity": [config.min_hydrophobicity, config.max_hydrophobicity],
            "min_motif_count": config.min_motif_count,
            "motif_count_mode": config.motif_count_mode,
        },
        "generation_workers": config.generation_workers,
        "seed_chunk_size": config.seed_chunk_size,
    }


def _generate_prescreened_candidates_sequential(
    config: ScreeningConfig,
    output_csv: Path,
    motifs: Sequence[str],
    seed_motifs: Sequence[str],
) -> dict:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    _remove_if_exists(output_csv)

    fields = [
        "candidate_id",
        "sequence",
        "seed",
        "mode",
        "seed_index",
        "seed_charge_ph7",
        "seed_motif_count",
        "charge_ph7",
        "hydrophobicity",
        "motif_count",
    ]
    seen: set[str] = set()
    seeds_scanned = 0
    seeds_passed = 0
    generated = 0
    accepted = 0
    duplicate_skipped = 0
    stop_reason = "exhausted_search_space"

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for seed_index, seed in iter_seed_strings(config.alphabet, config.seed_length, config.max_seeds):
            seeds_scanned += 1
            seed_passed, seed_charge, seed_motif_hits = evaluate_seed_prescreen(config, seed, seed_motifs)
            if not seed_passed:
                continue
            seeds_passed += 1
            for mode, sequence in expand_seed(seed, config.modes):
                generated += 1
                if sequence in seen:
                    duplicate_skipped += 1
                    continue
                charge = charge_at_ph(sequence)
                hydrophobicity = hydrophobicity_index(sequence)
                motif_hits = count_motifs(sequence, motifs, config.motif_count_mode)
                if (
                    config.min_charge <= charge <= config.max_charge
                    and config.min_hydrophobicity <= hydrophobicity <= config.max_hydrophobicity
                    and motif_hits >= config.min_motif_count
                ):
                    if seed_charge is None:
                        seed_charge = charge_at_ph(seed)
                    if seed_motifs and seed_motif_hits is None:
                        seed_motif_hits = count_motifs(seed, seed_motifs)
                    seen.add(sequence)
                    writer.writerow(
                        {
                            "candidate_id": f"cand_{accepted:08d}",
                            "sequence": sequence,
                            "seed": seed,
                            "mode": mode,
                            "seed_index": seed_index,
                            "seed_charge_ph7": seed_charge,
                            "seed_motif_count": seed_motif_hits,
                            "charge_ph7": charge,
                            "hydrophobicity": hydrophobicity,
                            "motif_count": motif_hits,
                        }
                    )
                    accepted += 1
                    if config.max_generated_candidates is not None and accepted >= config.max_generated_candidates:
                        stop_reason = "max_generated_candidates_reached"
                        break
            if stop_reason == "max_generated_candidates_reached":
                break
            if (
                config.generation_progress_every > 0
                and seeds_scanned > 0
                and seeds_scanned % config.generation_progress_every == 0
            ):
                print(f"Scanned {seeds_scanned:,} seeds; {seeds_passed:,} seeds expanded; {accepted:,} candidates accepted")
        else:
            if config.max_seeds is not None:
                stop_reason = "max_seeds_reached"

    return _generation_summary(
        config,
        output_csv,
        seeds_scanned,
        seeds_passed,
        generated,
        accepted,
        duplicate_skipped,
        stop_reason,
        seed_motifs,
    )


def _generate_prescreened_candidates_parallel(
    config: ScreeningConfig,
    output_csv: Path,
    motifs: Sequence[str],
    seed_motifs: Sequence[str],
) -> dict:
    if config.max_generated_candidates is not None:
        raise ValueError("--max-generated-candidates is only supported when --generation-workers is 1")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    _remove_if_exists(output_csv)

    fields = [
        "candidate_id",
        "sequence",
        "seed",
        "mode",
        "seed_index",
        "seed_charge_ph7",
        "seed_motif_count",
        "charge_ph7",
        "hydrophobicity",
        "motif_count",
    ]
    total_seeds = seed_space_size(config.alphabet, config.seed_length)
    if config.max_seeds is not None:
        total_seeds = min(total_seeds, config.max_seeds)

    jobs = (
        (config, tuple(motifs), tuple(seed_motifs), start, end)
        for start, end in iter_seed_ranges(total_seeds, config.seed_chunk_size)
    )
    seen: set[str] = set()
    seeds_scanned = 0
    seeds_passed = 0
    generated = 0
    accepted = 0
    duplicate_skipped = 0
    next_progress = config.generation_progress_every

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        with ProcessPoolExecutor(max_workers=config.generation_workers) as executor:
            for chunk in executor.map(_prescreen_seed_range, jobs, chunksize=1):
                seeds_scanned += chunk["seeds_scanned"]
                seeds_passed += chunk["seeds_passed"]
                generated += chunk["generated_mode_sequences"]
                duplicate_skipped += chunk["duplicate_skipped"]

                for row in chunk["rows"]:
                    sequence = row["sequence"]
                    if sequence in seen:
                        duplicate_skipped += 1
                        continue
                    seen.add(sequence)
                    writer.writerow({"candidate_id": f"cand_{accepted:08d}", **row})
                    accepted += 1

                if config.generation_progress_every > 0 and seeds_scanned >= next_progress:
                    print(
                        f"Scanned {seeds_scanned:,}/{total_seeds:,} seeds; "
                        f"{seeds_passed:,} seeds expanded; {accepted:,} candidates accepted"
                    )
                    while next_progress <= seeds_scanned:
                        next_progress += config.generation_progress_every

    stop_reason = "max_seeds_reached" if config.max_seeds is not None else "exhausted_search_space"
    return _generation_summary(
        config,
        output_csv,
        seeds_scanned,
        seeds_passed,
        generated,
        accepted,
        duplicate_skipped,
        stop_reason,
        seed_motifs,
    )


def generate_prescreened_candidates(
    config: ScreeningConfig,
    output_csv: Path,
    motifs: Sequence[str],
    seed_motifs: Sequence[str],
) -> dict:
    if config.generation_workers < 1:
        raise ValueError("--generation-workers must be at least 1")
    if config.generation_workers == 1:
        return _generate_prescreened_candidates_sequential(config, output_csv, motifs, seed_motifs)
    return _generate_prescreened_candidates_parallel(config, output_csv, motifs, seed_motifs)


def pair_sampling_mode(n_items: int, requested_pairs: int, unique_sample_limit: int) -> tuple[str, int]:
    if n_items < 2 or requested_pairs <= 0:
        return "none", 0
    possible = n_items * (n_items - 1) // 2
    target = min(requested_pairs, possible)
    if possible <= requested_pairs:
        return "all_pairs", possible
    if target <= unique_sample_limit:
        return "random_unique", target
    return "random_with_replacement", target


def iter_all_pair_batches(n_items: int, batch_size: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    left_batch: list[int] = []
    right_batch: list[int] = []
    for left in range(n_items - 1):
        for right in range(left + 1, n_items):
            left_batch.append(left)
            right_batch.append(right)
            if len(left_batch) >= batch_size:
                yield np.asarray(left_batch, dtype=np.int32), np.asarray(right_batch, dtype=np.int32)
                left_batch = []
                right_batch = []
    if left_batch:
        yield np.asarray(left_batch, dtype=np.int32), np.asarray(right_batch, dtype=np.int32)


def iter_random_unique_pair_batches(
    n_items: int,
    n_pairs: int,
    batch_size: int,
    seed: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    rng = random.Random(seed)
    seen: set[tuple[int, int]] = set()
    while len(seen) < n_pairs:
        left = rng.randrange(n_items)
        right = rng.randrange(n_items)
        if left == right:
            continue
        if left > right:
            left, right = right, left
        seen.add((left, right))
    pairs = sorted(seen)
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        left_values, right_values = zip(*batch)
        yield np.asarray(left_values, dtype=np.int32), np.asarray(right_values, dtype=np.int32)


def iter_random_replacement_pair_batches(
    n_items: int,
    n_pairs: int,
    batch_size: int,
    seed: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    remaining = n_pairs
    while remaining > 0:
        size = min(batch_size, remaining)
        left = rng.integers(0, n_items, size=size, dtype=np.int32)
        right = rng.integers(0, n_items, size=size, dtype=np.int32)
        invalid = left == right
        while np.any(invalid):
            right[invalid] = rng.integers(0, n_items, size=int(np.sum(invalid)), dtype=np.int32)
            invalid = left == right
        swap = left > right
        if np.any(swap):
            left_swap = left[swap].copy()
            left[swap] = right[swap]
            right[swap] = left_swap
        yield left, right
        remaining -= size


def iter_pair_batches(
    n_items: int,
    requested_pairs: int,
    batch_size: int,
    seed: int,
    unique_sample_limit: int,
) -> tuple[str, int, Iterator[tuple[np.ndarray, np.ndarray]]]:
    mode, target = pair_sampling_mode(n_items, requested_pairs, unique_sample_limit)
    if mode == "none":
        return mode, target, iter(())
    if mode == "all_pairs":
        return mode, target, iter_all_pair_batches(n_items, batch_size)
    if mode == "random_unique":
        return mode, target, iter_random_unique_pair_batches(n_items, target, batch_size, seed)
    return mode, target, iter_random_replacement_pair_batches(n_items, target, batch_size, seed)
