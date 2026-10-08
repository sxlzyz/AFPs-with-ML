from __future__ import annotations

import math
from itertools import combinations
from typing import Iterable

import numpy as np
import pandas as pd


AA = "ACDEFGHIKLMNPQRSTVWY"
AA_SET = set(AA)

KYTE_DOOLITTLE = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

PKA = {
    "N_terminal": 9.0,
    "C_terminal": 2.0,
    "D": 3.9,
    "E": 4.1,
    "C": 8.3,
    "Y": 10.1,
    "H": 6.0,
    "K": 10.5,
    "R": 12.5,
}


def normalize_sequence(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip().upper()


def is_valid_peptide(sequence: str) -> bool:
    return bool(sequence) and set(sequence).issubset(AA_SET)


def charge_at_ph(sequence: str, ph: float = 7.0) -> float:
    charge = 1.0 / (1.0 + 10 ** (ph - PKA["N_terminal"]))
    charge -= 1.0 / (1.0 + 10 ** (PKA["C_terminal"] - ph))
    for aa in sequence:
        if aa in {"D", "E", "C", "Y"}:
            charge -= 1.0 / (1.0 + 10 ** (PKA[aa] - ph))
        elif aa in {"H", "K", "R"}:
            charge += 1.0 / (1.0 + 10 ** (ph - PKA[aa]))
    return float(charge)


def isoelectric_point(sequence: str, tolerance: float = 0.001) -> float:
    low, high = 0.0, 14.0
    for _ in range(64):
        mid = (low + high) / 2.0
        if charge_at_ph(sequence, mid) > 0:
            low = mid
        else:
            high = mid
        if high - low < tolerance:
            break
    return float((low + high) / 2.0)


def hydrophobic_moment(sequence: str, angle_degrees: float = 100.0) -> float:
    angle = math.radians(angle_degrees)
    x = 0.0
    y = 0.0
    for i, aa in enumerate(sequence):
        h = KYTE_DOOLITTLE[aa]
        x += h * math.cos(i * angle)
        y += h * math.sin(i * angle)
    return float(math.sqrt(x * x + y * y) / max(len(sequence), 1))


def alternating_hydrophobic_cationic_count(sequence: str) -> int:
    hydrophobic = set("AVLIPFWMYC")
    cationic = set("KRH")
    groups = []
    for aa in sequence:
        if aa in hydrophobic:
            groups.append("H")
        elif aa in cationic:
            groups.append("C")
        else:
            groups.append("O")
    return sum(1 for a, b in zip(groups, groups[1:]) if {a, b} == {"H", "C"})


def motif_count(sequence: str, motifs: Iterable[str] = ("RLLR", "RVVR", "LLRR", "LRRL", "RRLL")) -> int:
    total = 0
    for motif in motifs:
        start = 0
        while True:
            idx = sequence.find(motif, start)
            if idx < 0:
                break
            total += 1
            start = idx + 1
    return total


def base_physicochemical_features(sequence: str) -> dict[str, float]:
    values = [KYTE_DOOLITTLE[aa] for aa in sequence]
    length = len(sequence)
    kd_mean = float(np.mean(values))
    hydrophobicity = -kd_mean
    charge = charge_at_ph(sequence)
    pi = isoelectric_point(sequence)
    moment = hydrophobic_moment(sequence)
    penetration_depth = kd_mean
    return {
        "length": float(length),
        "hydrophobicity": hydrophobicity,
        "charge_ph7": charge,
        "isoelectric_point": pi,
        "penetration_depth": penetration_depth,
        "amphipathic_index": moment,
        "cationic_fraction": sum(aa in "KRH" for aa in sequence) / length,
        "hydrophobic_fraction": sum(aa in "AVLIPFWMYC" for aa in sequence) / length,
        "alternating_hydrophobic_cationic": float(alternating_hydrophobic_cationic_count(sequence)),
        "high_frequency_motif_count": float(motif_count(sequence)),
    }


def add_nonlinear_features(features: pd.DataFrame, base_columns: list[str]) -> pd.DataFrame:
    result = features.copy()
    eps = 1e-8
    for col in base_columns:
        result[f"{col}__square"] = result[col] * result[col]
    for left, right in combinations(base_columns, 2):
        result[f"{left}__x__{right}"] = result[left] * result[right]
        result[f"{left}__over__{right}"] = result[left] / (result[right].abs() + eps)
        result[f"{right}__over__{left}"] = result[right] / (result[left].abs() + eps)
    return result


def calculate_sequence_features(sequences: Iterable[str]) -> pd.DataFrame:
    rows = [base_physicochemical_features(seq) for seq in sequences]
    features = pd.DataFrame(rows)
    nonlinear_base = [
        "length",
        "hydrophobicity",
        "charge_ph7",
        "isoelectric_point",
        "penetration_depth",
        "amphipathic_index",
    ]
    return add_nonlinear_features(features, nonlinear_base)
