import itertools
import multiprocessing as mp
import numpy as np
from tqdm import tqdm
import time
import os

top_100_4mer_top5 = ["RLLR", "RVVR", "LLRR", "LRRL", "RRLL"]

kyte_doolittle_index = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}

# Define hydrophobicity calculation function
def get_hydrophobicity(seq):
    return sum(kyte_doolittle_index[aa] for aa in seq) / len(seq) * -1

# Calculate charge for a subsequence
def calculate_charge(subsequence, charge):
    return sum(charge[aa] for aa in subsequence)

# Check combinations for specific criteria
def check_combinations(start, end, window_size, aa, charge, top_100_4mer_top5):
    seqs = []
    num = 0

    base_patterns = ["LR", "VR"]

    def count_patterns(sequence):
        count = 0
        for i in range(len(sequence)-3):
            substr = sequence[i:i+4]
            for pattern in base_patterns:
                if pattern[0] * 2 + pattern[1] * 2 == substr:
                    count += 1
                elif (pattern[0] + pattern[1]) * 2 == substr:
                    count += 1
                elif (pattern[1] + pattern[0]) * 2 == substr:
                    count += 1
                elif pattern[1] * 2 + pattern[0] * 2 == substr:
                    count += 1
                elif pattern in substr and all(c in 'LR' for c in substr):
                    count += 1
                elif 'VR' in substr and all(c in 'VR' for c in substr):
                    count += 1
        return count

    for combination in itertools.product(aa, repeat=window_size):
        num += 1
        if num < start:
            continue
        if num > end:
            break

        combination_str = ''.join(combination)
        copy_seq1 = combination_str + combination_str[::-1]
        copy_seq2 = combination_str + combination_str

        if (3 <= calculate_charge(copy_seq1, charge) <= 7
            and count_patterns(copy_seq1) > 1):
            seqs.append(copy_seq2)

        if (3 <= calculate_charge(copy_seq2, charge) <= 7
            and count_patterns(copy_seq2) > 1):
            seqs.append(copy_seq2)

    return seqs

# Generate positive charge combinations in parallel
def positive_charge_combinations_parallel(window_size=8, processes=4):
    aa = "GAVLIPFYWSTCMNQDEKRH"
    charge = {a: 0 for a in aa}
    charge["D"] = -1
    charge["E"] = -1
    charge["K"] = 1
    charge["R"] = 1
    charge["H"] = 0.1

    total_combinations = len(aa) ** window_size
    print(f"Total combinations: {total_combinations:,}")

    chunk_size = total_combinations // processes
    ranges = [(i * chunk_size + 1, (i + 1) * chunk_size) for i in range(processes)]
    ranges[-1] = (ranges[-1][0], total_combinations)

    pool = mp.Pool(processes)
    results = [pool.apply_async(check_combinations, (start, end, window_size, aa, charge, top_100_4mer_top5)) for start, end in ranges]

    pool.close()
    pool.join()

    return np.concatenate([result.get() for result in results])

start = time.time()

window_size = 7
processes = 16
result = positive_charge_combinations_parallel(window_size=window_size, processes=processes)

print(f"Total sequences: {len(result):,}")

# Save results
output_filename = f"positive_charge_combinations_{window_size}_ext_0303.txt"
with open(output_filename, "w+") as f:
    for seq in result:
        f.write(seq + "\n")

print(f"Results saved to {output_filename}")

# Calculate sequence statistics
seq_lengths = [len(seq) for seq in result]
avg_length = sum(seq_lengths) / len(seq_lengths) if seq_lengths else 0
min_length = min(seq_lengths) if seq_lengths else 0
max_length = max(seq_lengths) if seq_lengths else 0

print(f"Average sequence length: {avg_length:.2f}")
print(f"Minimum sequence length: {min_length}")
print(f"Maximum sequence length: {max_length}")

# Calculate amino acid percentages
aa_counts = {}
for seq in result:
    for aa in seq:
        aa_counts[aa] = aa_counts.get(aa, 0) + 1
total_aa = sum(aa_counts.values())
aa_percentages = {aa: (count/total_aa)*100 for aa, count in aa_counts.items()}

print("Amino acid percentages:")
for aa, percentage in aa_percentages.items():
    print(f"{aa}: {percentage:.2f}%")

# Calculate hydrophobicity statistics
hydrophobicity_values = [get_hydrophobicity(seq) for seq in result]
avg_hydrophobicity = sum(hydrophobicity_values) / len(hydrophobicity_values) if hydrophobicity_values else 0

print(f"Average hydrophobicity: {avg_hydrophobicity:.2f}")

# Calculate charge statistics
aa = "GAVLIPFYWSTCMNQDEKRH"
charge = {a: 0 for a in aa}
charge["D"] = -1
charge["E"] = -1
charge["K"] = 1
charge["R"] = 1
charge["H"] = 0.1
charge_values = [sum(charge.get(aa, 0) for aa in seq) for seq in result]
avg_charge = sum(charge_values) / len(charge_values) if charge_values else 0

print(f"Average charge: {avg_charge:.2f}")
