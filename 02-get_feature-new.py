import pandas as pd
import numpy as np
import time
import datetime
from features import get_features
import joblib
import math
import xgboost as xgb
from tqdm import tqdm

# Start timing the entire process
start_time_total = time.time()
print(f"Script started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Load sequence data
load_start = time.time()
with open("positive_charge_combinations_7_ext_0303.txt", "r") as f:
    ext_seqs = f.read().splitlines()

print(f"Total sequences: {len(ext_seqs):,}")
print(f"Data loading time: {time.time() - load_start:.2f} seconds")

# Load models
model_load_start = time.time()
clf = joblib.load("outputs/models/00-model_20250220_191804/model.pkl")
rank = joblib.load("outputs/models/01-model_20250221_132045/model.pkl")
scaler = joblib.load("outputs/models/01-model_20250221_132045/scaler.pkl")
reg = joblib.load("outputs/models/02-model_20250109_155323/model.pkl")
print(f"Model loading time: {time.time() - model_load_start:.2f} seconds")

# Generate features
feature_start = time.time()
ext_feats = get_features(ext_seqs, with_alphafold=False, save_embeds=False)
print(f"Feature generation time: {time.time() - feature_start:.2f} seconds")
print(f"Feature shape: {ext_feats.shape}")

# Predict classifications
classify_start = time.time()
ext_label = clf.predict(ext_feats)
print(f"Classification prediction time: {time.time() - classify_start:.2f} seconds")

# View classification results with detailed stats
label_counts = pd.Series(ext_label).value_counts()
print("Classification results:")
for label, count in label_counts.items():
    print(f"  Label {label}: {count:,} sequences ({count/len(ext_label)*100:.2f}%)")

class Seq:
    def __init__(self, seq, index):
        self.seq = seq
        self.index = index
        self.score = 0

# Process positive sequences
sample = None  # Set to a number if you want to limit, or None for all
pos_seq_start = time.time()
ext_embds_pos = ext_feats[ext_label == 1][:sample].fillna(0).to_numpy()
ext_seq_pos = [ext_seqs[i] for i in range(len(ext_seqs)) if ext_label[i] == 1][:sample]
ext_seq_pos = [Seq(seq, index) for index, seq in enumerate(ext_seq_pos)]
print(f"Total positive sequences: {len(ext_seq_pos):,}")
print(f"Positive sequence processing time: {time.time() - pos_seq_start:.2f} seconds")

# Save positive sequences
save_start = time.time()
with open("ext_seq_pos-14-0222.txt", "w") as f:
    for seq in ext_seq_pos:
        f.write(seq.seq + "\n")
print(f"Saving positive sequences time: {time.time() - save_start:.2f} seconds")

# Prepare for pair combinations
pairs_prep_start = time.time()
indices = np.triu_indices(len(ext_embds_pos), k=1)
combs = np.vstack(indices).T

max_pairs = int(math.comb(len(ext_embds_pos), 2))
total_pairs = min(max_pairs, 10000000)
print(f"Total pairs to process: {total_pairs:,}, Max possible pairs: {max_pairs:,}")
print(f"Pair preparation time: {time.time() - pairs_prep_start:.2f} seconds")

# Clean data
clean_start = time.time()
for i in range(len(ext_embds_pos)):
    ext_embds_pos[i] = np.where(np.isinf(ext_embds_pos[i]), 0, ext_embds_pos[i])
    ext_embds_pos[i] = np.where(np.isnan(ext_embds_pos[i]), 0, ext_embds_pos[i])
print(f"Data cleaning time: {time.time() - clean_start:.2f} seconds")

# Batch processing parameters
batch_size = 10000

# Construct features
def get_pair_features(fea1, fea2):
    concat_fea = np.concatenate([fea1, fea2])
    diff_fea = fea1 - fea2
    ratio_fea = fea1 / (fea2 + 1e-6)  # Avoid division by zero
    return np.concatenate([concat_fea, diff_fea, ratio_fea])

# Process in batches with timing for each batch
pairs_start = time.time()
batch_times = []
tie_counts = []

for start in tqdm(range(0, total_pairs, batch_size)):
    batch_start = time.time()
    end = min(start + batch_size, total_pairs)
    batch_pairs_index = combs[start:end]

    # Get pair features
    feature_start = time.time()
    ext_pair = np.array([get_pair_features(ext_embds_pos[a], ext_embds_pos[b]) for a, b in batch_pairs_index])
    feature_time = time.time() - feature_start

    # Scale features
    scale_start = time.time()
    ext_pair_scaled = scaler.transform(ext_pair)
    scale_time = time.time() - scale_start

    # Predict
    predict_start = time.time()
    dtest = xgb.DMatrix(ext_pair_scaled)
    results = rank.predict(dtest)
    predict_time = time.time() - predict_start

    # Process results
    process_start = time.time()
    ties = 0
    for (i, j), pred in zip(batch_pairs_index, results):
        if pred == 1:  # j wins
            ext_seq_pos[i].score -= 1
            ext_seq_pos[j].score += 1
        elif pred == 2:  # i wins
            ext_seq_pos[i].score += 1
            ext_seq_pos[j].score -= 1
        else:  # tie
            ties += 1

    process_time = time.time() - process_start
    batch_time = time.time() - batch_start
    batch_times.append(batch_time)
    tie_counts.append(ties)

    # Print detailed statistics for this batch
    print(f"Batch {start//batch_size + 1}: {start}-{end-1} ({end-start} pairs)")
    print(f"  Feature time: {feature_time:.2f}s, Scale time: {scale_time:.2f}s, Predict time: {predict_time:.2f}s, Process time: {process_time:.2f}s")
    print(f"  Total batch time: {batch_time:.2f}s, Ties: {ties} ({ties/(end-start)*100:.2f}%)")

# Print overall pair processing statistics
print("\nPair processing statistics:")
print(f"Total pair processing time: {time.time() - pairs_start:.2f} seconds")
print(f"Average batch processing time: {np.mean(batch_times):.2f} seconds")
print(f"Total ties: {sum(tie_counts):,} ({sum(tie_counts)/total_pairs*100:.2f}%)")

# Sort and select top sequences
sort_start = time.time()
ext_seq_pos_sort = sorted(ext_seq_pos, key=lambda x: -x.score)
ext_seq_pos_top = ext_seq_pos_sort[:1000]
print(f"Sorting time: {time.time() - sort_start:.2f} seconds")

# Print top sequence information
print("\nTop sequence information:")
print(f"Top1 Sequence: {ext_seq_pos_top[0].seq}")
print(f"Top1 Score: {ext_seq_pos_top[0].score}")

# Print score distribution statistics
scores = [seq.score for seq in ext_seq_pos]
print("\nScore statistics:")
print(f"Min score: {min(scores)}")
print(f"Max score: {max(scores)}")
print(f"Mean score: {np.mean(scores):.2f}")
print(f"Median score: {np.median(scores)}")
print(f"Standard deviation: {np.std(scores):.2f}")

# Print score distribution for top sequences
top_scores = [seq.score for seq in ext_seq_pos_top]
print("\nTop 1000 score statistics:")
print(f"Min score: {min(top_scores)}")
print(f"Max score: {max(top_scores)}")
print(f"Mean score: {np.mean(top_scores):.2f}")
print(f"Median score: {np.median(top_scores)}")
print(f"Standard deviation: {np.std(top_scores):.2f}")

# Print score distribution for top 10, 100, 500 sequences
for n in [10, 100, 500]:
    if n <= len(ext_seq_pos_top):
        print(f"\nTop {n} score statistics:")
        top_n_scores = [seq.score for seq in ext_seq_pos_top[:n]]
        print(f"Min score: {min(top_n_scores)}")
        print(f"Max score: {max(top_n_scores)}")
        print(f"Mean score: {np.mean(top_n_scores):.2f}")

# Print detailed top sequence information
print("\nTop 100 sequences:")
for i, seq in enumerate(ext_seq_pos_top[:100]):
    print(f"Rank {i+1}: score = {seq.score}, sequence = {seq.seq}")

# Save results
save_start = time.time()
seqs = [seq.seq for seq in ext_seq_pos_top]
scores = [seq.score for seq in ext_seq_pos_top]
results = pd.DataFrame({"Seqs": seqs, "score": scores})
results.to_csv("outputs/results_ranked-0303.csv", index=False)
print(f"Results saving time: {time.time() - save_start:.2f} seconds")

# Total execution time
total_time = time.time() - start_time_total
print(f"\nTotal execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
print(f"Script completed at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

feature = get_features(pd.DataFrame({"Seqs": [seq.seq for seq in ext_seq_pos_top]}), with_alphafold=True)
ext_mic = reg.predict(feature)

seqs = [seq.seq for seq in ext_seq_pos_top]
scores = [seq.score for seq in ext_seq_pos_top]
results = pd.DataFrame({"Seqs": seqs, "MIC": ext_mic, "score": scores})
results = results.sort_values("MIC")
results.to_csv("outputs/results_sorted-0303.csv", index=False)





