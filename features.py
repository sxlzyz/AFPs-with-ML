import numpy as np

# Define hydropathy index tables
kyte_doolittle_index = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}

hopp_woods_index = {
    'A': -0.5, 'R': 3.0, 'N': 0.2, 'D': 3.0, 'C': -1.0,
    'Q': 0.2, 'E': 3.0, 'G': 0.0, 'H': -0.5, 'I': -1.8,
    'L': -1.8, 'K': 3.0, 'M': -1.3, 'F': -2.5, 'P': 0.0,
    'S': 0.3, 'T': -0.4, 'W': -3.4, 'Y': -2.3, 'V': -1.5
}

# Define pKa values for amino acids and terminals
pKa_values = {
    'N_terminal': 9.0, 'C_terminal': 2.0,
    'D': 3.9, 'E': 4.1, 'C': 8.3, 'Y': 10.1, 'H': 6.0,
    'K': 10.5, 'R': 12.5
}

# Get charge
def get_charge(seq, pH=7.0):
    net_charge = 0
    # N-terminal charge
    net_charge += 1 / (1 + 10**(pH - pKa_values['N_terminal']))
    # C-terminal charge
    net_charge -= 1 / (1 + 10**(pKa_values['C_terminal'] - pH))

    # Side chain charges
    for aa in seq:
        if aa in pKa_values:
            if aa in ['D', 'E', 'C', 'Y']:
                # Acidic amino acids
                net_charge -= 1 / (1 + 10**(pKa_values[aa] - pH))
            elif aa in ['H', 'K', 'R']:
                # Basic amino acids
                net_charge += 1 / (1 + 10**(pH - pKa_values[aa]))

    return net_charge

# Get hydrophobicity
def get_hydrophobicity(seq):
    return sum(kyte_doolittle_index[aa] for aa in seq) / len(seq) * -1

# Get isoelectric point
def get_pI(seq, precision=0.01):
    pH = 0.0
    while pH <= 14.0:
        net_charge = get_charge(seq, pH)
        if abs(net_charge) < precision:
            return pH
        pH += precision
    return None

def get_amphipathic_index(sequence, index_type='Kyte-Doolittle'):
    # Select hydropathy index table
    if index_type == 'Kyte-Doolittle':
        index_table = kyte_doolittle_index
    elif index_type == 'Hopp-Woods':
        index_table = hopp_woods_index
    else:
        raise ValueError("Unsupported index type. Choose 'Kyte-Doolittle' or 'Hopp-Woods'.")

    # Calculate amphipathic index
    total_index = 0
    for amino_acid in sequence:
        if amino_acid in index_table:
            total_index += index_table[amino_acid]
        else:
            raise ValueError(f"Unknown amino acid: {amino_acid}")

    # Calculate average amphipathic index
    average_index = total_index / len(sequence)
    return average_index * -1

def get_penetration_depth(sequence):
    # Calculate hydropathy index for each residue
    hydropathy_values = [kyte_doolittle_index[aa] for aa in sequence]

    # Calculate penetration depth
    penetration_depth = sum(hydropathy_values) / len(sequence)
    return penetration_depth

import os
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

# Load model and tokenizer
def get_embeds(seqs, device="cuda:0", save_embeds=True):
    bert_path = "/home/zwj/workspace/projects/ai4food/port_bert"
    embeddings_path = "/home/zwj/workspace/projects/ai4food/data/embeddings_dict.npy"

    tokenizer = BertTokenizer.from_pretrained(bert_path, do_lower_case=False)
    model = BertModel.from_pretrained(bert_path)

    model.to(device)
    model.eval()

    # Function to get embedding representations
    def get_protbert_embeddings(sequence, tokenizer, model):
        sequence = " ".join(sequence)
        inputs = tokenizer(sequence, return_tensors='pt', add_special_tokens=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}  # Move input tensors to specified device
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.squeeze(0)
        return embeddings.mean(dim=0).cpu().numpy()

    # Initialize list to store embeddings
    if os.path.exists(embeddings_path):
        with open(embeddings_path, "rb") as f:
            embeddings_dict = np.load(f, allow_pickle=True).item()

    else:
        embeddings_dict = {}

    embeddings_list = []
    for sequence in tqdm(seqs):
        if embeddings_dict.get(sequence) is None:
            embeddings = get_protbert_embeddings(sequence, tokenizer, model)
            if save_embeds:
                embeddings_dict[sequence] = embeddings
        else:
            embeddings = embeddings_dict[sequence]

        embeddings_list.append(embeddings)

    with open(embeddings_path, "wb") as f:
        np.save(f, embeddings_dict)

    embds = np.array(embeddings_list)
    return embds

def get_features(data, use_feature="calculated", with_alphafold=False, with_embeds=True, save_embeds=False):
    features = pd.DataFrame()

    if isinstance(data, pd.DataFrame):
        seqs = data["序列"]
        # Save original index for later alignment
        original_index = data.index

        if use_feature == 1 or use_feature == "all":
            features = data[english_names.keys()].copy()

        elif use_feature == 2 or use_feature == "calculable":
            keys = [key for key, func in func_dict.items() if func]
            features = data[keys].copy()

        elif use_feature == 3 or use_feature == "calculated":
            keys = [key for key, func in func_dict.items() if func]
            # Create empty DataFrame with same index
            features = pd.DataFrame(index=original_index)
            for key in keys:
                func = func_dict[key]
                features[key] = data["序列"].apply(func)
    else:
        seqs = data
        # For non-DataFrame input, create index starting from 0
        original_index = pd.RangeIndex(start=0, stop=len(seqs))
        features = pd.DataFrame(index=original_index)

        keys = [key for key, func in func_dict.items() if func]
        for key in keys:
            func = func_dict[key]
            features[key] = [func(seq) for seq in data]

    features_li = []
    # Keep original index
    features_li.append(features)

    # Get vector representations
    if with_embeds:
        embds = get_embeds(seqs, save_embeds=save_embeds)
        # Create DataFrame with original index
        embds = pd.DataFrame(embds, columns=[f'emb_{i}' for i in range(embds.shape[1])], index=original_index)
        features_li.append(embds)

    # Read AlphaFold features
    if with_alphafold:
        alphafold_features = pd.read_csv("alphafold_feature.csv")
        alphafold_features = alphafold_features.set_index("seq")

        # Initialize AlphaFold features DataFrame with original index
        alphafold_features_for_seqs = pd.DataFrame(0.0,
                                                index=original_index,
                                                columns=alphafold_features.columns,
                                                dtype=np.float64)

        # Add AlphaFold features to existing features using sequence as index
        seq_to_index = {seq: idx for idx, seq in zip(original_index, seqs)}
        for seq in alphafold_features.index:
            if seq in seq_to_index:
                idx = seq_to_index[seq]
                alphafold_features_for_seqs.loc[idx] = alphafold_features.loc[seq]
            else:
                # Skip sequences not in current dataset
                continue

        features_li.append(alphafold_features_for_seqs)

    # Merge all features, ensuring same index
    features = pd.concat(features_li, axis=1)
    # Verify all features share same index
    assert len(features) == len(original_index), "Number of features does not match original index length"
    assert features.index.equals(original_index), "Feature index does not match original index"

    print(f"{features.shape=}")
    return features

func_dict = {
    '疏水性': get_hydrophobicity,
    '电荷数': get_charge,
    '等电点': get_pI,
    '穿透深度': get_penetration_depth,  # Requires molecular dynamics simulation
    '两亲性指数': get_amphipathic_index,
}

english_names = {
    '疏水性': 'hydrophobicity',
    '电荷数': 'charge',
    '等电点': 'pI',
    '穿透深度': 'penetration depth',
    '倾斜角度': 'tilt angle',
    '无序构象倾向': 'disorder conformation tendency',
    '线性力矩': 'linear moment',
    '体外聚集倾向': 'extracellular aggregation tendency',
    '疏水性残留物掩盖的角度': 'angle covered by hydrophobic residues',
    '两亲性指数': 'amphipathic index',
    'PPII线圈的倾向': 'PPII loop tendency'
}
