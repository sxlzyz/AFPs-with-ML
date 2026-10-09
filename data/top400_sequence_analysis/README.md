# Top400 sequence analysis

Composition, global sequence alignments, and physicochemical descriptors.
The candidate table is [../screening/top400_cluster_assignments_final.csv](../screening/top400_cluster_assignments_final.csv).
Positive references are 188 unique MIC <= 2 uM sequences from both original partitions.

- `amino_acid_composition.csv`: pooled residue frequencies for references and candidates.
- `nearest_positive_sequence_identity.csv`: best reference alignment per candidate.
- `all_pairwise_alignments.csv`: all 75,200 candidate/reference alignments.
- `physicochemical_descriptors.csv`: 400 candidates and 188 reference sequences.
- `positive_references.csv`: reference sequences and MIC values.
- `analysis_metadata.json`: alignment settings, definitions, and source hashes.
- `validation.json`: validation results recorded by the original analysis.

Global alignment uses match +1, mismatch -1, linear gap -1 including terminal gaps;
identity includes gap columns in the denominator. Absence of exact matches to this
reference set does not establish novelty against all known peptides.
The `penetration_depth` column is mean Kyte-Doolittle hydrophobicity, not a measured
physical depth. `hydrophobicity` is its negative; hydrophobic moment uses 100 degrees
per residue. Original numeric values and column names are preserved.
