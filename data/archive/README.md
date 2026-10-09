# Historical results

These files were produced by the earlier workflow, before removal of designated
sequence augmentation and target-hit acceptance checks. That ranker added 50
copies per available designated sequence with assigned MIC 0.01. The results
must not be reported as outputs of `observed_labels_v1`.

The historical clustering configuration was selected after a sensitivity search;
recorded metadata lists 17,168 configurations, 11,500 random restarts, and one
configuration that hit all designated targets. Its parameters are retained in
the current code, but the old panel membership is no longer an acceptance gate.
This archive preserves that distinction rather than treating a new fit as a new
independent parameter-selection study.

The `reference/` tables allow inspecting prior rankings and exercising the
selection step without training. The input data and historical code are also
available in Git history at commit `a4055c9`.

See the [data inventory](../README.md) for supplementary datasets and
`supplementary_manifest.json` for published-file checksums and source mappings.
