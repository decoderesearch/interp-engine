"""Per-engine capture adapters. Each exposes ``capture(hf_id, input_ids, layers, points, saes, device)``
returning ``(arrays, sae_summaries)``. Engines import their own heavy backend lazily so a worker
env only needs its own backend installed."""
