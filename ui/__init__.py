"""ui: the Streamlit presentation layer.

shell         page chrome: sidebar settings, system status, title
media_tab     one ingest -> analyse -> render tab, driven by a spec (image / video / audio)
adapters      per-file orchestration adapters over the package pipelines (process_single_*)
batch_views   summary table and file picker for multi-file runs
results/      result pages and their sections
stages        check-table helpers; validators: file validation and URL checks
"""
