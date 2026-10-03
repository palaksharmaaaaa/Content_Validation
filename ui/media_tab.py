"""
ui.media_tab: the single implementation of an ingest -> analyse -> render tab.

The image, video and audio tabs differ only in labels, accepted file types, the detector bundle and the results
renderer, which live in a ``MediaTabSpec``. Everything else (upload/URL ingestion into the session scratch
directory, content-hash cache signatures, batch analysis with progress, summary/table/selector, per-file rendering)
is shared.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List

import streamlit as st

from core.hashing import file_sha256
from ui.batch_ui import (
    render_batch_file_selector,
    render_batch_overview_table,
    render_batch_summary_dashboard,
    run_batch_pipeline,
)
from ui.validators import fetch_media_from_url


@dataclass(frozen=True)
class MediaTabSpec:
    key: str                       # "img" | "vid" | "aud": prefix for widget keys and session-state entries
    modality: str                  # "image" | "video" | "audio"
    noun: str                      # "Image" | "Video" | "Audio"
    subheader: str
    uploader_label: str
    file_types: List[str]
    uploader_help: str
    url_placeholder: str
    url_help: str
    empty_info: str
    render_result: Callable[[Dict[str, Any]], None]
    tag_upload_source: bool = False  # image items record where they came from


def _ingest_uploads(spec: MediaTabSpec, session_dir: Path) -> List[Dict[str, Any]]:
    uploaded = st.file_uploader(
        spec.uploader_label, type=spec.file_types, accept_multiple_files=True,
        key=f"uploader_{spec.key}", help=spec.uploader_help,
    )
    items: List[Dict[str, Any]] = []
    for idx, up in enumerate(uploaded or []):
        clean_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", up.name)
        save_path = str(session_dir / f"{spec.key}_batch_{idx}_{clean_name}")
        with open(save_path, "wb") as f:
            f.write(up.getbuffer())
        item = {"path": save_path, "filename": up.name, "size": up.size}
        if spec.tag_upload_source:
            item["source"] = "Local Device Upload"
        items.append(item)
    return items


def _ingest_urls(spec: MediaTabSpec, session_dir: Path) -> List[Dict[str, Any]]:
    urls_input = st.text_area(
        f"Paste {spec.noun} URL(s) (One per line or comma-separated)",
        placeholder=spec.url_placeholder, key=f"{spec.key}_urls_input", help=spec.url_help,
    )
    items: List[Dict[str, Any]] = []
    if not (st.button(f"Fetch & Analyze {spec.noun}(s)", key=f"btn_fetch_{spec.key}") and urls_input):
        return items
    urls = [u.strip() for u in urls_input.replace(",", "\n").splitlines() if u.strip().startswith("http")]
    if not urls:
        st.error("Please enter at least one valid URL starting with http:// or https://")
        return items
    for idx, url in enumerate(urls):
        with st.spinner(f"Downloading {spec.modality} #{idx+1} from {url[:40]}..."):
            fetched = fetch_media_from_url(url, expected_type=spec.modality, dest_dir=session_dir)
        if not fetched.get("success"):
            st.error(f"❌ Failed to download {url}: {fetched.get('error')}")
            continue
        path = fetched["file_path"]
        item = {"path": path, "filename": fetched["filename"], "size": Path(path).stat().st_size}
        if spec.tag_upload_source:
            item["source"] = f"URL Stream ({url[:35]}...)"
        items.append(item)
    return items


def _analyse(spec: MediaTabSpec, items: List[Dict[str, Any]], detectors: Dict[str, Any], sensitivity_key: str, session_dir: Path) -> List[Dict[str, Any]]:
    """Batch-analyse ``items``, reusing the session's previous results when the same bytes were already analysed."""
    if not items:
        return []
    sig_key, results_key = f"{spec.key}_batch_sig", f"{spec.key}_batch_results"
    signature = tuple((it["filename"], file_sha256(it["path"]), sensitivity_key) for it in items)
    if st.session_state.get(sig_key) == signature:
        return st.session_state.get(results_key, [])

    progress_bar = st.progress(0, text=f"Initializing batch {spec.modality} analysis...")

    def on_progress(curr: int, total: int, name: str) -> None:
        progress_bar.progress(curr / total, text=f"Analyzing {spec.modality} {curr}/{total}: {name}...")

    results = run_batch_pipeline(
        items=items, modality=spec.modality, detectors=detectors, sensitivity=sensitivity_key,
        cache_dir=session_dir, progress_callback=on_progress,
    )
    progress_bar.empty()
    st.session_state[results_key] = results
    st.session_state[sig_key] = signature
    return results


def _show_results(spec: MediaTabSpec, results: List[Dict[str, Any]]) -> None:
    if len(results) > 1:
        render_batch_summary_dashboard(results, modality=spec.modality)
        render_batch_overview_table(results, modality=spec.modality)
        selected = render_batch_file_selector(results, modality=spec.modality, key=f"{spec.key}_selector")
    else:
        selected = results[0]
    if selected and selected.get("success"):
        spec.render_result(selected)
    elif selected:
        st.error(f"❌ Failed to process `{selected['filename']}`: {selected.get('error')}")


def render_media_tab(spec: MediaTabSpec, detectors: Dict[str, Any], sensitivity_key: str, session_dir: Path) -> None:
    st.subheader(spec.subheader)
    upload_label = f"Upload {spec.noun} File(s) (Single or Batch)"
    mode = st.radio(
        f"{spec.noun} Input Method", [upload_label, f"Fetch {spec.noun} from URL(s)"],
        horizontal=True, key=f"{spec.key}_mode",
    )
    items = _ingest_uploads(spec, session_dir) if mode == upload_label else _ingest_urls(spec, session_dir)
    results = _analyse(spec, items, detectors, sensitivity_key, session_dir)
    if results:
        _show_results(spec, results)
    else:
        st.info(spec.empty_info)
