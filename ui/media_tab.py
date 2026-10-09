"""
ui.media_tab: the single implementation of an ingest -> analyse -> read-the-result tab.

The image, video and audio tabs differ only in labels, accepted file types, the detector bundle and the result page,
which live in a ``MediaTabSpec``. Everything else is shared: uploads and links are saved to the session scratch
directory, results are cached by file content, several files get a comparison table and a picker.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List

import streamlit as st

from core.hashing import file_sha256
from core.security import sanitize_filename
from ui.adapters import run_batch_pipeline
from ui.batch_views import render_batch_overview
from ui.validators import fetch_media_from_url


@dataclass(frozen=True)
class MediaTabSpec:
    """Everything that differs between the image, video and audio tabs: labels, accepted types, hint and result renderer."""
    key: str                       # "img" | "vid" | "aud": prefix for widget keys and session-state entries
    modality: str                  # "image" | "video" | "audio"
    noun: str                      # "Image" | "Video" | "Audio"
    file_types: List[str]
    hint: str                      # one line under the uploader: what the engine looks at
    url_placeholder: str
    render_result: Callable[[Dict[str, Any]], None]
    tag_upload_source: bool = False  # image items record where they came from


def _save_upload(spec_key: str, session_dir: Path, up: Any, index: int) -> Path:
    """Write one uploaded file to the session folder and return its path. Streamlit re-runs the whole script on every click, so a
    file already saved under the same upload id and size is not written again (a 500 MB video would otherwise be rewritten on
    every widget interaction). The name carries Streamlit's per-upload id, so two files with the same name never collide."""
    ident = getattr(up, "file_id", None) or f"{index}_{up.size}"
    path = session_dir / f"{spec_key}_{ident}_{sanitize_filename(up.name, max_len=80)}"
    if not (path.is_file() and path.stat().st_size == up.size):
        path.write_bytes(up.getbuffer())
    return path


def _ingest_uploads(spec: MediaTabSpec, session_dir: Path) -> List[Dict[str, Any]]:
    round_key = f"{spec.key}_uploader_round"
    uploaded = st.file_uploader(
        f"Drop {spec.noun.lower()} files here (one or several)", type=spec.file_types,
        accept_multiple_files=True, key=f"uploader_{spec.key}_{st.session_state.get(round_key, 0)}",
    )
    st.caption(spec.hint)
    if uploaded:
        with st.expander(f"{len(uploaded)} file(s) selected", expanded=len(uploaded) <= 5):
            st.markdown(chr(10).join(f"- `{up.name}` ({up.size / 1_000_000:.2f} MB)" for up in uploaded))
            if st.button("Remove all", key=f"{spec.key}_remove_all"):
                st.session_state[round_key] = st.session_state.get(round_key, 0) + 1
                st.rerun()
    items: List[Dict[str, Any]] = []
    for idx, up in enumerate(uploaded or []):
        item = {"path": str(_save_upload(spec.key, session_dir, up, idx)), "filename": up.name, "size": up.size}
        if spec.tag_upload_source:
            item["source"] = "Local Device Upload"
        items.append(item)
    return items


def _ingest_urls(spec: MediaTabSpec, session_dir: Path) -> List[Dict[str, Any]]:
    """Links the user asked to fetch. They are fetched when the button is pressed and then kept in the session, because Streamlit
    re-runs the script on every interaction and the button is only "pressed" in the run that follows the click."""
    url_key = f"{spec.key}_url_items"
    with st.expander("Or analyse files from links"):
        urls_input = st.text_area("Direct links, one per line", placeholder=spec.url_placeholder, key=f"{spec.key}_urls_input")
        clicked = st.button("Fetch and analyse", key=f"btn_fetch_{spec.key}", disabled=not urls_input.strip())
        if st.session_state.get(url_key) and st.button("Forget fetched links", key=f"btn_forget_{spec.key}"):
            st.session_state[url_key] = []
    if clicked:
        urls = [u.strip() for u in urls_input.replace(",", "\n").splitlines() if u.strip().startswith("http")]
        if not urls:
            st.error("Enter at least one link starting with http:// or https://")
        else:
            fetched_items: List[Dict[str, Any]] = []
            for idx, url in enumerate(urls):
                with st.spinner(f"Downloading {idx + 1} of {len(urls)}"):
                    fetched = fetch_media_from_url(url, expected_type=spec.modality, dest_dir=session_dir)
                if not fetched.get("success"):
                    st.error(f"Could not download {url}: {fetched.get('error')}")
                    continue
                path = fetched["file_path"]
                item = {"path": path, "filename": fetched["filename"], "size": Path(path).stat().st_size}
                if spec.tag_upload_source:
                    item["source"] = f"Link ({url})"
                fetched_items.append(item)
            st.session_state[url_key] = fetched_items
    # a file the user wiped with "Clear uploaded files" no longer exists: drop it instead of analysing a missing path
    return [it for it in st.session_state.get(url_key, []) if Path(it["path"]).is_file()]


def _analyse(spec: MediaTabSpec, items: List[Dict[str, Any]], detectors: Dict[str, Any], sensitivity_key: str, session_dir: Path) -> List[Dict[str, Any]]:
    """Batch-analyse ``items``, reusing the session's previous results when the same bytes were already analysed."""
    if not items:
        return []
    sig_key, results_key = f"{spec.key}_batch_sig", f"{spec.key}_batch_results"
    signature = tuple((it["filename"], file_sha256(it["path"]), sensitivity_key) for it in items)
    if st.session_state.get(sig_key) == signature:
        return st.session_state.get(results_key, [])

    progress_bar = st.progress(0, text="Starting")

    def on_progress(curr: int, total: int, name: str) -> None:
        progress_bar.progress(curr / total, text=f"Analysed {curr} of {total} (latest: {name})")

    results = run_batch_pipeline(
        items=items, modality=spec.modality, detectors=detectors, sensitivity=sensitivity_key,
        cache_dir=session_dir, progress_callback=on_progress,
    )
    progress_bar.empty()
    st.session_state[results_key] = results
    st.session_state[sig_key] = signature
    return results


def render_media_tab(spec: MediaTabSpec, detectors: Dict[str, Any], sensitivity_key: str, session_dir: Path) -> None:
    """Draw one tab: uploader and optional links, analyse (cached by file content), then show the result or a comparison table."""
    items = _ingest_uploads(spec, session_dir) + _ingest_urls(spec, session_dir)
    results = _analyse(spec, items, detectors, sensitivity_key, session_dir)
    if not results:
        return
    selected = render_batch_overview(results, spec.modality) if len(results) > 1 else results[0]
    if selected:
        st.divider()
        spec.render_result(selected)
