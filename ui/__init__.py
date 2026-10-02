"""
ui: Presentation and user interface components for OmniForensics.
Provides:
- Linear and batch forensic inspectors for Image, Video, and Audio.
- Continuous learning rating and calibration widgets.
- Media validation and secure URL fetching wrappers.
"""
from ui.feedback_ui import (
    render_linear_image_pipeline_results,
    render_linear_video_pipeline_results,
    render_linear_audio_pipeline_results,
    render_learning_dashboard,
    render_bottom_feedback_panel,
    render_feedback_box,
    render_media_specs,
    render_pre_analysis_specifications,
    render_nine_dimensions_breakdown,
    render_scene_and_content_intelligence,
    profile_media,
)
from ui.batch_ui import (
    process_single_image,
    process_single_video,
    process_single_audio,
    run_batch_pipeline,
    render_batch_summary_dashboard,
    render_batch_overview_table,
    render_batch_file_selector,
)
from ui.validators import (
    validate_file,
    analyze_provenance,
    fetch_media_from_url,
    validate_expected_platform,
)

__all__ = [
    "render_linear_image_pipeline_results",
    "render_linear_video_pipeline_results",
    "render_linear_audio_pipeline_results",
    "render_learning_dashboard",
    "render_bottom_feedback_panel",
    "render_feedback_box",
    "render_media_specs",
    "render_pre_analysis_specifications",
    "render_nine_dimensions_breakdown",
    "render_scene_and_content_intelligence",
    "profile_media",
    "process_single_image",
    "process_single_video",
    "process_single_audio",
    "run_batch_pipeline",
    "render_batch_summary_dashboard",
    "render_batch_overview_table",
    "render_batch_file_selector",
    "validate_file",
    "analyze_provenance",
    "fetch_media_from_url",
    "validate_expected_platform",
]
