"""ui.results: result pages and their sections.

pages        render_image_result / render_video_result / render_audio_result (verdict card + Overview | Evidence | Details | Feedback)
summary      the verdict card
checks       Evidence tab: integrity / metadata / context checks, attribution, localisation
media        previews, heatmaps, timelines
profile      file profile sections (Details tab)
content      scene, entity and inventory sections (Details tab)
dimensions   the nine-dimension breakdown (Details tab)
explanation  plain-English explanation
feedback     "was this right?" form and report download
learning     Learning tab: what was learned, retraining
"""
from ui.results.learning import render_learning_dashboard, render_retrain_panel
from ui.results.pages import render_audio_result, render_image_result, render_video_result

__all__ = [
    "render_audio_result", "render_image_result", "render_learning_dashboard", "render_retrain_panel", "render_video_result",
]
