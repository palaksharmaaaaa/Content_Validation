import cv2
import numpy as np

from config.settings import (
    BLANK_VARIANCE_THRESHOLD,
    BLANK_EXTREME_PIXEL_RATIO,
    BLUR_THRESHOLD,
    MIN_CONTENT_COVERAGE,
    PARTIAL_CONTENT_COVERAGE,
)

from video.frame_extractor import (
    get_video_metadata,
    extract_sampled_frames,
)

def analyze_frame(frame):

    if frame is None:
        return {
            "is_blank": True,
            "is_blurry": True,
            "is_unusable": True,
            "blur_score": 0.0,
            "brightness": 0.0,
        }

    if len(frame.shape) != 3:
        return {
            "is_blank": True,
            "is_blurry": True,
            "is_unusable": True,
            "blur_score": 0.0,
            "brightness": 0.0,
        }

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    variance = float(
        np.var(gray)
    )

    dark_pixels = np.sum(
        gray <= 5
    )

    bright_pixels = np.sum(
        gray >= 250
    )

    total_pixels = gray.size

    extreme_ratio = (
        (dark_pixels + bright_pixels)
        / total_pixels
        if total_pixels > 0
        else 1.0
    )

    blur_score = float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()
    )

    brightness = float(
        np.mean(gray)
    )

    is_blank = (
        variance <= BLANK_VARIANCE_THRESHOLD
        or extreme_ratio >= BLANK_EXTREME_PIXEL_RATIO
    )

    is_blurry = (
        blur_score < BLUR_THRESHOLD
    )

    # A frame is usable only when it is neither blank nor blurry
    is_unusable = (
        is_blank
        or is_blurry
    )

    return {
        "is_blank": bool(is_blank),
        "is_blurry": bool(is_blurry),
        "is_unusable": bool(is_unusable),
        "blur_score": round(
            blur_score,
            2
        ),
        "brightness": round(
            brightness,
            2
        ),
    }

# def analyze_video(
#     video_path,
#     sample_count=30
# ):

#     # -----------------------------
#     # VIDEO METADATA
#     # -----------------------------

#     metadata = get_video_metadata(
#         video_path
#     )

#     if metadata is None:

#         return {
#             "valid": False,
#             "content_status": "INVALID",
#             "error": "Unable to read video metadata.",
#         }

#     # -----------------------------
#     # FRAME EXTRACTION
#     # -----------------------------

#     frames = extract_sampled_frames(
#         video_path,
#         num_frames=sample_count
#     )

#     if not frames:

#         return {
#             "valid": False,
#             "content_status": "INVALID",
#             "metadata": metadata,
#             "error": "No readable frames found.",
#         }

#     # -----------------------------
#     # FRAME ANALYSIS
#     # -----------------------------

#     analyzed_frames = []

#     for item in frames:

#         frame_result = analyze_frame(
#             item["frame"]
#         )

#         frame_result["frame_index"] = item[
#             "index"
#         ]

#         analyzed_frames.append(
#             frame_result
#         )

#     total = len(
#         analyzed_frames
#     )

#     blank_count = sum(
#         1
#         for frame in analyzed_frames
#         if frame["is_blank"]
#     )

#     blurry_count = sum(
#         1
#         for frame in analyzed_frames
#         if frame["is_blurry"]
#     )

#     valid_count = sum(
#         1
#         for frame in analyzed_frames
#         if not frame["is_blank"]
#         and not frame["is_blurry"]
#     )

#     content_coverage = (
#         valid_count / total
#         if total > 0
#         else 0
#     )

#     # -----------------------------
#     # CONTENT STATUS
#     # -----------------------------

#     if (
#         content_coverage
#         >= MIN_CONTENT_COVERAGE
#     ):

#         content_status = "VALID"

#     elif (
#         content_coverage
#         >= PARTIAL_CONTENT_COVERAGE
#     ):

#         content_status = "PARTIALLY_VALID"

#     else:

#         content_status = "INVALID"

#     # -----------------------------
#     # FINAL RESULT
#     # -----------------------------

#     return {

#         "valid": (
#             content_status != "INVALID"
#         ),

#         "content_status": content_status,

#         "metadata": metadata,

#         "frames_analyzed": total,

#         "blank_frames": blank_count,

#         "blurry_frames": blurry_count,

#         "valid_frames": valid_count,

#         "content_coverage": round(
#             content_coverage,
#             4
#         ),

#         "frame_details": analyzed_frames,
#     }

def analyze_video(
    video_path,
    sample_count=30
):

    # -----------------------------
    # VIDEO METADATA
    # -----------------------------

    metadata = get_video_metadata(video_path)

    if metadata is None:

        return {
            "valid": False,
            "content_status": "INVALID",
            "error": "Unable to read video metadata.",
        }

    duration = float(
        metadata.get(
            "duration_seconds",
            0
        )
    )

    # -----------------------------
    # FRAME EXTRACTION
    # -----------------------------

    frames = extract_sampled_frames(
        video_path,
        num_frames=sample_count
    )

    if not frames:

        return {
            "valid": False,
            "content_status": "INVALID",
            "metadata": metadata,
            "error": "No readable frames found.",
        }

    analyzed_frames = []

    total_frames = len(frames)

    # Approximate time represented by each sampled frame
    time_per_sample = (
        duration / total_frames
        if duration > 0
        else 0
    )

    # -----------------------------
    # ANALYZE EACH FRAME
    # -----------------------------

    for item in frames:

        frame_result = analyze_frame(
            item["frame"]
        )

        frame_result["frame_index"] = item[
            "index"
        ]

        # Approximate timestamp
        if duration > 0 and metadata.get("frame_count", 0) > 1:

            timestamp = (
                item["index"]
                / (metadata["frame_count"] - 1)
            ) * duration

        else:

            timestamp = 0

        frame_result["timestamp_seconds"] = round(
            timestamp,
            2
        )

        analyzed_frames.append(
            frame_result
        )

    # -----------------------------
    # COUNT FRAME TYPES
    # -----------------------------

    usable_count = sum(
        1
        for frame in analyzed_frames
        if not frame["is_unusable"]
    )

    blurry_count = sum(
        1
        for frame in analyzed_frames
        if frame["is_blurry"]
        and not frame["is_blank"]
    )

    blank_count = sum(
        1
        for frame in analyzed_frames
        if frame["is_blank"]
    )

    # -----------------------------
    # PERCENTAGES
    # -----------------------------

    usable_percentage = (
        usable_count / total_frames * 100
    )

    blurry_percentage = (
        blurry_count / total_frames * 100
    )

    blank_percentage = (
        blank_count / total_frames * 100
    )

    unusable_percentage = (
        blurry_percentage
        + blank_percentage
    )

    # -----------------------------
    # APPROXIMATE DURATIONS
    # -----------------------------

    usable_duration = (
        duration
        * usable_percentage
        / 100
    )

    blurry_duration = (
        duration
        * blurry_percentage
        / 100
    )

    blank_duration = (
        duration
        * blank_percentage
        / 100
    )

    unusable_duration = (
        blurry_duration
        + blank_duration
    )

    # -----------------------------
    # CONTENT STATUS
    # -----------------------------

    content_coverage = (
        usable_count / total_frames
        if total_frames > 0
        else 0
    )

    if (
        content_coverage
        >= MIN_CONTENT_COVERAGE
    ):

        content_status = "VALID"

    elif (
        content_coverage
        >= PARTIAL_CONTENT_COVERAGE
    ):

        content_status = "PARTIALLY_VALID"

    else:

        content_status = "INVALID"

    # -----------------------------
    # FINAL RESULT
    # -----------------------------

    return {

        "valid": (
            content_status != "INVALID"
        ),

        "content_status": content_status,

        "metadata": metadata,

        "frames_analyzed": total_frames,

        "usable_frames": usable_count,

        "valid_frames": usable_count,

        "blank_frames": blank_count,

        "blurry_frames": blurry_count,

        "content_coverage": round(
            content_coverage,
            4
        ),

        # Percentages
        "usable_percentage": round(
            usable_percentage,
            2
        ),

        "blurry_percentage": round(
            blurry_percentage,
            2
        ),

        "blank_percentage": round(
            blank_percentage,
            2
        ),

        "unusable_percentage": round(
            unusable_percentage,
            2
        ),

        # Approximate duration
        "usable_duration_seconds": round(
            usable_duration,
            2
        ),

        "blurry_duration_seconds": round(
            blurry_duration,
            2
        ),

        "blank_duration_seconds": round(
            blank_duration,
            2
        ),

        "unusable_duration_seconds": round(
            unusable_duration,
            2
        ),

        "frame_details": analyzed_frames,
    }