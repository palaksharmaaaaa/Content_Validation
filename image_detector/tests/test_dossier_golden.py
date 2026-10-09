"""Seeded 1500-case golden for the image nine-dimension dossier (random key presence)."""
import json
import random

from image_detector.explain import IPTC_SOURCE_TYPE_MAPPING, build_nine_dimensions_dossier


from tests.golden_support import assert_golden


def maybe(r, p, v):
    return v if r.random() < p else None


def case(r):
    def put(d, k, v, p=.7):
        if r.random() < p:
            d[k] = v

    geom = {}
    put(geom, "width", r.randint(64, 4000)); put(geom, "height", r.randint(64, 4000)); put(geom, "megapixels", round(r.uniform(.1, 20), 2))
    put(geom, "total_pixels", r.randint(1000, 10**7)); put(geom, "aspect_ratio_str", "16:9"); put(geom, "aspect_ratio", 1.78); put(geom, "orientation", "Portrait")
    disp = {}
    put(disp, "dpi_str", "300 x 300 DPI"); put(disp, "bit_depth", "24-bit"); put(disp, "color_space", "ICC")
    pcol = {}
    for k, v in (("shannon_entropy_bpp", 7.1), ("luminance_mean", 100.5), ("luminance_min", 1), ("luminance_max", 250), ("luminance_median", 99.0),
                 ("highlight_clipped_pct", 1.2), ("highlight_clipped_count", 12345), ("shadow_crushed_pct", .5), ("shadow_crushed_count", 99),
                 ("dominant_palette", [{"hex": "#fff"}]), ("unique_quantized_colors", 77)):
        put(pcol, k, v)
    exif = {}
    for k, v in (("camera_make", "Canon"), ("camera_model", "R5"), ("lens_model", "RF"), ("exposure_time", "1/250s"), ("aperture", "f/2.8"), ("iso", 100),
                 ("focal_length", "50mm"), ("flash", "Fired"), ("white_balance", "Manual"), ("date_time", "2024"), ("software", "Adobe"),
                 ("gps_details", {"coordinates_str": "12N"})):
        put(exif, k, v, .4)
    phys = {}
    for k in ("flat_region_noise_mean", "surface_smoothness_index", "fft_decay_alpha", "dark_line_art_pct", "canny_edge_pct", "laplacian_sharpness_var"):
        put(phys, k, round(r.uniform(0, 5), 3), .7)
    prof = {"spatial_geometry": geom, "display_attributes": disp, "pixel_color_profile": pcol, "exif_device_details": exif, "raw_physical_signals": phys}
    put(prof, "width", 10, .3); put(prof, "height", 11, .3); put(prof, "channels", r.choice([1, 3, 4]), .5); put(prof, "pixel_entropy", 6.5, .5)
    if r.random() < .15:
        prof = {"pixel_specifications": geom}
    ai = {"taxonomy_state": r.choice([None] + list(IPTC_SOURCE_TYPE_MAPPING) + ["UNKNOWN"])}
    put(ai, "forensic_metrics", {"noise_residual_mean": 1.1, "surface_smoothness": 2.2, "spectral_decay_alpha": 3.9}, .5)
    put(ai, "subject_genre", "Landscape", .4); put(ai, "visual_medium", "Watercolor", .4); put(ai, "digital_art_detected", True, .3)
    put(ai, "sensor_spectrum", "NIR", .4); put(ai, "watermark_detected", True, .3); put(ai, "watermark_details", "sparkle", .3)
    put(ai, "ai_spatial_area_pct", 12.5, .4)
    put(ai, "model_attribution", {"attributed_model": "X", "region_of_origin": "Y"}, .3)
    inv = {}
    put(inv, "living_entities", {"humans": {"persons_count": 2, "faces_count": 1, "bounding_boxes": [[1, 2, 3, 4]], "is_stylized_character": True},
                                 "animals": {"count": 1, "animal_types": ["dog"]}}, .4)
    put(inv, "entities", {"humans": {"persons_count": 3}}, .3)
    put(inv, "environment_and_surroundings", {"setting_type": "Street", "setting": "Outdoor"}, .4)
    put(inv, "environment", {"setting_type": "Room"}, .2)
    put(inv, "lighting_and_daytime", {"daytime": "Night", "lighting_quality": "Neon"}, .4)
    put(inv, "tone_and_mood", {"atmospheric_mood": "Tense"}, .4)
    put(inv, "purpose_and_depiction", {"primary_genre": "Portrait"}, .4)
    put(inv, "vehicles", {"count": 2, "vehicle_types": ["car"]}, .3)
    put(inv, "contents_and_items", {"identified_items": ["cup"]}, .3)
    put(inv, "persons_count", 5, .2); put(inv, "faces_count", 4, .2)
    prov = {}
    put(prov, "camera_make", "Sony", .3); put(prov, "camera_model", "A7", .3); put(prov, "c2pa_present", True, .4); put(prov, "date_time", "2020", .3); put(prov, "software", "GIMP", .3)
    attr = {}
    put(attr, "attributed_model", "Flux", .4); put(attr, "region_of_origin", "EU", .4); put(attr, "attribution_confidence", .8, .3); put(attr, "confidence", .6, .3)
    put(attr, "top_candidates", [{"model": "m", "confidence": .5}], .3)
    return prof, ai, inv, (prov if r.random() < .7 else None), (attr if r.random() < .6 else None)



def test_dossier_matches_golden():
    r = random.Random(21)
    actual = []
    for _ in range(1500):
        p, a, i, pr, at = case(r)
        actual.append(json.loads(json.dumps(build_nine_dimensions_dossier(p, a, i, pr, at), sort_keys=True, default=str)))
    assert_golden("image_dossier_golden", actual)
