from urllib.parse import urlparse


PLATFORM_DOMAINS = {
    "Instagram": [
        "instagram.com",
        "www.instagram.com",
    ],
    "YouTube": [
        "youtube.com",
        "www.youtube.com",
        "youtu.be",
        "www.youtu.be",
    ],
    "Facebook": [
        "facebook.com",
        "www.facebook.com",
        "fb.watch",
    ],
    "TikTok": [
        "tiktok.com",
        "www.tiktok.com",
    ],
    "X": [
        "x.com",
        "www.x.com",
        "twitter.com",
        "www.twitter.com",
    ],
}


def normalize_domain(domain):
    domain = domain.lower().strip()

    if domain.startswith("www."):
        domain = domain[4:]

    return domain


def detect_platform(url):

    try:

        parsed = urlparse(url)

        domain = normalize_domain(
            parsed.netloc
        )

        if not domain:
            return "Unknown"

        for platform, domains in PLATFORM_DOMAINS.items():

            normalized_domains = [
                normalize_domain(d)
                for d in domains
            ]

            if domain in normalized_domains:
                return platform

        return "Unknown"

    except Exception:

        return "Unknown"


def validate_url(url):

    result = {
        "valid_url": False,
        "platform": "Unknown",
        "message": "",
    }

    if not url:
        result["message"] = "URL is empty."
        return result

    url = url.strip()

    # --------------------------------
    # BASIC URL VALIDATION
    # --------------------------------

    try:

        parsed = urlparse(url)

        if parsed.scheme not in (
            "http",
            "https",
        ):

            result["message"] = (
                "URL must start with http:// or https://."
            )

            return result

        if not parsed.netloc:

            result["message"] = (
                "Invalid URL."
            )

            return result

    except Exception:

        result["message"] = (
            "Invalid URL."
        )

        return result

    # --------------------------------
    # PLATFORM DETECTION
    # --------------------------------

    platform = detect_platform(url)

    result["valid_url"] = True
    result["platform"] = platform

    if platform == "Unknown":

        result["message"] = (
            "Valid URL, but supported platform "
            "could not be detected."
        )

    else:

        result["message"] = (
            f"Valid {platform} URL."
        )

    return result


def validate_expected_platform(
    url,
    expected_platform
):

    result = validate_url(url)

    if not result["valid_url"]:

        return {
            **result,
            "platform_match": False,
        }

    detected_platform = result["platform"]

    if expected_platform == "Auto":

        return {
            **result,
            "platform_match": True,
        }

    platform_match = (
        detected_platform
        == expected_platform
    )

    if platform_match:

        message = (
            f"✅ URL matches {expected_platform}."
        )

    else:

        message = (
            f"❌ Platform mismatch. "
            f"Expected {expected_platform}, "
            f"but URL belongs to {detected_platform}."
        )

    return {
        **result,
        "platform_match": platform_match,
        "message": message,
    }