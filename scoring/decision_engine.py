def generate_final_decision(
    file_validation,
    quality_result,
    ai_result=None,
):

    if not file_validation["readable"]:

        return {
            "content_valid": False,
            "final_status": "INVALID_FILE",
            "reason": file_validation.get(
                "error",
                "File cannot be read."
            ),
        }

    if quality_result.get(
        "content_status"
    ) == "INVALID":

        return {
            "content_valid": False,
            "final_status": "INVALID_CONTENT",
            "reason": (
                "Content is mostly blank "
                "or unusable."
            ),
        }

    if quality_result.get(
        "is_blank"
    ) is True:

        return {
            "content_valid": False,
            "final_status": "BLANK_CONTENT",
            "reason": (
                "The uploaded image is "
                "effectively blank."
            ),
        }

    result = {
        "content_valid": True,
        "final_status": "VALID",
        "reason": "Usable content detected.",
    }

    if (
        quality_result.get(
            "content_status"
        )
        == "PARTIALLY_VALID"
    ):

        result[
            "final_status"
        ] = "PARTIALLY_VALID"

        result[
            "reason"
        ] = (
            "Only part of the uploaded "
            "content contains usable content."
        )

    if ai_result:

        result[
            "ai_detection"
        ] = ai_result

    return result