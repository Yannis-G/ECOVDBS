def bytes_to_mb(bytes: int) -> float:
    """
    Converts bytes to MB

    :param bytes: bytes to convert
    :return: bytes converted to MB
    """
    return round(bytes / 1024 / 1024, 2)
