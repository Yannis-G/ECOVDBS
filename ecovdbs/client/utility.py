from logging import Logger

from docker.models.containers import Container


def bytes_to_mb(bytes: int) -> float:
    """
    Converts bytes to MB

    :param bytes: bytes to convert
    :return: bytes converted to MB
    """
    return round(bytes / 1024 / 1024, 2)


def get_size_of(path: str, container: Container, log: Logger) -> int:
    """
    Get the size of a directory or file within the Docker container.

    :param path: Path to the directory or file within the container.
    :param container: Docker container object.
    :param log: Logger object.
    :return: Size in bytes. If the container is not available, return -1.
    """
    # Return -1 if the Docker container is not available
    if not container:
        log.error("The database container was not found.")
        return -1
    # Execute the 'du' command within the Docker container to get the size of the specified path
    result = container.exec_run(f"du -sb {path}")
    # Check if the command executed successfully
    if result.exit_code == 0:
        output = result.output.decode("utf-8").strip()
        size_in_bytes = int(output.split()[0])
        return size_in_bytes
    else:
        log.error(f"The database container returned an error: {result.exit_code}")
        return -1
