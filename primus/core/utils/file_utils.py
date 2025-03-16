from pathlib import Path


class PathNotFoundError(Exception):
    """Custom exception for path not found errors."""


def path_exists(path):
    """
    Check if the path exists.

    :param path: The path to check
    :return: Boolean indicating whether the path exists
    """
    return Path(path).exists()


def is_file(path):
    """
    Check if the path is a file.

    :param path: The path to check
    :return: Boolean indicating whether the path is a file
    """
    return Path(path).is_file()


def is_directory(path):
    """
    Check if the path is a directory.

    :param path: The path to check
    :return: Boolean indicating whether the path is a directory
    """
    return Path(path).is_dir()


def check_file_exists(path):
    """
    Check if the path is a file, raise an exception if it does not exist.

    :param path: The path to check
    :raises PathNotFoundError: If the file does not exist
    """
    if not is_file(path):
        raise PathNotFoundError(f"The file '{path}' does not exist.")


def check_path_exists(path):
    """
    Check if the path exists, raise an exception if it does not.

    :param path: The path to check
    :raises PathNotFoundError: If the path does not exist
    """
    if not path_exists(path):
        raise PathNotFoundError(f"The path '{path}' does not exist.")


def create_path_if_not_exists(path):
    """
    Check if the path exists, if not, create it recursively.

    :param path: The path to check and create
    """
    if not path_exists(path):
        Path(path).mkdir(parents=True, exist_ok=True)
        print(f"Created path: '{path}'")
