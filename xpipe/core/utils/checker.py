"""checker"""


def check_equal(actual, expected, msg=None):
    """
    Check if two values are equal.

    :param actual: The actual value to check.
    :param expected: The expected value.
    :param msg: Optional message to display on failure.
    :raises RuntimeError: If actual does not equal expected.
    """
    if actual != expected:
        raise RuntimeError(msg or f"Expected {expected}, but got {actual}.")


def check_not_equal(actual, unexpected, msg=None):
    """
    Check if two values are not equal.

    :param actual: The actual value to check.
    :param unexpected: The unexpected value.
    :param msg: Optional message to display on failure.
    :raises RuntimeError: If actual equals unexpected.
    """
    if actual == unexpected:
        raise RuntimeError(msg or f"Did not expect {unexpected}, but got {actual}.")


def check_true(value, msg=None):
    """
    Check if a value is True.

    :param value: The value to check.
    :param msg: Optional message to display on failure.
    :raises RuntimeError: If value is not True.
    """
    if not value:
        raise RuntimeError(msg or f"Expected True, but got {value}.")


def check_false(value, msg=None):
    """
    Check if a value is False.

    :param value: The value to check.
    :param msg: Optional message to display on failure.
    :raises RuntimeError: If value is not False.
    """
    if value:
        raise RuntimeError(msg or f"Expected False, but got {value}.")
