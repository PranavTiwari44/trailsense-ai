"""Errors our data code can raise.

Later, when the AI asks for data, we'll catch these and send back a clear
message instead of crashing.
"""


class DataUnavailableError(Exception):
    """An outside website failed, was too slow, or sent data we can't use."""


class TrailNotFoundError(Exception):
    """No trail matched the name, online or in our saved file."""
