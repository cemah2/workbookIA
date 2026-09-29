"""Exceptions shared by the wb modules."""


class MylearnNotFoundError(ImportError):
    """The requested mylearn implementation does not exist yet."""


class MylearnMissingError(AttributeError):
    """A mylearn module (or the whole package) is not available yet.

    ``wb.attempt`` turns it into a friendly "⏳ pas encore fait" message.
    """
