"""Implementation modules of the ``mark`` command line (see :mod:`mark.cli`).

Everything here imports only the standard library at module load; MARK
itself is loaded lazily by :class:`mark._cli.engine._Engine`, and never on
the client path when a ``mark serve`` process answers.
"""
