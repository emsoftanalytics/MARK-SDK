# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 MARK Contributors
"""Implementation modules for :class:`mark.agent.MarkAgent`.

* ``run_types``   AgentRun / AgentResult value objects
* ``steps``       prompt rendering, LLM call, skills, output storage
* ``middleware``  ``agent.run`` middleware routing and stack ownership

Import the public names from :mod:`mark.agent`; this package is internal and
its ``__init__`` stays empty to avoid import cycles.
"""
