"""Stage 16 paper-traceable reference-tracking RL components.

The package deliberately separates paper-exact MDP terms from engineering
simulation choices.  Importing it does not require MuJoCo or Torch; those are
optional backends used by dedicated modules and scripts.
"""

from .contracts import Stage16ReferenceClip, Stage16ReferenceValidationError

__all__ = ["Stage16ReferenceClip", "Stage16ReferenceValidationError"]
