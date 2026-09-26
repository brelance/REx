"""Prompt policy helpers shared by rolling-horizon runners."""

from __future__ import annotations


def depth_context(*, depth: int, max_depth: int) -> str:
    remaining = max(0, max_depth - depth)
    if remaining == 0:
        remainder = "No decomposition levels remain."
    elif remaining == 1:
        remainder = "One decomposition level remains."
    elif remaining == 2:
        remainder = "Two decomposition levels remain."
    else:
        remainder = f"{remaining} decomposition levels remain."
    return f"Current depth: {depth}. Maximum depth: {max_depth}. {remainder}"


def execution_mode_guidance(*, depth: int, max_depth: int) -> str:
    remaining = max(0, max_depth - depth)
    guidance = """- Use "execute" when the step can be completed in one focused
  execution, normally involving simple, direct work or a synthesis action.
- Use "decompose" when the step requires two or more distinct retrieval,
  verification, transformation, or action stages. A decomposed step runs in a
  child frame with its own plan and execution budget."""
    if remaining:
        return f"""{guidance}
- When recursion depth remains, prefer "decompose" for multi-source research,
  investigate-then-act workflows, and gather-then-synthesize workflows."""
    return f"""{guidance}
- No recursion depth remains. Use "execute" for every step in this frame."""
