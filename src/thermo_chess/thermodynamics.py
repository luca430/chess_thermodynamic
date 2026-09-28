"""Finite heat/work/accessibility decomposition of adaptive observables."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

from .search import AdaptiveBranchObservation


@dataclass(frozen=True)
class TransitionDecomposition:
    u_before: float
    u_after: float
    delta_u: float
    delta_q: float
    delta_w: float
    delta_a: float
    decomposition_error: float
    old_support_size: int
    new_support_size: int
    common_support_size: int
    common_mass_old: float
    common_mass_new: float
    old_only_probability_mass: float
    new_only_probability_mass: float
    boundary_accessibility: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def adaptive_observable(branches: Iterable[AdaptiveBranchObservation]) -> float:
    return sum(
        branch.probability * branch.adaptive_branch_value for branch in branches
    )


def decompose_transition(
    old_branches: Iterable[AdaptiveBranchObservation],
    new_branches: Iterable[AdaptiveBranchObservation],
    u_before: float | None = None,
    u_after: float | None = None,
) -> TransitionDecomposition:
    """Decompose a finite transition using exact adaptive branch values.

    A terminal or depth-zero adaptive value can have no branch representation.
    Its static fallback is treated as a boundary accessibility contribution.
    """

    old = {branch.uci: branch for branch in old_branches}
    new = {branch.uci: branch for branch in new_branches}
    old_sum = adaptive_observable(old.values())
    new_sum = adaptive_observable(new.values())
    before = old_sum if u_before is None else u_before
    after = new_sum if u_after is None else u_after

    common = old.keys() & new.keys()
    old_only = old.keys() - new.keys()
    new_only = new.keys() - old.keys()

    delta_q = sum(
        0.5
        * (old[uci].adaptive_branch_value + new[uci].adaptive_branch_value)
        * (new[uci].probability - old[uci].probability)
        for uci in common
    )
    delta_w = sum(
        0.5
        * (old[uci].probability + new[uci].probability)
        * (new[uci].adaptive_branch_value - old[uci].adaptive_branch_value)
        for uci in common
    )
    boundary_accessibility = (after - new_sum) - (before - old_sum)
    delta_a = (
        sum(
            new[uci].probability * new[uci].adaptive_branch_value
            for uci in new_only
        )
        - sum(
            old[uci].probability * old[uci].adaptive_branch_value
            for uci in old_only
        )
        + boundary_accessibility
    )
    delta_u = after - before
    error = delta_u - (delta_q + delta_w + delta_a)

    return TransitionDecomposition(
        u_before=before,
        u_after=after,
        delta_u=delta_u,
        delta_q=delta_q,
        delta_w=delta_w,
        delta_a=delta_a,
        decomposition_error=error,
        old_support_size=len(old),
        new_support_size=len(new),
        common_support_size=len(common),
        common_mass_old=sum(old[uci].probability for uci in common),
        common_mass_new=sum(new[uci].probability for uci in common),
        old_only_probability_mass=sum(old[uci].probability for uci in old_only),
        new_only_probability_mass=sum(new[uci].probability for uci in new_only),
        boundary_accessibility=boundary_accessibility,
    )
