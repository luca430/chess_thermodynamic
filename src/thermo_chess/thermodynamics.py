"""Finite heat/work/accessibility decomposition of landscape observables."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

from .search import LandscapeObservation


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

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def landscape_observable(branches: Iterable[LandscapeObservation]) -> float:
    return sum(
        branch.probability * branch.observable_value for branch in branches
    )


def decompose_transition(
    old_branches: Iterable[LandscapeObservation],
    new_branches: Iterable[LandscapeObservation],
) -> TransitionDecomposition:
    """Decompose a transition between two complete normalized landscapes."""

    old = {branch.uci: branch for branch in old_branches}
    new = {branch.uci: branch for branch in new_branches}
    old_sum = landscape_observable(old.values())
    new_sum = landscape_observable(new.values())
    before = old_sum
    after = new_sum

    common = old.keys() & new.keys()
    old_only = old.keys() - new.keys()
    new_only = new.keys() - old.keys()

    delta_q = sum(
        0.5
        * (old[uci].observable_value + new[uci].observable_value)
        * (new[uci].probability - old[uci].probability)
        for uci in common
    )
    delta_w = sum(
        0.5
        * (old[uci].probability + new[uci].probability)
        * (new[uci].observable_value - old[uci].observable_value)
        for uci in common
    )
    delta_a = (
        sum(
            new[uci].probability * new[uci].observable_value
            for uci in new_only
        )
        - sum(
            old[uci].probability * old[uci].observable_value
            for uci in old_only
        )
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
    )
