from __future__ import annotations

from abc import ABC, abstractmethod

from traffic_ai.simulation_engine.types import SignalPhase


class BaseController(ABC):
    name: str

    def __init__(self, name: str) -> None:
        self.name = name
        self.n_intersections = 0

    def reset(self, n_intersections: int) -> None:
        self.n_intersections = n_intersections

    @abstractmethod
    def compute_actions(
        self, observations: dict[int, dict[str, float]], step: int
    ) -> dict[int, SignalPhase]:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # New unified interface (Phase 3+ controllers)
    # ------------------------------------------------------------------

    def select_action(self, obs: dict[str, float]) -> int:
        """Select a phase action index for a single intersection.

        Returns
        -------
        int
            0 = NS_THROUGH (or legacy NS)
            1 = EW_THROUGH (or legacy EW)
            2 = NS_LEFT  (4-phase RL controllers only)
            3 = EW_LEFT  (4-phase RL controllers only)

        Non-RL controllers always return 0 or 1. RL controllers may return 0–3.
        """
        from traffic_ai.simulation_engine.types import PHASE_TO_IDX
        phase_map = self.compute_actions({0: obs}, step=int(obs.get("step", 0)))
        phase = phase_map.get(0, "NS")
        return PHASE_TO_IDX.get(str(phase), 0)

    def update(
        self,
        obs: dict[str, float],
        action: int,
        reward: float,
        next_obs: dict[str, float],
        done: bool = False,
    ) -> None:
        """Update internal state/model with a transition. Default: no-op."""

