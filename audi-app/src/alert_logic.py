"""Alert smoothing and color decision state for the Pi detector."""


class HysteresisState:
    """Schmitt-trigger state tracker with moving-average confirmation.

    Uses asymmetric thresholds: turns ON only when >= ratio of the last
    ``window`` scores exceed ``threshold + margin``, and turns OFF only when
    >= ratio fall below ``threshold - margin``.

    Mirrors ``audi.hysteresis.apply_hysteresis`` for standalone RPi use.
    """

    def __init__(
        self,
        threshold: float = 0.70,
        window: int = 5,
        ratio: float = 0.6,
        margin: float = 0.05,
    ):
        self.threshold = threshold
        self.window = window
        self.ratio = ratio
        self.margin = margin
        self.history: list[float] = []
        self.state = False

    def add(self, score: float) -> bool:
        """Feed a new score, return current hysteresis state."""
        self.history.append(score)
        if len(self.history) > self.window:
            self.history.pop(0)

        recent = self.history
        k = max(1, int(len(recent) * self.ratio))
        lo = self.threshold - self.margin
        hi = self.threshold + self.margin

        if self.state:
            below = sum(1 for s in recent if s < lo)
            if below >= k:
                self.state = False
        else:
            above = sum(1 for s in recent if s > hi)
            if above >= k:
                self.state = True

        return self.state

    def clear(self):
        self.history.clear()
        self.state = False

    @property
    def confidence(self) -> float:
        """Mean of recent scores for display."""
        if not self.history:
            return 0.0
        return sum(self.history) / len(self.history)


class ColorHysteresisState:
    """Stateful blue/red typing with sticky RED behavior.

    BLUE->RED uses ``enter_red_threshold``. RED->BLUE uses the lower
    ``exit_red_threshold``, so returning to blue requires stronger evidence.
    """

    def __init__(
        self,
        enter_red_threshold: float = 0.45,
        exit_red_threshold: float = 0.35,
        window: int = 5,
        ratio: float = 0.6,
    ):
        self.enter_red_threshold = enter_red_threshold
        self.exit_red_threshold = exit_red_threshold
        self.window = window
        self.ratio = ratio
        self.history: list[float] = []
        self.state = "UNKNOWN"

    def add(self, red_score: float) -> str:
        self.history.append(red_score)
        if len(self.history) > self.window:
            self.history.pop(0)

        recent = self.history
        k = max(1, int(len(recent) * self.ratio))
        above_enter = sum(1 for s in recent if s >= self.enter_red_threshold)
        below_exit = sum(1 for s in recent if s <= self.exit_red_threshold)

        if self.state == "RED":
            if below_exit >= k:
                self.state = "BLUE"
        else:
            if above_enter >= k:
                self.state = "RED"
            elif below_exit >= k:
                self.state = "BLUE"
        return self.state

    def clear(self):
        self.history.clear()
        self.state = "UNKNOWN"

    @property
    def confidence(self) -> float | None:
        if not self.history:
            return None
        return sum(self.history) / len(self.history)
