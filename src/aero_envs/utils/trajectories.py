import numpy as np
import bisect
from typing import Callable, List, Optional, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


TILT_MAX = 40 * np.pi / 180

@dataclass
class TrajectorySegment:
    func: Callable[[float], float]
    deriv_func: Callable[[float], float]
    duration: float
    description: str

def clamp(value: float) -> float:
    """Clamps values to the defined tilt boundaries."""
    return np.clip(value, -TILT_MAX, TILT_MAX)

class TrajectoryGenerator:
    def __init__(self):
        self.segments: List[TrajectorySegment] = []
        self.breakpoints: List[float] = [0.0]  # Store the cumulative start time of each segment for O(log N) search.
        self.total_duration = 0.0

    def reset(self):
        """Resets the generator logic (clears data)."""
        self.segments = []
        self.breakpoints = [0.0]
        self.total_duration = 0.0

    def _get_start_val(self) -> float:
        """Helper to get the end value of the previous segment to ensure continuity."""
        if not self.segments:
            return 0.0
        last_seg = self.segments[-1]
        return last_seg.func(last_seg.duration)

    def _add_segment_logic(self, segment: TrajectorySegment):
        """Internal helper to append segment and update breakpoints."""
        self.segments.append(segment)
        self.total_duration += segment.duration
        self.breakpoints.append(self.total_duration)

    def add_ramp(self, end_value: float, duration: float):
        """Linear ramp from current value to end_value."""
        end_value = clamp(end_value)
        start_val = self._get_start_val()
        slope = (end_value - start_val) / duration

        # Lambda captures start_val and slope
        self._add_segment_logic(TrajectorySegment(
            func=lambda t, s=start_val, m=slope: s + m * t,
            deriv_func=lambda t, m=slope: m,
            duration=duration,
            description=f"Ramp to {end_value:.2f}"
        ))
        return self

    def add_sine(self, amplitude: float, frequency: float, duration: float):
        """
        Adds a relative sine wave to ensure continuity.
        Formula: start_val + A * sin(2*pi*f*t)
        Note: This starts the sine phase at 0. For smoother transitions,
        consider fading in or ensuring the previous derivative matches (advanced).
        """
        amplitude = clamp(amplitude)
        start_val = self._get_start_val()

        def sine_func(t: float, s=start_val, A=amplitude, f=frequency) -> float:
            return clamp(s + A * np.sin(2 * np.pi * f * t))

        def func_vel(t, A=amplitude, f=frequency):
            return A * 2 * np.pi * f * np.cos(2 * np.pi * f * t)

        self._add_segment_logic(TrajectorySegment(
            func=sine_func,
            deriv_func=func_vel,
            duration=duration,
            description=f"Sine (Amp={amplitude:.2f}, Freq={frequency:.2f})"
        ))
        return self

    def add_s_curve(self, end_value: float, duration: float):
        """
        Smooth interpolation (cosine ease-in-out).
        Best for Quanser Aero to avoid infinite jerk (derivative spikes).
        """
        end_value = clamp(end_value)
        start_val = self._get_start_val()
        delta = end_value - start_val

        self._add_segment_logic(TrajectorySegment(
            func=lambda t, s=start_val, d=delta, dur=duration: s + d * (1 - np.cos(np.pi * t / dur)) / 2,
            deriv_func=lambda t, d=delta, dur=duration: d * (np.pi / (2 * dur)) * np.sin(np.pi * t / dur),
            duration=duration,
            description=f"S-Curve to {end_value:.2f}"
        ))
        return self

    def add_step(self, value: float, duration: float):
        """Instant jump. Use with caution in physical systems."""
        value = clamp(value)
        self._add_segment_logic(TrajectorySegment(
            func=lambda t: value,
            deriv_func=lambda t: 0.0,
            duration=duration,
            description=f"Step to {value:.2f}"
        ))
        return self

    def add_hold(self, duration: float):
        """Holds the current value."""
        value = self._get_start_val()
        self._add_segment_logic(TrajectorySegment(
            func=lambda t: value,
            deriv_func=lambda t: 0.0,
            duration=duration,
            description=f"Hold at {value:.2f}"
        ))
        return self

    def _resolve_local_time(self, t: float) -> Tuple[Optional[TrajectorySegment], float]:
        """
        Resolves global time t to the appropriate segment and local time within that segment.
        Returns (segment, local_time) or (None, 0.0) if no segments exist.
        """
        if not self.segments:
            return None, 0.0

        t_mod = t % self.total_duration

        idx = bisect.bisect_right(self.breakpoints, t_mod) - 1
        idx = max(0, min(idx, len(self.segments) - 1))  # Safety check for floating point edge cases
        segment = self.segments[idx]
        local_t = t_mod - self.breakpoints[idx]
        return segment, local_t

    def get_value(self, t: float) -> float:
        """
        Calculates the value at any global time t.
        """
        segment, local_t = self._resolve_local_time(t)
        if segment is None:
            return 0.0
        return segment.func(local_t)

    def get_derivative(self, t: float) -> float:
        """
        Calculates the derivative at any global time t.
        """
        segment, local_t = self._resolve_local_time(t)
        if segment is None:
            return 0.0
        return segment.deriv_func(local_t)

    def plot(self, title="Trajectory", show=True):
        """Visualizes the trajectory."""
        if self.total_duration == 0:
            print("No trajectory to plot.")
            return

        times = np.arange(0.0, self.total_duration, 1.0 / 10)
        vals = [self.get_value(t) * 180 / np.pi for t in times]
        derivs = [self.get_derivative(t) * 180 / np.pi for t in times]

        plt.figure(figsize=(10, 5))
        plt.plot(times, vals, label="Target Angle")
        plt.plot(times, derivs, label="Target Velocity", linestyle='--')
        plt.xlabel("Time (s)")
        plt.ylabel("Angle (deg)")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        if show:
            plt.show()

    def __call__(self, t: float) -> float:
        """Allows the trajectory to be called as a function."""
        return self.get_value(t)


class RandomTrajectoryBase(TrajectoryGenerator, ABC):
    """
    Abstract base class for random trajectory generators.
    Requires a regenerate() method to create new profiles.
    """

    def __init__(self, max_duration: float = 60.0, seed: int | np.random.Generator = None):
        self.rng = np.random.default_rng(seed)
        super().__init__()
        self.max_duration = max_duration

    @abstractmethod
    def regenerate(self):
        """Clears and creates a new random profile."""
        pass


# ------------------------------------------------------------------
# Concrete Implementations for RL
# ------------------------------------------------------------------

class RandomTrajectory(RandomTrajectoryBase):
    """
    Generates a random trajectory for training.
    Can be regenerated without re-instantiation.
    """

    def __init__(self, max_duration: float = 60.0, seed: int | np.random.Generator = None):
        super().__init__(max_duration=max_duration, seed=seed)
        self.regenerate()

    def __str__(self):
        return "RandomTrajectory"

    def regenerate(self):
        """Clears and creates a new random profile."""
        self.reset()
        current_t = 0.0
        min_duration, max_duration = 2.25, 15.0

        while current_t < self.max_duration:
            duration = self.rng.uniform(min_duration, max_duration)
            if self.total_duration + duration > self.max_duration:
                duration = self.max_duration - self.total_duration

            target = self.rng.uniform(-40, 40) * np.pi / 180

            #choice = self.rng.random.choice(["s_curve", "ramp", "hold", "sine", "step"])
            choice = self.rng.choice(["s_curve", "hold"], p=[0.7, 0.3])

            if duration < min_duration:
                choice = "hold"

            if choice == "s_curve":
                self.add_s_curve(target, duration)
            elif choice == "sine":
                amp = np.random.uniform(5, 15) * np.pi / 180
                freq = 1 / duration
                self.add_sine(amp, freq, duration)
            elif choice == "hold":
                self.add_hold(duration)
            elif choice == "ramp":
                self.add_ramp(target, duration)
            elif choice == "step":
                self.add_step(target, duration)

            current_t += duration


class EvaluationTrajectory(TrajectoryGenerator):
    """
    Fixed trajectory for evaluation.
    Combines holds, s-curves, ramps, and sine waves.
    """

    def __init__(self):
        super().__init__()
        self.add_hold(3.0)
        self.add_s_curve(40 * np.pi / 180, 5.0)
        self.add_s_curve(-40 * np.pi / 180, 2.25)
        self.add_hold(5.0)
        self.add_s_curve(0, 7.75)
        self.add_s_curve(25 * np.pi / 180, 12.5)
        self.add_hold(5.0)
        self.add_s_curve(-25 * np.pi / 180, 12.5)
        self.add_s_curve(0, 2.0)
        self.add_hold(5.0)

    def __str__(self):
        return "EvaluationTrajectory"



class EvaluationStepTrajectory(TrajectoryGenerator):
    """
    Fixed trajectory for evaluation.
    Consists of a series of steps to predefined angles.
    """

    def __init__(self):
        super().__init__()
        self.add_step(0.0, 10.0)
        self.add_step(5 * np.pi / 180, 10.0)
        self.add_step(-5 * np.pi / 180, 10.0)
        self.add_step(20 * np.pi / 180, 10.0)
        self.add_step(-20 * np.pi / 180, 10.0)
        self.add_step(40 * np.pi / 180, 10.0)
        self.add_step(-40 * np.pi / 180, 10.0)
        self.add_step(0.0, 10.0)


    def __str__(self):
        return "EvaluationStepTrajectory"


    def plot(self, title="Evaluation Step Trajectory", show=True):
        """Override to set a specific title."""
        super().plot(title=title, show=show)


class RandomStepTrajectory(RandomTrajectoryBase):
    """
    Random step trajectory for evaluation.
    Series of steps to random angles within limits.
    """

    def __init__(self, max_duration: float = 60.0):
        super().__init__(max_duration)
        self.regenerate()

    def __str__(self):
        return "RandomStepTrajectory"

    def regenerate(self):
        current_t = 0.0
        while current_t < self.max_duration:
            duration = np.random.uniform(5.0, 15.0)
            if current_t + duration > self.max_duration:
                duration = self.max_duration - current_t
            target = np.random.uniform(-40, 40) * np.pi / 180
            self.add_step(target, duration)
            current_t += duration
