import numpy as np
from itertools import cycle
from dataclasses import dataclass
from typing import Iterator, Sequence

from melondy.temporal import Time, Hertz
from melondy.processing import normalise, remove_dc
from melondy.burst import BurstGenerator, WhiteNoise

SAMPLING_RATE = 44100


@dataclass(frozen=True)
class Synthesiser:
    burst_generator: BurstGenerator = WhiteNoise()
    sampling_rate: int = SAMPLING_RATE

    def vibrate(
        self, frequency: Hertz, duration: Time, damping: float = 0.5
    ) -> np.ndarray:

        def feedback_loop() -> Iterator[float]:
            buffer = self.burst_generator(
                num_samples=round(self.sampling_rate / frequency),
                sampling_rate=self.sampling_rate,
            )
            for i in cycle(range(buffer.size)):
                yield (current_sample := buffer[i])
                next_sample = buffer[(i + 1) % buffer.size]
                buffer[i] = (current_sample + next_sample) * damping

        assert 0 < damping <= 0.5
        return normalise(
            remove_dc(
                np.fromiter(
                    feedback_loop(),
                    np.float64,
                    duration.get_num_samples(self.sampling_rate),
                )
            )
        )

    def overlay(self, sounds: Sequence[np.ndarray], delay: Time) -> np.ndarray:
        num_delay_samples = delay.get_num_samples(self.sampling_rate)
        num_samples = max(
            i * num_delay_samples + sound.size for i, sound in enumerate(sounds)
        )
        samples = np.zeros(num_samples, dtype=np.float64)
        for i, sound in enumerate(sounds):
            offset = i * num_delay_samples
            samples[offset : offset + sound.size] += sound
        return samples
