from typing import Self
from dataclasses import dataclass
from functools import cache, cached_property

from melondy.chord import Chord
from melondy.pitch import Pitch
from melondy.temporal import Time


@dataclass(frozen=True)
class VibratingString:
    pitch: Pitch

    def press_fret(self, fret_number: int | None = None) -> Pitch:
        if not fret_number:
            return self.pitch
        return self.pitch.adjust(fret_number)


@dataclass(frozen=True)
class StringTuning:
    strings: tuple[VibratingString, ...]

    @classmethod
    def from_notes(cls, *notes: str) -> Self:
        return cls(
            tuple(
                VibratingString(Pitch.from_scientific_notation(note))
                for note in reversed(notes)
            )
        )


@dataclass(frozen=True)
class PluckedStringInstrument:
    tuning: StringTuning
    vibration: Time
    damping: float = 0.5

    def __post_init__(self) -> None:
        if not (0 < self.damping <= 0.5):
            raise ValueError("String damping must be in the range (0, 0.5])")

    @cached_property
    def num_strings(self) -> int:
        return len(self.tuning.strings)

    @cache
    def downstroke(self, chord: Chord) -> tuple[Pitch, ...]:
        return tuple(reversed(self.upstroke(chord)))

    @cache
    def upstroke(self, chord: Chord) -> tuple[Pitch, ...]:
        if len(chord) != self.num_strings:
            raise ValueError(
                f"Chord and instrument must have the same string count. Chord: {len(chord)} string, Instrument: {self.num_strings} strings"
            )
        return tuple(
            string.press_fret(fret_number)
            for string, fret_number in zip(self.tuning.strings, chord)
            if fret_number is not None
        )
