"""Durable JSON serialization for editable projects, independent of the UI.

Version 1 stores musical events, user timing settings, instrument choice, and
optional normalized calibration. It deliberately excludes video bytes and model
weights. Constructors revalidate imports so malformed files cannot bypass domain
invariants. Future incompatible changes must increment the schema version and
provide an explicit migration rather than guessing how old fields should behave.
"""

import json
from dataclasses import asdict

from autotab.domain import Calibration, Note, Project

SCHEMA_VERSION = 1


def dumps(project: Project) -> str:
    """Serialize validated data to readable UTF-8-compatible JSON with a version marker."""
    return json.dumps(
        {"schema_version": SCHEMA_VERSION, **asdict(project)}, indent=2, allow_nan=False
    )


def loads(payload: str | bytes) -> Project:
    """Read schema version 1 and rebuild validated, immutable domain objects.

    Missing optional fields get their version-1 defaults for compatibility. Structural
    JSON errors are normalized to ValueError, as are domain validation failures.
    """
    try:
        data = json.loads(payload)
        if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported project format; expected schema_version 1.")
        calibration = data.get("calibration")
        if calibration is not None:
            calibration = Calibration(
                tuple(tuple(p) for p in calibration["corners"]),
                calibration["first_fret"],
                calibration["last_fret"],
            )
        return Project(
            title=data["title"],
            bpm=data["bpm"],
            notes=tuple(Note(**note) for note in data["notes"]),
            beat_offset_seconds=data.get("beat_offset_seconds", 0.0),
            subdivisions=data.get("subdivisions", 4),
            calibration=calibration,
            instrument=data.get("instrument", "guitar"),
            capo=data.get("capo", 0),
        )
    except (KeyError, TypeError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid project file: {exc}") from exc
