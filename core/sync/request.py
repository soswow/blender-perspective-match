"""Complete numerical Sync inputs and a strict, versioned JSON representation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
import hashlib
import json

import numpy as np

from ..geometry import Calibration, CameraIntrinsics
from .types import SimilarityTransform, SyncLineObservation, SyncMatchInput, SyncObservation


@dataclass
class SyncSolveRequest:
    """Owned numerical evidence; execution callbacks and caches are separate."""

    matches: list[SyncMatchInput]
    observations: list[SyncObservation]
    anchor_id: str
    known_world: dict[str, np.ndarray] | None = None
    line_observations: list[SyncLineObservation] | None = None
    known_lines: dict[str, tuple[np.ndarray, np.ndarray]] | None = None
    parallel_pairs: list[tuple[str, str]] | None = None
    initial_similarities: dict[str, SimilarityTransform] | None = None
    fixed_similarities: dict[str, SimilarityTransform] | None = None
    lock_rotation: bool = False
    lock_translation: bool = False
    ground_slack: float | None = None
    known_3d_slack: float | None = None
    mirror_pairs: list[tuple[str, str]] | None = None
    mirror_plane: tuple[np.ndarray, np.ndarray] | None = None
    mirror_slack: float | None = None
    mirror_landmark_id: str | None = None
    plane_groups: list[tuple[str, str, int]] | None = None
    plane_slack: float | None = None
    location_match_ids: set[str] | None = None
    readonly_match_ids: set[str] | None = None

    def solver_kwargs(self) -> dict:
        """Borrow inputs for a solve; calibration repairs remain visible to apply."""
        return {item.name: getattr(self, item.name) for item in fields(SyncSolveRequest)}

    def leave_one_out_kwargs(self) -> dict:
        """Diagnose has no warm-start input; retain all evidence and constraints."""
        if self.initial_similarities:
            raise ValueError("Leave-one-out cannot replay initial similarities")
        arguments = self.solver_kwargs()
        del arguments["initial_similarities"]
        return arguments

    def to_record(self) -> dict:
        """Copy inputs into JSON values, preserving order and explicit defaults."""
        inputs = json_values(self.solver_kwargs())
        # Numerical callers may supply integer lists where Blender supplies arrays.
        # Canonicalize geometric arrays so decoding does not change their checksum.
        for match in inputs["matches"]:
            calibration = match["calibration"]
            calibration["rotation_w2c"] = _array(calibration["rotation_w2c"], (3, 3)).tolist()
            calibration["camera_center"] = _array(calibration["camera_center"], (3,)).tolist()
        for key in ("initial_similarities", "fixed_similarities"):
            for similarity in (inputs[key] or {}).values():
                similarity["rotation"] = _array(similarity["rotation"], (3, 3)).tolist()
                similarity["translation"] = _array(similarity["translation"], (3,)).tolist()
        for key, shape in (("known_world", (3,)), ("known_lines", (2, 3))):
            if inputs[key] is not None:
                inputs[key] = {name: _array(value, shape).tolist() for name, value in inputs[key].items()}
        if inputs["mirror_plane"] is not None:
            inputs["mirror_plane"] = _array(inputs["mirror_plane"], (2, 3)).tolist()
        return {"format": "perspective-match-sync-request", "version": 4,
                "inputs": inputs, "sha256": request_fingerprint(inputs)}

    @classmethod
    def from_record(cls, record: dict) -> SyncSolveRequest:
        """Decode a complete snapshot; reject missing fields and unknown versions."""
        version = record.get("version")
        if record.get("format") != "perspective-match-sync-request" or type(version) is not int or version not in {1, 2, 3, 4}:
            raise ValueError("Unsupported Sync request format/version")
        inputs = record["inputs"]
        if record.get("sha256") != request_fingerprint(inputs):
            raise ValueError("Sync request checksum mismatch")
        values = dict(inputs)
        if version == 1:
            # Before camera roles existed, every camera could contribute to 3D.
            if "location_match_ids" in values or "readonly_match_ids" in values:
                raise ValueError("Camera role fields require Sync request version 2")
            values.update(location_match_ids=None, readonly_match_ids=None)
        if version in {1, 2}:
            if "plane_groups" in values or "plane_slack" in values:
                raise ValueError("Plane fields require Sync request version 3")
            values.update(plane_groups=None, plane_slack=None)
        if version in {1, 2, 3}:
            if "mirror_landmark_id" in values:
                raise ValueError("Mirror landmark field requires Sync request version 4")
            values["mirror_landmark_id"] = None
        _check_fields(values, SyncSolveRequest)
        if values["mirror_landmark_id"] is not None and (
            not isinstance(values["mirror_landmark_id"], str)
            or not values["mirror_landmark_id"]
        ):
            raise ValueError("Invalid mirror_landmark_id")
        for key in ("location_match_ids", "readonly_match_ids"):
            members = values[key]
            if members is not None:
                if not isinstance(members, list) or not all(isinstance(member, str) for member in members):
                    raise ValueError(f"Invalid {key}")
                if len(members) != len(set(members)):
                    raise ValueError(f"Duplicate members in {key}")
                values[key] = set(members)
        values["matches"] = []
        for match in inputs["matches"]:
            _check_fields(match, SyncMatchInput)
            calibration = dict(match["calibration"])
            _check_fields(calibration, Calibration)
            _check_fields(calibration["intrinsics"], CameraIntrinsics)
            calibration["intrinsics"] = CameraIntrinsics(**calibration["intrinsics"])
            calibration["rotation_w2c"] = _array(calibration["rotation_w2c"], (3, 3))
            calibration["camera_center"] = _array(calibration["camera_center"], (3,))
            calibration["brown_conrady"] = tuple(calibration["brown_conrady"])
            values["matches"].append(SyncMatchInput(match["match_id"], Calibration(**calibration)))
        for key, kind in (("observations", SyncObservation), ("line_observations", SyncLineObservation)):
            if inputs[key] is not None:
                values[key] = []
                for observation in inputs[key]:
                    _check_fields(observation, kind)
                    values[key].append(kind(**observation))
        for key in ("initial_similarities", "fixed_similarities"):
            if inputs[key] is not None:
                values[key] = {}
                for name, similarity in inputs[key].items():
                    _check_fields(similarity, SimilarityTransform)
                    values[key][name] = SimilarityTransform(
                        similarity["scale"], _array(similarity["rotation"], (3, 3)),
                        _array(similarity["translation"], (3,)),
                    )
        if inputs["known_world"] is not None:
            values["known_world"] = {key: _array(point, (3,)) for key, point in inputs["known_world"].items()}
        if inputs["known_lines"] is not None:
            values["known_lines"] = {key: tuple(_array(ends, (2, 3))) for key, ends in inputs["known_lines"].items()}
        for key in ("parallel_pairs", "mirror_pairs"):
            if inputs[key] is not None:
                if any(len(pair) != 2 or not all(isinstance(value, str) for value in pair) for pair in inputs[key]):
                    raise ValueError(f"Invalid {key}")
                values[key] = [tuple(pair) for pair in inputs[key]]
        if inputs["mirror_plane"] is not None:
            values["mirror_plane"] = tuple(_array(inputs["mirror_plane"], (2, 3)))
        if values["plane_groups"] is not None:
            parsed = []
            for item in values["plane_groups"]:
                if (
                    not isinstance(item, (list, tuple))
                    or len(item) != 3
                    or not isinstance(item[0], str)
                    or not isinstance(item[1], str)
                    or type(item[2]) is not int
                ):
                    raise ValueError("Invalid plane_groups")
                parsed.append((str(item[0]), str(item[1]), int(item[2])))
            values["plane_groups"] = parsed
        return cls(**values)


def json_values(value):
    """Copy dataclasses/arrays into JSON values without serializing executable code."""
    if is_dataclass(value):
        return json_values(asdict(value))
    if isinstance(value, np.ndarray):
        return value.astype(np.float64).tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_values(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_values(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [json_values(item) for item in sorted(value)]
    return value


def request_fingerprint(inputs: dict) -> str:
    """Hash evidence values and list order; exclude capture time and machine data."""
    encoded = json.dumps(inputs, sort_keys=True, allow_nan=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _check_fields(value: dict, kind) -> None:
    expected = {item.name for item in fields(kind)}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"Incomplete or unknown fields in {kind.__name__}")


def _array(value, shape) -> np.ndarray:
    result = np.array(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"Expected finite array with shape {shape}")
    return result
