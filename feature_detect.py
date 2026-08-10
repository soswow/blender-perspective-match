"""Automatic ORB feature detection / matching for sync auto-tracks.

Uses the bundled OpenCV wheel (same as AprilTags). Detection + pairwise matching
run off the Blender main thread; callers apply plain ``TrackData`` onto RNA.

Default detector is ORB (fast, strong on textured stills in OpenCV 5.0 spike).
SIFT is available as an alternative. AKAZE is not exposed in this wheel build.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

import numpy as np

DEFAULT_DETECTOR = "ORB"
DEFAULT_MAX_FEATURES = 1500
DEFAULT_RATIO = 0.75
DEFAULT_RANSAC_PX = 3.0
# After RANSAC, drop the worst multi-view tracks by residual (keep this %).
DEFAULT_KEEP_PERCENTILE = 80.0
DEFAULT_MAX_MULTI_VIEW = 400
# Cap unmatched dots drawn per still so overlays stay readable.
DEFAULT_MAX_ORPHANS_PER_MATCH = 120
# Sync residual weight for auto tracks (between Low=0.25 and Normal=1.0).
AUTO_FEATURE_SYNC_WEIGHT = 0.35

DETECTOR_ITEMS = (
    ("ORB", "ORB", "Fast binary features — default for still matching"),
    ("SIFT", "SIFT", "Slower float descriptors; sometimes stronger under scale change"),
)


class FeatureDetectDependencyError(RuntimeError):
    """Raised when OpenCV is unavailable for feature detection."""


@dataclass(frozen=True)
class TrackObservationData:
    """One 2D pick for an auto-track, in source-image pixels (top-left origin)."""

    match_id: str
    x: float
    y: float
    # Keypoint index inside that still's detection list (orphan exclusion).
    keypoint_index: int = -1


@dataclass
class TrackData:
    """One auto feature track ready to write into RNA."""

    track_id: str
    observations: list[TrackObservationData]
    # Lower is better (mean RANSAC / Sampson residual over supporting pairs).
    residual_px: float = 0.0
    multi_view: bool = False


@dataclass
class FeatureJobResult:
    """Outcome of a detect+match job across several stills."""

    tracks: list[TrackData] = field(default_factory=list)
    detected_by_match: dict[str, int] = field(default_factory=dict)
    pair_inliers: int = 0
    ratio_matches: int = 0
    multi_view_count: int = 0
    orphan_count: int = 0
    filtered_out: int = 0
    message: str = ""


@dataclass(frozen=True)
class ImageSource:
    """Still input for a background job — prefer on-disk path so the UI stays free."""

    match_id: str
    image_path: str = ""
    # Used only when ``image_path`` cannot be read (Blender buffer, loaded on main).
    gray: object | None = None


@dataclass
class _ImageFeatures:
    match_id: str
    points: np.ndarray  # (N, 2) float64
    descriptors: np.ndarray
    responses: np.ndarray  # (N,) float64


def _import_cv2():
    """Import OpenCV; raise FeatureDetectDependencyError if missing."""
    try:
        import cv2
    except ImportError as error:
        raise FeatureDetectDependencyError(
            "Auto features need the bundled OpenCV wheel. "
            "Run ./fetch-wheels.sh, then disable and re-enable Perspective Match "
            "(or install a fresh platform zip from ./build-extension.sh)."
        ) from error
    return cv2


def _gray_from_path(cv2_module, image_path: str) -> np.ndarray | None:
    """Load a grayscale image from disk; None if the path cannot be read."""
    path = Path(image_path).expanduser()
    if not path.is_file():
        return None
    bgr = cv2_module.imread(str(path), cv2_module.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2_module.cvtColor(bgr, cv2_module.COLOR_BGR2GRAY)


def _gray_from_blender_image(cv2_module, image) -> np.ndarray:
    """Convert a Blender Image to grayscale uint8 (top-left origin)."""
    from . import distortion

    rgba = distortion._image_pixels_top_left(image)
    rgb_u8 = np.clip(np.round(rgba[:, :, :3] * 255.0), 0, 255).astype(np.uint8)
    return cv2_module.cvtColor(rgb_u8, cv2_module.COLOR_RGB2GRAY)


def load_match_gray(settings) -> np.ndarray:
    """Prefer the on-disk still; fall back to Blender's pixel buffer."""
    cv2 = _import_cv2()
    image_path = getattr(settings, "image_path", "") or ""
    if image_path:
        from_disk = _gray_from_path(cv2, image_path)
        if from_disk is not None:
            return from_disk
    image = getattr(settings, "image", None)
    if image is None:
        raise ValueError("Match has no reference image")
    return _gray_from_blender_image(cv2, image)


def _create_detector(cv2_module, detector: str, max_features: int):
    """Build an OpenCV Feature2D detector/descriptor."""
    name = (detector or DEFAULT_DETECTOR).upper()
    count = max(int(max_features), 50)
    if name == "SIFT":
        if not hasattr(cv2_module, "SIFT_create"):
            raise FeatureDetectDependencyError("This OpenCV build has no SIFT_create")
        return cv2_module.SIFT_create(nfeatures=count), cv2_module.NORM_L2
    if not hasattr(cv2_module, "ORB_create"):
        raise FeatureDetectDependencyError("This OpenCV build has no ORB_create")
    return (
        cv2_module.ORB_create(nfeatures=count, scaleFactor=1.2, nlevels=8),
        cv2_module.NORM_HAMMING,
    )


def detect_features(
    gray: np.ndarray,
    *,
    match_id: str,
    detector: str = DEFAULT_DETECTOR,
    max_features: int = DEFAULT_MAX_FEATURES,
) -> _ImageFeatures:
    """Detect keypoints + descriptors in a grayscale still."""
    cv2 = _import_cv2()
    if gray.ndim != 2:
        raise ValueError("Detection expects a single-channel grayscale image")
    feature2d, _norm = _create_detector(cv2, detector, max_features)
    keypoints, descriptors = feature2d.detectAndCompute(gray, None)
    if not keypoints or descriptors is None or len(keypoints) == 0:
        return _ImageFeatures(
            match_id=match_id,
            points=np.zeros((0, 2), dtype=np.float64),
            descriptors=np.zeros((0, 1), dtype=np.uint8),
            responses=np.zeros((0,), dtype=np.float64),
        )
    points = np.asarray([kp.pt for kp in keypoints], dtype=np.float64)
    responses = np.asarray([float(kp.response) for kp in keypoints], dtype=np.float64)
    return _ImageFeatures(
        match_id=match_id,
        points=points,
        descriptors=np.asarray(descriptors),
        responses=responses,
    )


def _ratio_matches(
    cv2_module,
    descriptors_a: np.ndarray,
    descriptors_b: np.ndarray,
    norm_type: int,
    ratio: float,
) -> list[tuple[int, int, float]]:
    """Lowe ratio test; returns (index_a, index_b, distance)."""
    if (
        descriptors_a is None
        or descriptors_b is None
        or len(descriptors_a) < 2
        or len(descriptors_b) < 2
    ):
        return []
    matcher = cv2_module.BFMatcher(norm_type, crossCheck=False)
    raw = matcher.knnMatch(descriptors_a, descriptors_b, k=2)
    good: list[tuple[int, int, float]] = []
    for pair in raw:
        if len(pair) < 2:
            continue
        best, second = pair
        if best.distance < float(ratio) * second.distance:
            good.append((int(best.queryIdx), int(best.trainIdx), float(best.distance)))
    return good


def _ransac_fundamental(
    cv2_module,
    points_a: np.ndarray,
    points_b: np.ndarray,
    threshold_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fundamental-matrix RANSAC; returns (keep_mask bool N, epipolar residual)."""
    count = int(points_a.shape[0])
    empty = (
        np.zeros((count,), dtype=bool),
        np.full((count,), np.inf, dtype=np.float64),
    )
    if count < 8:
        return empty
    matrix, mask = cv2_module.findFundamentalMat(
        points_a.astype(np.float32),
        points_b.astype(np.float32),
        cv2_module.FM_RANSAC,
        float(threshold_px),
        0.99,
    )
    if mask is None or matrix is None:
        return empty
    keep = np.asarray(mask).reshape(-1).astype(bool)
    residuals = np.full((count,), np.inf, dtype=np.float64)
    try:
        lines_b = cv2_module.computeCorrespondEpilines(
            points_a.reshape(-1, 1, 2).astype(np.float32), 1, matrix
        ).reshape(-1, 3)
        lines_a = cv2_module.computeCorrespondEpilines(
            points_b.reshape(-1, 1, 2).astype(np.float32), 2, matrix
        ).reshape(-1, 3)
        num_b = np.abs(
            lines_b[:, 0] * points_b[:, 0]
            + lines_b[:, 1] * points_b[:, 1]
            + lines_b[:, 2]
        )
        den_b = np.sqrt(lines_b[:, 0] ** 2 + lines_b[:, 1] ** 2) + 1.0e-12
        num_a = np.abs(
            lines_a[:, 0] * points_a[:, 0]
            + lines_a[:, 1] * points_a[:, 1]
            + lines_a[:, 2]
        )
        den_a = np.sqrt(lines_a[:, 0] ** 2 + lines_a[:, 1] ** 2) + 1.0e-12
        residuals = 0.5 * (num_a / den_a + num_b / den_b)
    except Exception:
        residuals = np.where(keep, float(threshold_px), np.inf)
    return keep, residuals


def _ransac_homography(
    cv2_module,
    points_a: np.ndarray,
    points_b: np.ndarray,
    threshold_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Homography RANSAC — works for pure rotation / near-planar stills where F fails."""
    count = int(points_a.shape[0])
    empty = (
        np.zeros((count,), dtype=bool),
        np.full((count,), np.inf, dtype=np.float64),
    )
    if count < 4:
        return empty
    matrix, mask = cv2_module.findHomography(
        points_a.astype(np.float32),
        points_b.astype(np.float32),
        cv2_module.RANSAC,
        float(threshold_px),
    )
    if mask is None or matrix is None:
        return empty
    keep = np.asarray(mask).reshape(-1).astype(bool)
    # Symmetric reprojection error in pixels.
    ones = np.ones((count, 1), dtype=np.float64)
    homo_a = np.hstack([points_a.astype(np.float64), ones])
    homo_b = np.hstack([points_b.astype(np.float64), ones])
    projected_b = (matrix @ homo_a.T).T
    projected_b = projected_b[:, :2] / np.maximum(projected_b[:, 2:3], 1.0e-12)
    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        residuals = np.linalg.norm(projected_b - points_b, axis=1)
        return keep, residuals
    projected_a = (inverse @ homo_b.T).T
    projected_a = projected_a[:, :2] / np.maximum(projected_a[:, 2:3], 1.0e-12)
    residuals = 0.5 * (
        np.linalg.norm(projected_b - points_b, axis=1)
        + np.linalg.norm(projected_a - points_a, axis=1)
    )
    return keep, residuals


def _norm_type_for_detector(cv2_module, detector: str) -> int:
    """Hamming for ORB binary descriptors; L2 for SIFT."""
    name = (detector or DEFAULT_DETECTOR).upper()
    if name == "SIFT":
        return cv2_module.NORM_L2
    return cv2_module.NORM_HAMMING


def match_feature_pair(
    features_a: _ImageFeatures,
    features_b: _ImageFeatures,
    *,
    detector: str = DEFAULT_DETECTOR,
    ratio: float = DEFAULT_RATIO,
    ransac_px: float = DEFAULT_RANSAC_PX,
) -> tuple[list[tuple[int, int, float]], int]:
    """Match two stills; return (inliers, ratio_match_count).

    Tries fundamental-matrix and homography RANSAC and keeps the stronger model
    (homography helps when cameras barely translate or the scene is near-planar).
    """
    cv2 = _import_cv2()
    norm_type = _norm_type_for_detector(cv2, detector)
    candidates = _ratio_matches(
        cv2,
        features_a.descriptors,
        features_b.descriptors,
        norm_type,
        ratio,
    )
    ratio_count = len(candidates)
    if ratio_count < 4:
        return [], ratio_count
    index_a = np.asarray([item[0] for item in candidates], dtype=np.int32)
    index_b = np.asarray([item[1] for item in candidates], dtype=np.int32)
    points_a = features_a.points[index_a]
    points_b = features_b.points[index_b]

    keep_f, res_f = _ransac_fundamental(cv2, points_a, points_b, ransac_px)
    keep_h, res_h = _ransac_homography(cv2, points_a, points_b, ransac_px)
    count_f = int(np.count_nonzero(keep_f))
    count_h = int(np.count_nonzero(keep_h))
    # Prefer the model with more inliers; on a tie, lower median residual wins.
    use_homography = count_h > count_f
    if count_h == count_f and count_h > 0:
        med_h = float(np.median(res_h[keep_h])) if count_h else np.inf
        med_f = float(np.median(res_f[keep_f])) if count_f else np.inf
        use_homography = med_h <= med_f
    keep = keep_h if use_homography else keep_f
    residuals = res_h if use_homography else res_f

    inliers: list[tuple[int, int, float]] = []
    for offset, alive in enumerate(keep):
        if not alive:
            continue
        inliers.append(
            (
                int(index_a[offset]),
                int(index_b[offset]),
                float(residuals[offset]),
            )
        )
    return inliers, ratio_count


class _UnionFind:
    """Disjoint-set for merging keypoints into multi-view tracks."""

    def __init__(self) -> None:
        self._parent: dict[tuple[str, int], tuple[str, int]] = {}
        self._rank: dict[tuple[str, int], int] = {}

    def add(self, node: tuple[str, int]) -> None:
        if node not in self._parent:
            self._parent[node] = node
            self._rank[node] = 0

    def find(self, node: tuple[str, int]) -> tuple[str, int]:
        parent = self._parent[node]
        if parent != node:
            self._parent[node] = self.find(parent)
        return self._parent[node]

    def union(self, left: tuple[str, int], right: tuple[str, int]) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self._rank[root_left] < self._rank[root_right]:
            self._parent[root_left] = root_right
        elif self._rank[root_left] > self._rank[root_right]:
            self._parent[root_right] = root_left
        else:
            self._parent[root_right] = root_left
            self._rank[root_left] += 1


def _filter_multi_view_tracks(
    tracks: list[TrackData],
    *,
    keep_percentile: float,
    max_multi_view: int,
) -> tuple[list[TrackData], int]:
    """Keep the better multi-view tracks by residual; return (kept, dropped)."""
    multi = [track for track in tracks if track.multi_view]
    orphans = [track for track in tracks if not track.multi_view]
    if not multi:
        return tracks, 0

    residuals = np.asarray([track.residual_px for track in multi], dtype=np.float64)
    finite = residuals[np.isfinite(residuals)]
    if finite.size == 0:
        cutoff = np.inf
    else:
        cutoff = float(np.percentile(finite, float(keep_percentile)))
    survivors = [
        track
        for track in multi
        if np.isfinite(track.residual_px) and track.residual_px <= cutoff + 1.0e-9
    ]
    survivors.sort(key=lambda item: item.residual_px)
    if max_multi_view > 0 and len(survivors) > int(max_multi_view):
        survivors = survivors[: int(max_multi_view)]
    dropped = len(multi) - len(survivors)
    return survivors + orphans, dropped


def resolve_image_source(source: ImageSource) -> np.ndarray:
    """Load grayscale for a job source (disk preferred; else preloaded array)."""
    cv2 = _import_cv2()
    path = (source.image_path or "").strip()
    if path:
        from_disk = _gray_from_path(cv2, path)
        if from_disk is not None:
            return from_disk
    if source.gray is not None:
        gray = np.asarray(source.gray)
        if gray.ndim == 2:
            return gray
    raise ValueError(f"No readable image for match '{source.match_id}'")


def build_tracks_from_sources(
    sources: list[ImageSource],
    *,
    detector: str = DEFAULT_DETECTOR,
    max_features: int = DEFAULT_MAX_FEATURES,
    ratio: float = DEFAULT_RATIO,
    ransac_px: float = DEFAULT_RANSAC_PX,
    keep_percentile: float = DEFAULT_KEEP_PERCENTILE,
    max_multi_view: int = DEFAULT_MAX_MULTI_VIEW,
    max_orphans_per_match: int = DEFAULT_MAX_ORPHANS_PER_MATCH,
    cancel_check=None,
    progress_callback=None,
) -> FeatureJobResult:
    """Load sources (disk I/O in the worker), then detect/match/build tracks."""

    def cancelled() -> bool:
        return bool(cancel_check and cancel_check())

    def progress(step: int, total: int, label: str) -> None:
        if progress_callback is not None:
            progress_callback(step, total, label)

    if len(sources) < 1:
        return FeatureJobResult(message="No stills to scan")

    load_steps = len(sources)
    images: list[tuple[str, np.ndarray]] = []
    for index, source in enumerate(sources):
        if cancelled():
            return FeatureJobResult(message="Auto feature detection cancelled")
        progress(index, load_steps + 1, f"Load {source.match_id}")
        images.append((source.match_id, resolve_image_source(source)))

    # Offset progress so detect/match continue after load steps.
    def _shifted_progress(step: int, total: int, label: str) -> None:
        progress(load_steps + step, load_steps + total, label)

    return build_tracks_from_images(
        images,
        detector=detector,
        max_features=max_features,
        ratio=ratio,
        ransac_px=ransac_px,
        keep_percentile=keep_percentile,
        max_multi_view=max_multi_view,
        max_orphans_per_match=max_orphans_per_match,
        cancel_check=cancel_check,
        progress_callback=_shifted_progress,
    )


def build_tracks_from_images(
    images: list[tuple[str, np.ndarray]],
    *,
    detector: str = DEFAULT_DETECTOR,
    max_features: int = DEFAULT_MAX_FEATURES,
    ratio: float = DEFAULT_RATIO,
    ransac_px: float = DEFAULT_RANSAC_PX,
    keep_percentile: float = DEFAULT_KEEP_PERCENTILE,
    max_multi_view: int = DEFAULT_MAX_MULTI_VIEW,
    max_orphans_per_match: int = DEFAULT_MAX_ORPHANS_PER_MATCH,
    cancel_check=None,
    progress_callback=None,
) -> FeatureJobResult:
    """Detect features in each still, match all pairs, build filtered tracks.

    ``images`` is ``(match_id, gray_uint8)``. Prefer ``build_tracks_from_sources``
    so disk loads happen off the Blender main thread.
    ``cancel_check`` is an optional zero-arg callable returning True to abort.
    ``progress_callback(step, total, label)`` is optional.
    """

    def cancelled() -> bool:
        return bool(cancel_check and cancel_check())

    def progress(step: int, total: int, label: str) -> None:
        if progress_callback is not None:
            progress_callback(step, total, label)

    if len(images) < 1:
        return FeatureJobResult(message="No stills to scan")

    pair_count = len(images) * (len(images) - 1) // 2
    total_steps = max(len(images) + pair_count + 1, 1)
    step = 0

    features_by_id: dict[str, _ImageFeatures] = {}
    detected_by_match: dict[str, int] = {}
    for match_id, gray in images:
        if cancelled():
            return FeatureJobResult(message="Auto feature detection cancelled")
        progress(step, total_steps, f"Detect {match_id}")
        features = detect_features(
            gray,
            match_id=match_id,
            detector=detector,
            max_features=max_features,
        )
        features_by_id[match_id] = features
        detected_by_match[match_id] = int(features.points.shape[0])
        step += 1

    union = _UnionFind()
    # Residual samples per keypoint node, used to score merged tracks.
    residual_samples: dict[tuple[str, int], list[float]] = {}
    pair_inliers = 0
    ratio_matches_total = 0
    match_ids = [match_id for match_id, _gray in images]

    for left_index, left_id in enumerate(match_ids):
        for right_id in match_ids[left_index + 1 :]:
            if cancelled():
                return FeatureJobResult(message="Auto feature detection cancelled")
            progress(step, total_steps, f"Match {left_id}↔{right_id}")
            step += 1
            left = features_by_id[left_id]
            right = features_by_id[right_id]
            if left.points.shape[0] < 4 or right.points.shape[0] < 4:
                continue
            inliers, ratio_count = match_feature_pair(
                left,
                right,
                detector=detector,
                ratio=ratio,
                ransac_px=ransac_px,
            )
            ratio_matches_total += int(ratio_count)
            pair_inliers += len(inliers)
            for index_a, index_b, residual in inliers:
                node_a = (left_id, int(index_a))
                node_b = (right_id, int(index_b))
                union.add(node_a)
                union.add(node_b)
                union.union(node_a, node_b)
                residual_samples.setdefault(node_a, []).append(float(residual))
                residual_samples.setdefault(node_b, []).append(float(residual))

    progress(step, total_steps, "Build tracks")
    # Group nodes by root.
    components: dict[tuple[str, int], list[tuple[str, int]]] = {}
    for node in list(residual_samples.keys()):
        root = union.find(node)
        components.setdefault(root, []).append(node)

    multi_tracks: list[TrackData] = []
    for nodes in components.values():
        # One observation per match — prefer the strongest response if duplicates.
        by_match: dict[str, tuple[str, int]] = {}
        for match_id, kp_index in nodes:
            features = features_by_id[match_id]
            response = float(features.responses[kp_index])
            previous = by_match.get(match_id)
            if previous is None:
                by_match[match_id] = (match_id, kp_index)
                continue
            prev_response = float(features_by_id[previous[0]].responses[previous[1]])
            if response > prev_response:
                by_match[match_id] = (match_id, kp_index)
        if len(by_match) < 2:
            continue
        observations: list[TrackObservationData] = []
        sample_residuals: list[float] = []
        for match_id, kp_index in by_match.values():
            point = features_by_id[match_id].points[kp_index]
            observations.append(
                TrackObservationData(
                    match_id=match_id,
                    x=float(point[0]),
                    y=float(point[1]),
                    keypoint_index=int(kp_index),
                )
            )
            sample_residuals.extend(residual_samples.get((match_id, kp_index), []))
        residual = (
            float(np.median(sample_residuals)) if sample_residuals else float("inf")
        )
        multi_tracks.append(
            TrackData(
                track_id=f"auto-{uuid4().hex}",
                observations=observations,
                residual_px=residual,
                multi_view=True,
            )
        )

    filtered, filtered_out = _filter_multi_view_tracks(
        multi_tracks,
        keep_percentile=keep_percentile,
        max_multi_view=max_multi_view,
    )
    kept_multi = [track for track in filtered if track.multi_view]
    matched_nodes: set[tuple[str, int]] = set()
    for track in kept_multi:
        for observation in track.observations:
            if observation.keypoint_index >= 0:
                matched_nodes.add((observation.match_id, observation.keypoint_index))

    orphan_tracks: list[TrackData] = []
    for match_id, features in features_by_id.items():
        candidates: list[tuple[float, int]] = []
        for index in range(features.points.shape[0]):
            if (match_id, index) in matched_nodes:
                continue
            candidates.append((float(features.responses[index]), index))
        candidates.sort(reverse=True)
        for _response, index in candidates[: max(0, int(max_orphans_per_match))]:
            point = features.points[index]
            orphan_tracks.append(
                TrackData(
                    track_id=f"auto-{uuid4().hex}",
                    observations=[
                        TrackObservationData(
                            match_id=match_id,
                            x=float(point[0]),
                            y=float(point[1]),
                            keypoint_index=int(index),
                        )
                    ],
                    residual_px=0.0,
                    multi_view=False,
                )
            )

    tracks = kept_multi + orphan_tracks
    multi_view_count = len(kept_multi)
    orphan_count = len(orphan_tracks)
    total_detected = sum(detected_by_match.values())
    message = (
        f"Auto features · {total_detected} detected · "
        f"{multi_view_count} multi-view · {orphan_count} orphans"
    )
    if filtered_out:
        message += f" · filtered {filtered_out}"
    if ratio_matches_total:
        message += f" · {ratio_matches_total} raw matches"
    if pair_inliers:
        message += f" · {pair_inliers} inliers"
    elif total_detected > 0 and len(images) >= 2:
        message += " · no overlap (try more features / check stills share the scene)"

    return FeatureJobResult(
        tracks=tracks,
        detected_by_match=detected_by_match,
        pair_inliers=pair_inliers,
        ratio_matches=ratio_matches_total,
        multi_view_count=multi_view_count,
        orphan_count=orphan_count,
        filtered_out=filtered_out,
        message=message,
    )


def apply_feature_tracks(context, result: FeatureJobResult) -> FeatureJobResult:
    """Replace workspace auto-tracks with ``result.tracks`` (main thread only)."""
    from . import properties

    space = properties.workspace(context)
    roots_by_name = {
        root.name: root
        for root in properties.iter_match_roots()
    }
    space.auto_tracks.clear()
    for track in result.tracks:
        item = space.auto_tracks.add()
        item.track_id = track.track_id
        item.multi_view = bool(track.multi_view)
        item.residual_px = float(track.residual_px)
        item.use_in_sync = bool(track.multi_view)
        for observation in track.observations:
            root = roots_by_name.get(observation.match_id)
            if root is None:
                continue
            row = item.observations.add()
            row.match_root = root
            row.match_name = root.name
            row.x = float(observation.x)
            row.y = float(observation.y)
            row.is_set = True

    space.auto_feature_status = result.message
    properties.tag_viewport_redraw(context)
    return result


def clear_auto_tracks(context) -> int:
    """Remove all auto-tracks; return how many were cleared."""
    from . import properties

    space = properties.workspace(context)
    count = len(space.auto_tracks)
    space.auto_tracks.clear()
    space.auto_feature_status = "Auto features cleared" if count else ""
    properties.tag_viewport_redraw(context)
    return count


def auto_track_observation_for_match(track, root):
    """Return the observation row for ``root``, or None."""
    if root is None:
        return None
    root_name = getattr(root, "name", "") or ""
    for observation in track.observations:
        if not observation.is_set:
            continue
        match = observation.match_root
        if match is not None and (match == root or match.name == root_name):
            return observation
        # Fallback when the pointer was not written / cleared.
        stored_name = getattr(observation, "match_name", "") or ""
        if stored_name and stored_name == root_name:
            return observation
    return None


def count_auto_track_observations(track) -> int:
    """How many stills have a set pick on this track."""
    return sum(1 for observation in track.observations if observation.is_set)
