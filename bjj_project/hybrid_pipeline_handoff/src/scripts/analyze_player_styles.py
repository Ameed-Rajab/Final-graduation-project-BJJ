#!/usr/bin/env python3
"""
Sequence-aware player style analysis for BJJ position predictions.

The script reads the `positions.csv` and `summary.json` outputs from the
position-classification pipeline, then builds:
  - per-player style profiles
  - sequence-based flow analysis from the order of positions
  - matchup-specific counter recommendations

Compared with a simple frequency counter, this version looks at:
  - recurring transitions
  - 3-step chains
  - long control phases
  - sequence motifs such as guard->turtle->attack or side_control->mount/back

Example:
  python3 src/scripts/analyze_player_styles.py \
    --position-dir outputs/detr_ioutrack_test_images/position_predictions
"""

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_POSITION_DIR = Path("outputs/detr_ioutrack_test_images/position_predictions")
DEFAULT_FPS = 30.0

GUARD_FAMILIES = {"open_guard", "closed_guard", "half_guard"}
TOP_FAMILIES = {"side_control", "mount", "back", "takedown"}
FINISH_FAMILIES = {"mount", "back"}

# --- Role model -----------------------------------------------------------
# After enforce_bottom_2, a label's 1/2 suffix encodes WHICH tracked athlete is
# on the bottom (suffix 2 = bottom, suffix 1 = top). Symmetric families carry no
# role. We use this to credit BOTH athletes their role in every frame, so the two
# players get genuinely different profiles (a passer vs a guard player).
SYMMETRIC_FAMILIES = {"standing", "5050_guard"}          # no top/bottom role
DOMINANT_TOP_FAMILIES = {"side_control", "mount", "back"}  # clear top control
ROLE_FAMILIES = {
    "open_guard", "closed_guard", "half_guard",
    "side_control", "mount", "back", "takedown", "turtle",
}
GUARD_DISPLAY = {
    "open_guard": "open guard",
    "closed_guard": "closed guard",
    "half_guard": "half guard",
}


def role_for(family: str, bottom_id: Optional[int], player: int) -> str:
    """Return 'top' | 'bottom' | 'neutral' for `player` in a frame.

    bottom_id is the athlete on the bottom (the label suffix). The other athlete
    is on top. Symmetric families (standing, 50/50) have no role.
    """
    if family in SYMMETRIC_FAMILIES:
        return "neutral"
    if bottom_id is None:
        return "neutral"
    return "bottom" if int(bottom_id) == int(player) else "top"


def role_label(family: str, role: str) -> str:
    """Human label for a (family, role) pair, e.g. 'half guard (passing)'."""
    base = FAMILY_DISPLAY.get(family, family.replace("_", " "))
    if role == "neutral" or family in SYMMETRIC_FAMILIES:
        return base
    if family in GUARD_FAMILIES:
        return f"{base} (passing)" if role == "top" else f"{base} (bottom)"
    if family in DOMINANT_TOP_FAMILIES:
        return f"{base} (top)" if role == "top" else f"{base} (defending)"
    if family == "takedown":
        return "takedown (finishing)" if role == "top" else "takedown (defending)"
    if family == "turtle":
        return "turtle (attacking)" if role == "top" else "turtle (defending)"
    return base

# Maps a label column to the matching confidence column in positions.csv.
CONF_COLUMN_FOR = {
    "pred": "pred_conf",
    "pred_smooth": "pred_smooth_conf",
    "pred_smooth_raw": "pred_smooth_conf",
    "pred_fixed": "pred_smooth_conf",
}

FAMILY_DISPLAY = {
    "open_guard": "Open guard",
    "closed_guard": "Closed guard",
    "half_guard": "Half guard",
    "side_control": "Side control",
    "mount": "Mount",
    "back": "Back control",
    "takedown": "Takedown",
    "turtle": "Turtle / scramble",
    "5050_guard": "50/50 guard",
    "standing": "Standing",
}

STANDING_TO_TAKEDOWN = {("standing", "takedown")}
TAKEDOWN_TO_CONTROL = {("takedown", fam) for fam in ("side_control", "mount", "back")}
TAKEDOWN_TO_RESET = {("takedown", fam) for fam in ("open_guard", "closed_guard", "half_guard", "turtle")}
GUARD_TO_TURTLE = {(fam, "turtle") for fam in GUARD_FAMILIES}
TURTLE_TO_GUARD = {("turtle", fam) for fam in GUARD_FAMILIES}
TURTLE_TO_ATTACK = {("turtle", fam) for fam in ("side_control", "mount", "back")}
PASS_TO_FINISH = {("side_control", fam) for fam in ("mount", "back")}
FINISH_CYCLE = {("mount", "back"), ("back", "mount")}
GUARD_REENTRY = {(fam, fam) for fam in GUARD_FAMILIES}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def natural_sort_key(text: str) -> List[object]:
    parts = re.split(r"(\d+)", text)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def family_name(family: str) -> str:
    return FAMILY_DISPLAY.get(family, family.replace("_", " ").title())


def format_sequence(sequence: Sequence[str]) -> str:
    return " -> ".join(family_name(item) for item in sequence)


def pct(value: float) -> float:
    return round(100.0 * value, 1)


def fmt_count(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.1f}"


def frames_to_seconds(rows: float, fps: float) -> float:
    if fps <= 0:
        return 0.0
    return round(float(rows) / float(fps), 2)


def format_seconds(seconds: float) -> str:
    if seconds <= 0:
        return "0s"
    if seconds >= 60:
        minutes = int(seconds // 60)
        rem = seconds - minutes * 60
        return f"{minutes}m{rem:04.1f}s"
    return f"{seconds:.1f}s"


def control_rank(family: str, owner: Optional[int], player: int) -> float:
    """Approximate BJJ control rank for `player` given a labeled position.

    Higher = `player` is in a more dominant position. Negative = dominated.
    """
    if family == "standing" or family == "5050_guard":
        return 1.0
    if owner is None:
        return 1.0
    is_self = (owner == player)
    if family in {"back", "mount"}:
        return 5.0 if is_self else -2.0
    if family == "side_control":
        return 4.0 if is_self else -1.0
    if family == "takedown":
        return 3.0 if is_self else 0.0
    if family == "turtle":
        return 0.0 if is_self else 3.0
    if family in {"open_guard", "closed_guard", "half_guard"}:
        return 0.0 if is_self else 2.0
    return 1.0


def analyze_advance_concession(
    label_segments: Sequence[Dict[str, object]],
    player: int,
) -> Dict[str, float]:
    advances = 0
    concessions = 0
    deltas: List[float] = []
    biggest_advance = 0.0
    biggest_concession = 0.0

    for prev, curr in zip(label_segments, label_segments[1:]):
        prev_owner = prev.get("player_id")
        curr_owner = curr.get("player_id")
        before = control_rank(str(prev["family"]), prev_owner, player)
        after = control_rank(str(curr["family"]), curr_owner, player)
        delta = after - before
        if abs(delta) < 1e-9:
            continue
        deltas.append(delta)
        if delta > 0:
            advances += 1
            if delta > biggest_advance:
                biggest_advance = delta
        else:
            concessions += 1
            if delta < biggest_concession:
                biggest_concession = delta

    return {
        "advances": advances,
        "concessions": concessions,
        "net_progression": round(float(sum(deltas)), 2),
        "biggest_advance": round(biggest_advance, 2),
        "biggest_concession": round(biggest_concession, 2),
    }


def filter_by_confidence(
    rows: Sequence[Dict[str, str]],
    conf_column: Optional[str],
    min_conf: float,
) -> Tuple[List[Dict[str, str]], int]:
    if min_conf <= 0 or not conf_column or not rows or conf_column not in rows[0]:
        return list(rows), 0
    kept: List[Dict[str, str]] = []
    dropped = 0
    for row in rows:
        raw = row.get(conf_column, "")
        try:
            value = float(raw) if raw not in (None, "") else 0.0
        except ValueError:
            value = 0.0
        if value >= min_conf:
            kept.append(row)
        else:
            dropped += 1
    return kept, dropped


def resolve_relative_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidate = base_dir / path
    if candidate.exists():
        return candidate
    return path


def resolve_input_paths(
    position_dir: Optional[str],
    summary_json: Optional[str],
    positions_csv: Optional[str],
) -> Tuple[Path, Path, Path]:
    if position_dir:
        base_dir = Path(position_dir)
    elif summary_json:
        base_dir = Path(summary_json).resolve().parent
    elif positions_csv:
        base_dir = Path(positions_csv).resolve().parent
    elif DEFAULT_POSITION_DIR.is_dir():
        base_dir = DEFAULT_POSITION_DIR
    else:
        raise FileNotFoundError(
            "Could not find input files. Pass --position-dir or --summary-json / --positions-csv."
        )

    summary_path = Path(summary_json) if summary_json else base_dir / "summary.json"
    csv_path = Path(positions_csv) if positions_csv else base_dir / "positions.csv"

    if not summary_path.is_file():
        raise FileNotFoundError(f"summary.json not found: {summary_path}")

    with summary_path.open("r", encoding="utf-8") as handle:
        summary_data = json.load(handle)

    if not csv_path.is_file():
        csv_ref = summary_data.get("csv")
        if csv_ref:
            csv_path = resolve_relative_path(summary_path.parent, str(csv_ref))

    if not csv_path.is_file():
        raise FileNotFoundError(f"positions.csv not found: {csv_path}")

    return base_dir, summary_path, csv_path


def load_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    def frame_key(row: Dict[str, str]) -> Tuple[int, List[object]]:
        frame_idx = row.get("frame_idx", "").strip()
        if frame_idx:
            try:
                return int(frame_idx), natural_sort_key(row.get("frame_file", ""))
            except ValueError:
                pass
        return 10**12, natural_sort_key(row.get("frame_file", ""))

    rows.sort(key=frame_key)
    return rows


def choose_label_column(rows: Sequence[Dict[str, str]], preferred: Optional[str]) -> str:
    if not rows:
        raise RuntimeError("positions.csv is empty.")

    columns = set(rows[0].keys())
    if preferred:
        if preferred not in columns:
            raise ValueError(f"Requested label column '{preferred}' not found in CSV.")
        return preferred

    for candidate in ("pred_fixed", "pred_smooth", "pred_smooth_raw", "pred"):
        if candidate in columns:
            return candidate

    raise ValueError(
        "Could not find a label column. Expected one of: pred_fixed, pred_smooth, pred_smooth_raw, pred."
    )


def parse_label(label: str) -> Tuple[str, Optional[int]]:
    text = str(label).strip()
    match = re.match(r"^(.*?)([12])$", text)
    if match:
        return match.group(1), int(match.group(2))
    return text, None


def row_frame_idx(row: Dict[str, str], fallback: int) -> int:
    value = row.get("frame_idx", "").strip()
    try:
        return int(value)
    except ValueError:
        return fallback


def top_items(counter: Counter, limit: int = 4) -> List[Tuple[str, float]]:
    items = [(key, float(val)) for key, val in counter.items() if float(val) > 0]
    items.sort(key=lambda item: (-item[1], item[0]))
    return items[:limit]


def dominant_family(counter: Counter, families: Iterable[str]) -> Optional[str]:
    candidates = [(fam, float(counter.get(fam, 0.0))) for fam in families]
    candidates = [item for item in candidates if item[1] > 0]
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[1], item[0]))
    return candidates[0][0]


def count_pairs(values: Sequence[str]) -> Counter:
    return Counter(zip(values, values[1:]))


def count_triples(values: Sequence[str]) -> Counter:
    return Counter(zip(values, values[1:], values[2:]))


def count_patterns(counter: Counter, patterns: Iterable[Tuple[str, str]]) -> int:
    return int(sum(int(counter.get(pattern, 0)) for pattern in patterns))


def serialize_pair_counts(counter: Counter, limit: int = 5) -> List[Dict[str, object]]:
    return [
        {
            "pattern": list(pattern),
            "label": format_sequence(pattern),
            "count": int(count),
        }
        for pattern, count in counter.most_common(limit)
    ]


def serialize_triple_counts(counter: Counter, limit: int = 5) -> List[Dict[str, object]]:
    return [
        {
            "pattern": list(pattern),
            "label": format_sequence(pattern),
            "count": int(count),
        }
        for pattern, count in counter.most_common(limit)
    ]


def build_label_segments(rows: Sequence[Dict[str, str]], label_column: str) -> List[Dict[str, object]]:
    if not rows:
        return []

    segments: List[Dict[str, object]] = []
    current_label = str(rows[0][label_column]).strip()
    current_family, current_pid = parse_label(current_label)
    start_row = 0
    start_frame = row_frame_idx(rows[0], 0)

    for idx in range(1, len(rows)):
        label = str(rows[idx][label_column]).strip()
        if label == current_label:
            continue

        prev_frame = row_frame_idx(rows[idx - 1], idx - 1)
        segments.append(
            {
                "label": current_label,
                "family": current_family,
                "player_id": current_pid,
                "start_row": start_row,
                "end_row": idx - 1,
                "start_frame": start_frame,
                "end_frame": prev_frame,
                "duration_rows": idx - start_row,
            }
        )
        current_label = label
        current_family, current_pid = parse_label(current_label)
        start_row = idx
        start_frame = row_frame_idx(rows[idx], idx)

    prev_frame = row_frame_idx(rows[-1], len(rows) - 1)
    segments.append(
        {
            "label": current_label,
            "family": current_family,
            "player_id": current_pid,
            "start_row": start_row,
            "end_row": len(rows) - 1,
            "start_frame": start_frame,
            "end_frame": prev_frame,
            "duration_rows": len(rows) - start_row,
        }
    )
    return segments


def build_family_segments(rows: Sequence[Dict[str, str]], label_column: str) -> List[Dict[str, object]]:
    if not rows:
        return []

    first_family, _ = parse_label(rows[0][label_column])
    current_family = first_family
    start_row = 0
    start_frame = row_frame_idx(rows[0], 0)
    segments: List[Dict[str, object]] = []

    for idx in range(1, len(rows)):
        family, _ = parse_label(rows[idx][label_column])
        if family == current_family:
            continue

        prev_frame = row_frame_idx(rows[idx - 1], idx - 1)
        segments.append(
            {
                "family": current_family,
                "start_row": start_row,
                "end_row": idx - 1,
                "start_frame": start_frame,
                "end_frame": prev_frame,
                "duration_rows": idx - start_row,
            }
        )
        current_family = family
        start_row = idx
        start_frame = row_frame_idx(rows[idx], idx)

    prev_frame = row_frame_idx(rows[-1], len(rows) - 1)
    segments.append(
        {
            "family": current_family,
            "start_row": start_row,
            "end_row": len(rows) - 1,
            "start_frame": start_frame,
            "end_frame": prev_frame,
            "duration_rows": len(rows) - start_row,
        }
    )
    return segments


def detect_reliability(
    rows: Sequence[Dict[str, str]],
    summary: Dict[str, object],
    label_column: str,
) -> Tuple[float, str, List[str], Dict[str, object]]:
    score = 1.0
    warnings: List[str] = []
    details: Dict[str, object] = {}

    if bool(summary.get("order_invariant")) and label_column in {"pred", "pred_smooth", "pred_smooth_raw"}:
        score -= 0.30
        warnings.append(
            "summary.json reports order_invariant=true, so labels ending with 1/2 may blur player ownership."
        )

    columns = set(rows[0].keys()) if rows else set()
    has_pred_fixed = "pred_fixed" in columns
    if bool(summary.get("enforce_bottom_2")) and not has_pred_fixed:
        score -= 0.10
        warnings.append(
            "summary.json sets enforce_bottom_2=true, but pred_fixed is missing from the CSV. "
            "Re-run the ROLEFIX prediction script for role-stable labels."
        )
    elif label_column != "pred_fixed" and has_pred_fixed:
        warnings.append(
            "A pred_fixed column is available but not in use. "
            "Pass --label-column pred_fixed for role-stable labels."
        )

    track_stats: Dict[str, Dict[str, object]] = {}
    for col in ("track_id_a", "track_id_b"):
        if not rows or col not in rows[0]:
            continue

        values = [
            row.get(col, "").strip()
            for row in rows
            if row.get(col, "").strip() not in {"", "-1"}
        ]
        if not values:
            continue

        unique_ids = sorted(set(values), key=natural_sort_key)
        changes = sum(1 for left, right in zip(values, values[1:]) if left != right)
        most_common_id, most_common_count = Counter(values).most_common(1)[0]
        dominant_fraction = most_common_count / max(1, len(values))

        track_stats[col] = {
            "unique_ids": unique_ids,
            "unique_count": len(unique_ids),
            "changes": changes,
            "dominant_id": most_common_id,
            "dominant_fraction": round(dominant_fraction, 4),
        }

        if dominant_fraction < 0.35:
            score -= 0.20
            warnings.append(
                f"{col} is unstable ({len(unique_ids)} ids, most common id covers only {pct(dominant_fraction)}% of rows)."
            )
        elif dominant_fraction < 0.55:
            score -= 0.10
            warnings.append(
                f"{col} changes often ({len(unique_ids)} ids, dominant share {pct(dominant_fraction)}%)."
            )

    details["track_columns"] = track_stats
    score = max(0.10, min(1.0, score))
    if score >= 0.75:
        band = "high"
    elif score >= 0.50:
        band = "medium"
    else:
        band = "low"

    if band == "low":
        warnings.append(
            "Treat Player 1 / Player 2 as label-role profiles unless you have stable tracking IDs or role-fixed predictions."
        )

    return score, band, warnings, details


def collect_player_counters(
    rows: Sequence[Dict[str, str]], label_column: str
) -> Tuple[Dict[int, Counter], Dict[int, List[Tuple[str, str]]], Counter]:
    """Role-aware attribution.

    Each frame credits BOTH athletes their role: the bottom athlete (the label
    suffix) is playing the position from the bottom, the other athlete is on top.
    Returns:
      - player_role_counts[pid]: Counter over (family, role) tuples
      - player_role_timeline[pid]: per-frame [(family, role), ...] in order
      - match_families: Counter over family across the whole match
    """
    player_role_counts: Dict[int, Counter] = {1: Counter(), 2: Counter()}
    player_role_timeline: Dict[int, List[Tuple[str, str]]] = {1: [], 2: []}
    match_families: Counter = Counter()

    for row in rows:
        family, bottom_id = parse_label(row[label_column])
        match_families[family] += 1.0
        for pid in (1, 2):
            role = role_for(family, bottom_id, pid)
            player_role_counts[pid][(family, role)] += 1.0
            player_role_timeline[pid].append((family, role))

    return player_role_counts, player_role_timeline, match_families


def role_frames(role_counter: Counter, family: str, role: str) -> float:
    return float(role_counter.get((family, role), 0.0))


def dominant_guard_by_role(role_counter: Counter, role: str) -> Optional[str]:
    """Which guard the player spends most time in for a given role."""
    candidates = [(g, role_frames(role_counter, g, role)) for g in GUARD_FAMILIES]
    candidates = [c for c in candidates if c[1] > 0]
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[1], item[0]))
    return candidates[0][0]


def build_player_metrics(role_counter: Counter) -> Dict[str, float]:
    total = float(sum(role_counter.values()))
    if total <= 0:
        total = 1.0

    def s(families: Iterable[str], role: str) -> float:
        return float(sum(role_counter.get((f, role), 0.0) for f in families))

    guard_bottom = s(GUARD_FAMILIES, "bottom")          # playing guard
    guard_top = s(GUARD_FAMILIES, "top")                # passing a guard
    dom_top = s(DOMINANT_TOP_FAMILIES, "top")           # side/mount/back on top
    dom_bottom = s(DOMINANT_TOP_FAMILIES, "bottom")     # being controlled
    takedown_top = role_frames(role_counter, "takedown", "top")
    takedown_bottom = role_frames(role_counter, "takedown", "bottom")
    turtle_top = role_frames(role_counter, "turtle", "top")
    turtle_bottom = role_frames(role_counter, "turtle", "bottom")
    standing = role_frames(role_counter, "standing", "neutral")
    leg = role_frames(role_counter, "5050_guard", "neutral")
    mount_top = role_frames(role_counter, "mount", "top")
    back_top = role_frames(role_counter, "back", "top")
    side_top = role_frames(role_counter, "side_control", "top")

    top_total = guard_top + dom_top + takedown_top + turtle_top
    bottom_total = guard_bottom + dom_bottom + takedown_bottom + turtle_bottom
    neutral_total = standing + leg

    return {
        "total_attributed_frames": total,
        # role-aware frame counts
        "guard_play_frames": guard_bottom,
        "guard_pass_frames": guard_top,
        "dominant_top_frames": dom_top,
        "bottom_defense_frames": dom_bottom,
        "takedown_top_frames": takedown_top,
        "takedown_bottom_frames": takedown_bottom,
        "turtle_top_frames": turtle_top,
        "turtle_bottom_frames": turtle_bottom,
        "standing_frames": standing,
        "leg_entanglement_frames": leg,
        "mount_top_frames": mount_top,
        "back_top_frames": back_top,
        "side_control_top_frames": side_top,
        "top_total_frames": top_total,
        "bottom_total_frames": bottom_total,
        # ratios
        "guard_play_ratio": guard_bottom / total,
        "guard_pass_ratio": guard_top / total,
        "dominant_top_ratio": dom_top / total,
        "bottom_defense_ratio": dom_bottom / total,
        "turtle_ratio": (turtle_top + turtle_bottom) / total,
        "leg_ratio": leg / total,
        "standing_ratio": standing / total,
        "neutral_ratio": neutral_total / total,
        "top_ratio": top_total / total,       # overall time on top (passing + control)
        "bottom_ratio": bottom_total / total,  # overall time on bottom
        "finish_ratio": (mount_top + back_top) / total,
        "wrestling_ratio": (takedown_top + takedown_bottom + 0.4 * standing) / total,
        # alias: "guard_ratio" now means time PLAYING guard (bottom)
        "guard_ratio": guard_bottom / total,
    }


def build_role_segments(timeline: Sequence[Tuple[str, str]]) -> List[Dict[str, object]]:
    """Collapse a per-frame [(family, role)] timeline into contiguous segments."""
    segments: List[Dict[str, object]] = []
    if not timeline:
        return segments
    cur_fam, cur_role = timeline[0]
    start = 0
    for i in range(1, len(timeline)):
        fam, role = timeline[i]
        if fam == cur_fam and role == cur_role:
            continue
        segments.append({"family": cur_fam, "role": cur_role,
                         "start": start, "end": i - 1, "frames": i - start})
        cur_fam, cur_role, start = fam, role, i
    segments.append({"family": cur_fam, "role": cur_role,
                     "start": start, "end": len(timeline) - 1,
                     "frames": len(timeline) - start})
    return segments


def analyze_passing(
    role_segments: Sequence[Dict[str, object]], total_frames: float, fps: float
) -> List[Dict[str, object]]:
    """Per-guard breakdown of the athlete's TOP-in-guard (passing) time.

    For each guard-passing segment, the next segment decides the outcome:
      - converted: next is a dominant top (side control / mount / back on top)
      - swept:     next puts this athlete on the bottom (reversed)
      - reset:     next is another guard / neutral (stalled or restarted)
    """
    per_guard: Dict[str, Dict[str, object]] = {}
    for i, seg in enumerate(role_segments):
        fam, role = str(seg["family"]), str(seg["role"])
        if role != "top" or fam not in GUARD_FAMILIES:
            continue
        g = per_guard.setdefault(fam, {"attempts": 0, "frames": 0.0, "converted": 0,
                                       "swept": 0, "reset": 0, "convert_frames": []})
        g["attempts"] = int(g["attempts"]) + 1
        g["frames"] = float(g["frames"]) + int(seg["frames"])
        nxt = role_segments[i + 1] if i + 1 < len(role_segments) else None
        if nxt is not None and nxt["role"] == "top" and nxt["family"] in DOMINANT_TOP_FAMILIES:
            g["converted"] = int(g["converted"]) + 1
            g["convert_frames"].append(int(seg["frames"]))  # type: ignore[attr-defined]
        elif nxt is not None and nxt["role"] == "bottom":
            g["swept"] = int(g["swept"]) + 1
        else:
            g["reset"] = int(g["reset"]) + 1

    out: List[Dict[str, object]] = []
    for g, d in per_guard.items():
        attempts = max(1, int(d["attempts"]))
        cf = d["convert_frames"]  # type: ignore[index]
        avg_to_pass = (sum(cf) / len(cf)) if cf else 0.0
        out.append({
            "guard": g,
            "label": GUARD_DISPLAY.get(g, g),
            "attempts": int(d["attempts"]),
            "frames": float(d["frames"]),
            "seconds": frames_to_seconds(float(d["frames"]), fps),
            "share_percent": pct(float(d["frames"]) / max(1.0, total_frames)),
            "converted": int(d["converted"]),
            "swept": int(d["swept"]),
            "reset": int(d["reset"]),
            "conversion_percent": pct(int(d["converted"]) / attempts),
            "avg_seconds_to_pass": frames_to_seconds(avg_to_pass, fps),
        })
    out.sort(key=lambda x: (-x["frames"], x["guard"]))
    return out


def analyze_guard_play(
    role_segments: Sequence[Dict[str, object]], total_frames: float, fps: float
) -> List[Dict[str, object]]:
    """Per-guard breakdown of the athlete's BOTTOM-in-guard (guard-playing) time.

    Outcome of each guard-bottom segment:
      - swept_up: next puts this athlete on top (sweep / wrestle-up)
      - passed:   next puts this athlete bottom in a dominant top (got passed)
      - retained: next is another guard / neutral (kept or reset the guard)
    """
    per_guard: Dict[str, Dict[str, object]] = {}
    for i, seg in enumerate(role_segments):
        fam, role = str(seg["family"]), str(seg["role"])
        if role != "bottom" or fam not in GUARD_FAMILIES:
            continue
        g = per_guard.setdefault(fam, {"instances": 0, "frames": 0.0,
                                       "swept_up": 0, "passed": 0, "retained": 0})
        g["instances"] = int(g["instances"]) + 1
        g["frames"] = float(g["frames"]) + int(seg["frames"])
        nxt = role_segments[i + 1] if i + 1 < len(role_segments) else None
        if nxt is not None and nxt["role"] == "top":
            g["swept_up"] = int(g["swept_up"]) + 1
        elif nxt is not None and nxt["role"] == "bottom" and nxt["family"] in DOMINANT_TOP_FAMILIES:
            g["passed"] = int(g["passed"]) + 1
        else:
            g["retained"] = int(g["retained"]) + 1

    out: List[Dict[str, object]] = []
    for g, d in per_guard.items():
        inst = max(1, int(d["instances"]))
        out.append({
            "guard": g,
            "label": GUARD_DISPLAY.get(g, g),
            "instances": int(d["instances"]),
            "frames": float(d["frames"]),
            "seconds": frames_to_seconds(float(d["frames"]), fps),
            "share_percent": pct(float(d["frames"]) / max(1.0, total_frames)),
            "swept_up": int(d["swept_up"]),
            "passed": int(d["passed"]),
            "retained": int(d["retained"]),
            "sweep_percent": pct(int(d["swept_up"]) / inst),
            "retention_percent": pct(int(d["retained"]) / inst),
        })
    out.sort(key=lambda x: (-x["frames"], x["guard"]))
    return out


def primary_profile_name(role_counter: Counter, metrics: Dict[str, float]) -> str:
    passing = metrics["guard_pass_ratio"]
    dom_top = metrics["dominant_top_ratio"]
    top_total = metrics["top_ratio"]
    bottom_total = metrics["bottom_ratio"]
    guard_play = metrics["guard_play_ratio"]

    # Top / passing athlete
    if top_total >= bottom_total and top_total >= 0.30:
        if dom_top >= passing and dom_top >= 0.18:
            if metrics["back_top_frames"] >= metrics["side_control_top_frames"] + 0.5 * metrics["mount_top_frames"]:
                return "Top-control player with back-taking bias"
            if metrics["mount_top_frames"] + metrics["side_control_top_frames"] >= 0.55 * max(1.0, metrics["dominant_top_frames"]):
                return "Pressure top player"
            return "Top-control player"
        pref = dominant_guard_by_role(role_counter, "top")
        if pref:
            return f"Guard passer ({GUARD_DISPLAY[pref]} passer)"
        return "Guard passer"

    # Bottom / guard athlete
    if guard_play >= 0.28 and guard_play >= metrics["turtle_ratio"]:
        pref = dominant_guard_by_role(role_counter, "bottom")
        if pref:
            return f"Guard player ({GUARD_DISPLAY[pref]} bottom)"
        return "Guard player"

    if metrics["turtle_ratio"] >= 0.20:
        return "Scramble / turtle-heavy player"
    if metrics["leg_ratio"] >= 0.15:
        return "50/50 and leg-entanglement player"

    return "Balanced positional player"


def secondary_traits(
    role_counter: Counter,
    metrics: Dict[str, float],
    primary_profile: str,
    sequence_profile: str,
) -> List[str]:
    traits: List[Tuple[float, str]] = []
    prof = primary_profile.lower()

    if metrics["guard_pass_ratio"] >= 0.15 and "passer" not in prof:
        pref = dominant_guard_by_role(role_counter, "top")
        traits.append((metrics["guard_pass_ratio"],
                       f"passes {GUARD_DISPLAY[pref]}" if pref else "guard passing"))

    if metrics["guard_play_ratio"] >= 0.15 and "guard player" not in prof:
        pref = dominant_guard_by_role(role_counter, "bottom")
        traits.append((metrics["guard_play_ratio"],
                       f"plays {GUARD_DISPLAY[pref]}" if pref else "guard bottom game"))

    if metrics["dominant_top_ratio"] >= 0.15 and "Top-control" not in primary_profile and "Pressure" not in primary_profile:
        if metrics["back_top_frames"] >= metrics["side_control_top_frames"]:
            traits.append((metrics["dominant_top_ratio"], "back-taking threat"))
        else:
            traits.append((metrics["dominant_top_ratio"], "top-control threat"))

    if metrics["bottom_defense_ratio"] >= 0.12:
        traits.append((metrics["bottom_defense_ratio"], "defends from bottom under control"))

    if metrics["turtle_ratio"] >= 0.15 and "Scramble" not in primary_profile:
        traits.append((metrics["turtle_ratio"], "scramble / turtle exchanges"))

    if metrics["leg_ratio"] >= 0.12 and "50/50" not in primary_profile:
        traits.append((metrics["leg_ratio"], "50/50 involvement"))

    if metrics["wrestling_ratio"] >= 0.06:
        traits.append((metrics["wrestling_ratio"], "takedown / standing exchanges"))

    if sequence_profile == "Top-control chain builder":
        traits.append((0.35, "chains attacks after the first control"))
    elif sequence_profile == "Guard/turtle recycler":
        traits.append((0.34, "recycles exchanges through recovery loops"))
    elif sequence_profile == "Repeated entry player":
        traits.append((0.33, "re-enters the fight through repeated entries"))
    elif sequence_profile == "Scramble opportunist":
        traits.append((0.32, "converts scrambles into offense"))
    elif sequence_profile == "Long-phase consolidator":
        traits.append((0.31, "holds positions once established"))

    traits.sort(key=lambda item: (-item[0], item[1]))
    return [text for _, text in traits[:4]]


def dedupe_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def sequence_profile_name(motifs: Dict[str, int], avg_duration: float, longest_duration: int, metrics: Dict[str, float]) -> str:
    chain_score = motifs["pass_to_finish"] + motifs["finish_cycle"] + motifs["turtle_to_attack"]
    recovery_score = motifs["guard_turtle_loops"] + motifs["guard_reentries"]
    entry_score = motifs["standing_to_takedown"] + motifs["takedown_to_control"] + motifs["takedown_to_reset"]

    if recovery_score >= 4 and recovery_score >= max(chain_score, entry_score) and metrics["guard_ratio"] >= 0.20:
        return "Guard/turtle recycler"
    if chain_score >= 4 and chain_score >= max(recovery_score, entry_score) and metrics["top_ratio"] >= 0.20:
        return "Top-control chain builder"
    if entry_score >= 3 and entry_score >= max(recovery_score, chain_score) and metrics["wrestling_ratio"] >= 0.05:
        return "Repeated entry player"
    if motifs["turtle_to_attack"] >= 2:
        return "Scramble opportunist"
    if avg_duration >= 8.0 and longest_duration >= 15:
        return "Long-phase consolidator"
    return "Mixed transition player"


def analyze_match_sequence(family_segments: Sequence[Dict[str, object]], fps: float) -> Dict[str, object]:
    families = [str(segment["family"]) for segment in family_segments]
    pair_counts = count_pairs(families)
    triple_counts = count_triples(families)

    motifs = {
        "standing_to_takedown": count_patterns(pair_counts, STANDING_TO_TAKEDOWN),
        "takedown_to_control": count_patterns(pair_counts, TAKEDOWN_TO_CONTROL),
        "takedown_to_reset": count_patterns(pair_counts, TAKEDOWN_TO_RESET),
        "guard_to_turtle": count_patterns(pair_counts, GUARD_TO_TURTLE),
        "turtle_to_guard": count_patterns(pair_counts, TURTLE_TO_GUARD),
        "turtle_to_attack": count_patterns(pair_counts, TURTLE_TO_ATTACK),
        "pass_to_finish": count_patterns(pair_counts, PASS_TO_FINISH),
        "finish_cycle": count_patterns(pair_counts, FINISH_CYCLE),
    }

    avg_duration = (
        float(sum(int(segment["duration_rows"]) for segment in family_segments)) / max(1, len(family_segments))
    )
    longest_phases = sorted(
        [
            {
                "family": str(segment["family"]),
                "label": family_name(str(segment["family"])),
                "duration_rows": int(segment["duration_rows"]),
                "duration_seconds": frames_to_seconds(int(segment["duration_rows"]), fps),
                "start_frame": int(segment["start_frame"]),
                "end_frame": int(segment["end_frame"]),
            }
            for segment in family_segments
        ],
        key=lambda item: (-item["duration_rows"], item["start_frame"]),
    )[:5]

    narratives: List[str] = []
    if motifs["takedown_to_reset"] >= max(2, motifs["takedown_to_control"] + 1):
        narratives.append(
            "Takedown phases often recycle into guard or turtle instead of settling immediately into top control."
        )
    if motifs["guard_to_turtle"] + motifs["turtle_to_attack"] >= 4:
        narratives.append(
            "Guard exchanges regularly spill into turtle scrambles before someone consolidates control."
        )
    if motifs["pass_to_finish"] + motifs["finish_cycle"] >= 4:
        narratives.append(
            "Once the sequence reaches passing positions, it often keeps climbing toward mount or back."
        )
    if len(family_segments) >= 70 and avg_duration <= 8.0:
        narratives.append("The match is transition-heavy, with short phases and frequent positional changes.")
    elif avg_duration >= 14.0:
        narratives.append("The match features long control phases once a dominant position is established.")
    if not narratives:
        narratives.append("The sequence is mixed, without a single transition pattern dominating the match.")

    return {
        "segment_count": len(family_segments),
        "average_phase_frames": round(avg_duration, 2),
        "average_phase_seconds": frames_to_seconds(avg_duration, fps),
        "opening_sequence": [family_name(item) for item in families[:8]],
        "ending_sequence": [family_name(item) for item in families[-8:]],
        "top_transitions": serialize_pair_counts(pair_counts, limit=8),
        "top_chains": serialize_triple_counts(triple_counts, limit=6),
        "motifs": motifs,
        "longest_phases": longest_phases,
        "narratives": narratives,
    }


def analyze_player_sequence(
    player_id: int,
    role_segments: Sequence[Dict[str, object]],
    label_segments: Sequence[Dict[str, object]],
    metrics: Dict[str, float],
    fps: float,
) -> Dict[str, object]:
    own_segments = list(role_segments)
    families = [str(segment["family"]) for segment in own_segments]
    pair_counts = count_pairs(families)
    triple_counts = count_triples(families)

    motifs = {
        "standing_to_takedown": count_patterns(pair_counts, STANDING_TO_TAKEDOWN),
        "takedown_to_control": count_patterns(pair_counts, TAKEDOWN_TO_CONTROL),
        "takedown_to_reset": count_patterns(pair_counts, TAKEDOWN_TO_RESET),
        "guard_turtle_loops": count_patterns(pair_counts, GUARD_TO_TURTLE | TURTLE_TO_GUARD),
        "guard_reentries": count_patterns(pair_counts, GUARD_REENTRY),
        "turtle_to_attack": count_patterns(pair_counts, TURTLE_TO_ATTACK),
        "pass_to_finish": count_patterns(pair_counts, PASS_TO_FINISH),
        "finish_cycle": count_patterns(pair_counts, FINISH_CYCLE),
    }

    avg_duration = float(sum(int(segment["frames"]) for segment in own_segments)) / max(1, len(own_segments))
    longest_duration = max((int(segment["frames"]) for segment in own_segments), default=0)
    profile = sequence_profile_name(motifs, avg_duration, longest_duration, metrics)

    narratives: List[str] = []
    if motifs["guard_turtle_loops"] + motifs["guard_reentries"] >= 3:
        narratives.append("Frequently revisits guard or turtle instead of letting the exchange end cleanly.")
    if motifs["pass_to_finish"] + motifs["finish_cycle"] >= 3:
        narratives.append("Often keeps advancing after the first top-control beat instead of stopping at the pass.")
    if motifs["takedown_to_control"] + motifs["takedown_to_reset"] + motifs["standing_to_takedown"] >= 2:
        narratives.append("Re-enters the match through repeated takedown or entry phases.")
    if motifs["turtle_to_attack"] >= 2:
        narratives.append("Turns scramble or turtle sequences into attacks more than once.")
    if avg_duration >= 8.0 and longest_duration >= 15:
        narratives.append("Has at least a few long ownership phases once the position is established.")
    if not narratives and pair_counts:
        top_pair = pair_counts.most_common(1)[0][0]
        narratives.append(f"Most common recurring transition: {format_sequence(top_pair)}.")

    longest_segments = sorted(
        [
            {
                "family": str(segment["family"]),
                "label": role_label(str(segment["family"]), str(segment["role"])),
                "role": str(segment["role"]),
                "duration_rows": int(segment["frames"]),
                "duration_seconds": frames_to_seconds(int(segment["frames"]), fps),
                "start_frame": int(segment["start"]),
                "end_frame": int(segment["end"]),
            }
            for segment in own_segments
        ],
        key=lambda item: (-item["duration_rows"], item["start_frame"]),
    )[:5]

    progression = analyze_advance_concession(label_segments, player_id)

    return {
        "segment_count": len(own_segments),
        "average_owned_phase_frames": round(avg_duration, 2),
        "average_owned_phase_seconds": frames_to_seconds(avg_duration, fps),
        "longest_owned_phase_frames": int(longest_duration),
        "longest_owned_phase_seconds": frames_to_seconds(longest_duration, fps),
        "sequence_profile": profile,
        "narratives": narratives[:4],
        "motifs": motifs,
        "top_transitions": serialize_pair_counts(pair_counts, limit=5),
        "top_chains": serialize_triple_counts(triple_counts, limit=4),
        "longest_owned_phases": longest_segments,
        "progression": progression,
    }


PASS_ADVICE = {
    "open_guard": "win grips and pin a shin or knee-line before stepping in, and pass by changing angle instead of driving straight.",
    "closed_guard": "open the guard with tall posture and inside elbows before committing pressure, keeping your arms out of attacking range.",
    "half_guard": "control the inside knee and head, clear the knee-line, then flatten before settling chest pressure.",
}
DEFEND_PASS_ADVICE = {
    "open_guard": "keep distance and active feet, recovering frames before they win inside grips.",
    "closed_guard": "stay tight, hip-escape early, and do not let them posture up and open you flat.",
    "half_guard": "fight for the underhook and knee-shield early so they cannot clear your knee-line.",
}


def build_counter_plan(
    role_counter: Counter,
    metrics: Dict[str, float],
    sequence_info: Dict[str, object],
    passing: Sequence[Dict[str, object]],
    guard_play: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    """Data-driven, guard-specific game plan for the OPPONENT of this athlete.

    Every line is keyed to this athlete's real per-guard numbers so the two
    players get different plans.
    """
    actions: List[str] = []
    is_top = metrics["top_ratio"] >= metrics["bottom_ratio"]

    if is_top and passing:
        most_used = passing[0]
        strongest = max(passing, key=lambda p: (p["conversion_percent"], p["frames"]))
        weakest = min(passing, key=lambda p: (p["conversion_percent"], -p["frames"]))
        style = f"Deny their {strongest['label']} pass and steer them into a guard they finish less often"
        speed = (
            f" in ~{format_seconds(float(most_used['avg_seconds_to_pass']))}"
            if most_used["avg_seconds_to_pass"] else ""
        )
        actions.append(
            f"They pass {most_used['label']} most ({most_used['share_percent']}% of top time, "
            f"{most_used['conversion_percent']}% reach control{speed}) — "
            f"{DEFEND_PASS_ADVICE.get(str(most_used['guard']), 'recover guard early.')}"
        )
        if weakest["guard"] != strongest["guard"] and weakest["conversion_percent"] + 5 < strongest["conversion_percent"]:
            actions.append(
                f"They convert {weakest['label']} passes only {weakest['conversion_percent']}% of the time — "
                f"steer the exchange into {weakest['label']} and reset there."
            )
        if metrics["dominant_top_ratio"] >= 0.12:
            actions.append(
                f"They reach dominant top control {pct(metrics['dominant_top_ratio'])}% of the match — "
                "recover guard before they settle chest-to-chest and climb to mount or back."
            )
    elif guard_play:
        main = guard_play[0]
        style = f"Structured passing that clears their {main['label']} and denies the recovery"
        retain = (
            f", holding or recovering it {main['retention_percent']}% of the time"
            if main["retention_percent"] else ""
        )
        actions.append(
            f"They play {main['label']} {main['share_percent']}% of bottom time{retain} — "
            f"{PASS_ADVICE.get(str(main['guard']), 'clear grips and pass with staple control.')}"
        )
        if main["sweep_percent"] >= 20:
            actions.append(
                f"They sweep or wrestle-up from {main['label']} {main['sweep_percent']}% of the time — "
                "kill the underhook and keep your weight back before passing."
            )
        if len(guard_play) > 1 and guard_play[1]["share_percent"] >= 10:
            second = guard_play[1]
            actions.append(
                f"They also use {second['label']} ({second['share_percent']}%) — "
                f"{PASS_ADVICE.get(str(second['guard']), 'clear grips first.')}"
            )
    else:
        style = "Force them out of their comfort phase and play first"
        actions.append(
            "Set the pace early and keep transitions tight so they never settle into a preferred position."
        )

    if metrics["leg_ratio"] >= 0.10:
        actions.append(
            f"They spend {pct(metrics['leg_ratio'])}% in 50/50 — clear your knee-line early and "
            "avoid stalling in equal leg positions."
        )
    if metrics["turtle_ratio"] >= 0.12:
        actions.append(
            "Follow hips and shoulders together in scrambles; they turn turtle exchanges into offense."
        )
    if metrics["back_top_frames"] / max(1.0, metrics["total_attributed_frames"]) >= 0.06:
        actions.append(
            "Guard your back in every scramble — a meaningful share of their control time is back control."
        )

    return {
        "recommended_style": style,
        "key_actions": dedupe_keep_order(actions)[:5],
    }


def build_player_report(
    player_id: int,
    player_name: str,
    role_counter: Counter,
    metrics: Dict[str, float],
    sequence_info: Dict[str, object],
    passing: Sequence[Dict[str, object]],
    guard_play: Sequence[Dict[str, object]],
    fps: float,
) -> Dict[str, object]:
    profile = primary_profile_name(role_counter, metrics)
    counter_plan = build_counter_plan(role_counter, metrics, sequence_info, passing, guard_play)
    traits = secondary_traits(role_counter, metrics, profile, sequence_info["sequence_profile"])

    total = max(1.0, metrics["total_attributed_frames"])
    top_families_list = [
        {
            "family": fam,
            "role": role,
            "label": role_label(fam, role),
            "frames": float(value),
            "seconds": frames_to_seconds(float(value), fps),
            "share_percent": pct(float(value) / total),
        }
        for (fam, role), value in sorted(role_counter.items(), key=lambda kv: -kv[1])[:6]
        if float(value) > 0
    ]

    return {
        "player_id": player_id,
        "player_name": player_name,
        "primary_profile": profile,
        "secondary_traits": traits,
        "metrics": {
            "attributed_frames": float(metrics["total_attributed_frames"]),
            "attributed_seconds": frames_to_seconds(metrics["total_attributed_frames"], fps),
            "top_ratio_percent": pct(metrics["top_ratio"]),
            "bottom_ratio_percent": pct(metrics["bottom_ratio"]),
            "guard_pass_ratio_percent": pct(metrics["guard_pass_ratio"]),
            "guard_play_ratio_percent": pct(metrics["guard_play_ratio"]),
            "dominant_top_ratio_percent": pct(metrics["dominant_top_ratio"]),
            "bottom_defense_ratio_percent": pct(metrics["bottom_defense_ratio"]),
            "turtle_ratio_percent": pct(metrics["turtle_ratio"]),
            "leg_ratio_percent": pct(metrics["leg_ratio"]),
            "neutral_ratio_percent": pct(metrics["neutral_ratio"]),
            "wrestling_ratio_percent": pct(metrics["wrestling_ratio"]),
            "finish_ratio_percent": pct(metrics["finish_ratio"]),
            # legacy alias (guard = time PLAYING guard from the bottom)
            "guard_ratio_percent": pct(metrics["guard_play_ratio"]),
        },
        "top_families": top_families_list,
        "passing": list(passing),
        "guard_play": list(guard_play),
        "sequence": sequence_info,
        "counter_plan": counter_plan,
    }


def build_metrics_rows(player_reports: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    rows = []
    for report in player_reports:
        metrics = report["metrics"]
        sequence_info = report["sequence"]
        motifs = sequence_info["motifs"]
        progression = sequence_info.get("progression", {})
        row = {
            "player_id": report["player_id"],
            "player_name": report["player_name"],
            "primary_profile": report["primary_profile"],
            "secondary_traits": " | ".join(report["secondary_traits"]),
            "sequence_profile": sequence_info["sequence_profile"],
            "attributed_frames": metrics["attributed_frames"],
            "attributed_seconds": metrics.get("attributed_seconds", 0.0),
            "top_ratio_percent": metrics["top_ratio_percent"],
            "bottom_ratio_percent": metrics.get("bottom_ratio_percent", 0.0),
            "guard_pass_ratio_percent": metrics.get("guard_pass_ratio_percent", 0.0),
            "guard_play_ratio_percent": metrics.get("guard_play_ratio_percent", 0.0),
            "dominant_top_ratio_percent": metrics.get("dominant_top_ratio_percent", 0.0),
            "bottom_defense_ratio_percent": metrics.get("bottom_defense_ratio_percent", 0.0),
            "turtle_ratio_percent": metrics["turtle_ratio_percent"],
            "leg_ratio_percent": metrics["leg_ratio_percent"],
            "wrestling_ratio_percent": metrics["wrestling_ratio_percent"],
            "finish_ratio_percent": metrics["finish_ratio_percent"],
            "main_pass_guard": (report["passing"][0]["label"] if report.get("passing") else ""),
            "main_pass_conversion_percent": (report["passing"][0]["conversion_percent"] if report.get("passing") else 0.0),
            "main_guard_played": (report["guard_play"][0]["label"] if report.get("guard_play") else ""),
            "main_guard_retention_percent": (report["guard_play"][0]["retention_percent"] if report.get("guard_play") else 0.0),
            "avg_owned_phase_frames": sequence_info["average_owned_phase_frames"],
            "avg_owned_phase_seconds": sequence_info.get("average_owned_phase_seconds", 0.0),
            "longest_owned_phase_frames": sequence_info["longest_owned_phase_frames"],
            "longest_owned_phase_seconds": sequence_info.get("longest_owned_phase_seconds", 0.0),
            "advances": progression.get("advances", 0),
            "concessions": progression.get("concessions", 0),
            "net_progression": progression.get("net_progression", 0.0),
            "guard_turtle_loops": motifs["guard_turtle_loops"],
            "guard_reentries": motifs["guard_reentries"],
            "takedown_to_control": motifs["takedown_to_control"],
            "takedown_to_reset": motifs["takedown_to_reset"],
            "turtle_to_attack": motifs["turtle_to_attack"],
            "pass_to_finish": motifs["pass_to_finish"],
            "finish_cycle": motifs["finish_cycle"],
            "recommended_counter_style": report["counter_plan"]["recommended_style"],
        }
        rows.append(row)
    return rows


def write_metrics_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def render_pattern_lines(items: Sequence[Dict[str, object]]) -> str:
    if not items:
        return "none"
    return ", ".join(f"{item['label']} ({item['count']})" for item in items)


def build_text_report(
    summary: Dict[str, object],
    base_dir: Path,
    csv_path: Path,
    label_column: str,
    rows: Sequence[Dict[str, str]],
    reliability_score: float,
    reliability_band: str,
    reliability_warnings: Sequence[str],
    match_sequence: Dict[str, object],
    player_reports: Sequence[Dict[str, object]],
    fps: float,
    confidence_info: Optional[Dict[str, object]] = None,
) -> str:
    lines: List[str] = []
    lines.append("Player Style Analysis")
    lines.append("")
    lines.append("Input")
    lines.append(f"- Position directory: {base_dir}")
    lines.append(f"- CSV: {csv_path}")
    lines.append(f"- Label column used: {label_column}")
    lines.append(f"- FPS assumed: {fps}")
    if confidence_info and confidence_info.get("min_conf", 0.0) > 0:
        lines.append(
            f"- Confidence filter: column={confidence_info.get('column')}, "
            f"min_conf={confidence_info.get('min_conf')}, "
            f"dropped={confidence_info.get('dropped')} rows"
        )
    if "frames_total" in summary and "frames_used" in summary:
        lines.append(
            f"- Frames used: {summary.get('frames_used')} / {summary.get('frames_total')} "
            f"(skipped: {summary.get('frames_skipped', 'n/a')})"
        )
    lines.append(f"- Attribution reliability: {reliability_band} ({reliability_score:.2f})")

    if reliability_warnings:
        lines.append("- Warnings:")
        for warning in reliability_warnings:
            lines.append(f"  * {warning}")

    lines.append("")
    lines.append("Match Sequence View")
    lines.append(
        f"- Family phases: {match_sequence['segment_count']} segments, "
        f"average {match_sequence['average_phase_frames']} frames "
        f"({format_seconds(float(match_sequence.get('average_phase_seconds', 0.0)))}) per phase"
    )
    lines.append(f"- Opening flow: {', '.join(match_sequence['opening_sequence'])}")
    lines.append(f"- Ending flow: {', '.join(match_sequence['ending_sequence'])}")
    lines.append(f"- Common transitions: {render_pattern_lines(match_sequence['top_transitions'][:5])}")
    lines.append(f"- Common chains: {render_pattern_lines(match_sequence['top_chains'][:4])}")
    lines.append(
        "- Strong motifs: "
        f"guard->turtle {match_sequence['motifs']['guard_to_turtle']}, "
        f"turtle->attack {match_sequence['motifs']['turtle_to_attack']}, "
        f"pass->finish {match_sequence['motifs']['pass_to_finish']}, "
        f"finish cycles {match_sequence['motifs']['finish_cycle']}, "
        f"takedown->reset {match_sequence['motifs']['takedown_to_reset']}, "
        f"takedown->control {match_sequence['motifs']['takedown_to_control']}"
    )
    for narrative in match_sequence["narratives"]:
        lines.append(f"- Read: {narrative}")

    for report in player_reports:
        metrics = report["metrics"]
        sequence_info = report["sequence"]
        progression = sequence_info.get("progression", {})
        lines.append("")
        lines.append(f"{report['player_name']}")
        lines.append(f"- Primary profile: {report['primary_profile']}")
        lines.append(f"- Sequence profile: {sequence_info['sequence_profile']}")
        if report["secondary_traits"]:
            lines.append(f"- Secondary traits: {', '.join(report['secondary_traits'])}")
        lines.append(
            "- Evidence: "
            f"guard passing {metrics['guard_pass_ratio_percent']}%, "
            f"dominant top {metrics['dominant_top_ratio_percent']}%, "
            f"guard bottom {metrics['guard_play_ratio_percent']}%, "
            f"bottom defense {metrics['bottom_defense_ratio_percent']}%, "
            f"50/50 {metrics['leg_ratio_percent']}% "
            f"(overall top {metrics['top_ratio_percent']}% / bottom {metrics['bottom_ratio_percent']}%)"
        )
        if report.get("passing"):
            lines.append(
                "- Passes: "
                + "; ".join(
                    f"{p['label']} {p['share_percent']}% ({p['conversion_percent']}% to control)"
                    for p in report["passing"]
                )
            )
        if report.get("guard_play"):
            lines.append(
                "- Guards played: "
                + "; ".join(
                    f"{g['label']} {g['share_percent']}% ({g['retention_percent']}% retained)"
                    for g in report["guard_play"]
                )
            )
        lines.append(
            f"- Attributed time: {fmt_count(metrics['attributed_frames'])} frames "
            f"({format_seconds(float(metrics.get('attributed_seconds', 0.0)))})"
        )
        lines.append(
            "- Control tempo: "
            f"average owned phase {sequence_info['average_owned_phase_frames']} frames "
            f"({format_seconds(float(sequence_info.get('average_owned_phase_seconds', 0.0)))}), "
            f"longest owned phase {sequence_info['longest_owned_phase_frames']} frames "
            f"({format_seconds(float(sequence_info.get('longest_owned_phase_seconds', 0.0)))})"
        )
        if progression:
            lines.append(
                "- Progression: "
                f"advances {progression.get('advances', 0)}, "
                f"concessions {progression.get('concessions', 0)}, "
                f"net {progression.get('net_progression', 0.0):+.1f} "
                f"(biggest advance {progression.get('biggest_advance', 0.0):+.1f}, "
                f"biggest concession {progression.get('biggest_concession', 0.0):+.1f})"
            )
        top_families_text = ", ".join(
            f"{item['label']} {fmt_count(item['frames'])} "
            f"({format_seconds(float(item.get('seconds', 0.0)))})"
            for item in report["top_families"]
        )
        lines.append(f"- Key positions: {top_families_text}")
        if sequence_info["narratives"]:
            lines.append(f"- Sequence read: {' | '.join(sequence_info['narratives'])}")
        lines.append(f"- Recurring transitions: {render_pattern_lines(sequence_info['top_transitions'])}")
        lines.append(f"- Recurring chains: {render_pattern_lines(sequence_info['top_chains'])}")
        lines.append(f"- Efficient counter style: {report['counter_plan']['recommended_style']}")
        lines.append("- Game plan:")
        for action in report["counter_plan"]["key_actions"]:
            lines.append(f"  * {action}")

    lines.append("")
    lines.append(f"Rows analyzed: {len(rows)}")
    return "\n".join(lines) + "\n"


def build_markdown_report(
    summary: Dict[str, object],
    base_dir: Path,
    csv_path: Path,
    label_column: str,
    rows: Sequence[Dict[str, str]],
    reliability_score: float,
    reliability_band: str,
    reliability_warnings: Sequence[str],
    match_sequence: Dict[str, object],
    player_reports: Sequence[Dict[str, object]],
    fps: float,
    confidence_info: Optional[Dict[str, object]] = None,
) -> str:
    out: List[str] = []
    out.append("# Player Style Analysis")
    out.append("")
    out.append("## Input")
    out.append(f"- **Position directory:** `{base_dir}`")
    out.append(f"- **CSV:** `{csv_path}`")
    out.append(f"- **Label column:** `{label_column}`")
    out.append(f"- **FPS assumed:** {fps}")
    if confidence_info and confidence_info.get("min_conf", 0.0) > 0:
        out.append(
            f"- **Confidence filter:** column `{confidence_info.get('column')}`, "
            f"min={confidence_info.get('min_conf')}, dropped {confidence_info.get('dropped')} rows"
        )
    if "frames_total" in summary and "frames_used" in summary:
        out.append(
            f"- **Frames used:** {summary.get('frames_used')} / {summary.get('frames_total')} "
            f"(skipped {summary.get('frames_skipped', 'n/a')})"
        )
    out.append(f"- **Attribution reliability:** **{reliability_band}** ({reliability_score:.2f})")
    if reliability_warnings:
        out.append("")
        out.append("### Warnings")
        for warning in reliability_warnings:
            out.append(f"- {warning}")

    out.append("")
    out.append("## Match Sequence")
    out.append(
        f"- {match_sequence['segment_count']} family phases, "
        f"avg {match_sequence['average_phase_frames']} frames "
        f"({format_seconds(float(match_sequence.get('average_phase_seconds', 0.0)))})"
    )
    out.append(f"- **Opening:** {' → '.join(match_sequence['opening_sequence'])}")
    out.append(f"- **Ending:** {' → '.join(match_sequence['ending_sequence'])}")

    out.append("")
    out.append("### Top transitions")
    out.append("| Pattern | Count |")
    out.append("|---|---|")
    for item in match_sequence["top_transitions"][:8]:
        out.append(f"| {item['label']} | {item['count']} |")

    out.append("")
    out.append("### Longest phases")
    out.append("| Family | Frames | Seconds | Frame range |")
    out.append("|---|---|---|---|")
    for phase in match_sequence.get("longest_phases", []):
        out.append(
            f"| {phase['label']} | {phase['duration_rows']} | "
            f"{format_seconds(float(phase.get('duration_seconds', 0.0)))} | "
            f"{phase['start_frame']}–{phase['end_frame']} |"
        )

    if match_sequence.get("narratives"):
        out.append("")
        out.append("### Read")
        for narrative in match_sequence["narratives"]:
            out.append(f"- {narrative}")

    for report in player_reports:
        metrics = report["metrics"]
        sequence_info = report["sequence"]
        progression = sequence_info.get("progression", {})

        out.append("")
        out.append(f"## {report['player_name']}")
        out.append(f"- **Primary profile:** {report['primary_profile']}")
        out.append(f"- **Sequence profile:** {sequence_info['sequence_profile']}")
        if report["secondary_traits"]:
            out.append(f"- **Secondary traits:** {', '.join(report['secondary_traits'])}")
        out.append(
            f"- **Attributed time:** {fmt_count(metrics['attributed_frames'])} frames "
            f"({format_seconds(float(metrics.get('attributed_seconds', 0.0)))})"
        )

        out.append("")
        out.append("### Position share (by role)")
        out.append("| Role | Percent |")
        out.append("|---|---|")
        out.append(f"| Guard passing (top) | {metrics['guard_pass_ratio_percent']}% |")
        out.append(f"| Dominant top (side/mount/back) | {metrics['dominant_top_ratio_percent']}% |")
        out.append(f"| Guard playing (bottom) | {metrics['guard_play_ratio_percent']}% |")
        out.append(f"| Bottom defense (under control) | {metrics['bottom_defense_ratio_percent']}% |")
        out.append(f"| Turtle / scramble | {metrics['turtle_ratio_percent']}% |")
        out.append(f"| 50/50 (neutral) | {metrics['leg_ratio_percent']}% |")
        out.append(f"| Overall top / bottom | {metrics['top_ratio_percent']}% / {metrics['bottom_ratio_percent']}% |")

        if report.get("passing"):
            out.append("")
            out.append("### Guard passing breakdown (their top game)")
            out.append("| Guard | Share of top time | Attempts | Reach control | Avg time to pass |")
            out.append("|---|---|---|---|---|")
            for p in report["passing"]:
                secs = format_seconds(float(p.get("avg_seconds_to_pass", 0.0))) if p.get("avg_seconds_to_pass") else "—"
                out.append(
                    f"| {p['label']} | {p['share_percent']}% | {p['attempts']} | "
                    f"{p['conversion_percent']}% | {secs} |"
                )

        if report.get("guard_play"):
            out.append("")
            out.append("### Guard defense breakdown (their bottom game)")
            out.append("| Guard | Share of bottom time | Instances | Retained/recovered | Swept up |")
            out.append("|---|---|---|---|---|")
            for g in report["guard_play"]:
                out.append(
                    f"| {g['label']} | {g['share_percent']}% | {g['instances']} | "
                    f"{g['retention_percent']}% | {g['sweep_percent']}% |"
                )

        if progression:
            out.append("")
            out.append("### Progression")
            out.append("| Metric | Value |")
            out.append("|---|---|")
            out.append(f"| Advances | {progression.get('advances', 0)} |")
            out.append(f"| Concessions | {progression.get('concessions', 0)} |")
            out.append(f"| Net progression | {progression.get('net_progression', 0.0):+.1f} |")
            out.append(f"| Biggest advance | {progression.get('biggest_advance', 0.0):+.1f} |")
            out.append(f"| Biggest concession | {progression.get('biggest_concession', 0.0):+.1f} |")

        out.append("")
        out.append("### Key positions")
        out.append("| Family | Frames | Seconds | Share |")
        out.append("|---|---|---|---|")
        for item in report["top_families"]:
            out.append(
                f"| {item['label']} | {fmt_count(item['frames'])} | "
                f"{format_seconds(float(item.get('seconds', 0.0)))} | {item['share_percent']}% |"
            )

        out.append("")
        out.append(
            f"- **Control tempo:** avg owned phase "
            f"{sequence_info['average_owned_phase_frames']} frames "
            f"({format_seconds(float(sequence_info.get('average_owned_phase_seconds', 0.0)))}), "
            f"longest {sequence_info['longest_owned_phase_frames']} frames "
            f"({format_seconds(float(sequence_info.get('longest_owned_phase_seconds', 0.0)))})"
        )
        if sequence_info.get("narratives"):
            out.append("- **Sequence read:**")
            for narrative in sequence_info["narratives"]:
                out.append(f"  - {narrative}")
        out.append(f"- **Recurring transitions:** {render_pattern_lines(sequence_info['top_transitions'])}")
        out.append(f"- **Recurring chains:** {render_pattern_lines(sequence_info['top_chains'])}")
        out.append(f"- **Efficient counter style:** {report['counter_plan']['recommended_style']}")
        out.append("")
        out.append("**Game plan:**")
        for action in report["counter_plan"]["key_actions"]:
            out.append(f"- {action}")

    out.append("")
    out.append(f"_Rows analyzed: {len(rows)}_")
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--position-dir",
        default=None,
        help="Directory containing summary.json and positions.csv. Defaults to the sample test_images folder if present.",
    )
    parser.add_argument("--summary-json", default=None, help="Path to summary.json")
    parser.add_argument("--positions-csv", default=None, help="Path to positions.csv")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for style_analysis.json / style_report.txt / style_metrics.csv (default: input dir).",
    )
    parser.add_argument(
        "--label-column",
        default=None,
        help="Optional override. Auto-selects pred_fixed > pred_smooth > pred_smooth_raw > pred.",
    )
    parser.add_argument("--player-1-name", default="Player 1")
    parser.add_argument("--player-2-name", default="Player 2")
    parser.add_argument(
        "--fps",
        type=float,
        default=DEFAULT_FPS,
        help=f"Frames per second of the source video (default: {DEFAULT_FPS}). Used to convert rows to seconds.",
    )
    parser.add_argument(
        "--min-conf",
        type=float,
        default=0.0,
        help="Drop rows whose label confidence is below this value (0.0 disables filtering).",
    )
    parser.add_argument(
        "--conf-column",
        default=None,
        help="Override the confidence column. Defaults to pred_smooth_conf or pred_conf based on --label-column.",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Also write a style_report.md alongside the .txt report.",
    )
    args = parser.parse_args()

    if args.fps <= 0:
        parser.error("--fps must be positive")

    base_dir, summary_path, csv_path = resolve_input_paths(
        position_dir=args.position_dir,
        summary_json=args.summary_json,
        positions_csv=args.positions_csv,
    )

    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    rows = load_rows(csv_path)
    label_column = choose_label_column(rows, args.label_column)

    conf_column = args.conf_column or CONF_COLUMN_FOR.get(label_column)
    rows_total = len(rows)
    rows, dropped_low_conf = filter_by_confidence(rows, conf_column, args.min_conf)
    if args.min_conf > 0:
        if not conf_column:
            print(
                f"[warn] --min-conf {args.min_conf} requested but no confidence column is known for "
                f"label '{label_column}'. Skipping filter.",
                file=sys.stderr,
            )
        elif dropped_low_conf > 0:
            print(
                f"[info] Dropped {dropped_low_conf} / {rows_total} rows below {conf_column} >= {args.min_conf}",
                file=sys.stderr,
            )
    if not rows:
        raise RuntimeError(
            "No rows left to analyze after confidence filtering. Lower --min-conf or check the confidence column."
        )

    confidence_info = {
        "column": conf_column,
        "min_conf": args.min_conf,
        "dropped": dropped_low_conf,
        "rows_in": rows_total,
        "rows_out": len(rows),
    }

    reliability_score, reliability_band, reliability_warnings, reliability_details = detect_reliability(
        rows,
        summary,
        label_column,
    )

    player_role_counts, player_role_timeline, match_families = collect_player_counters(rows, label_column)
    label_segments = build_label_segments(rows, label_column)
    family_segments = build_family_segments(rows, label_column)
    match_sequence = analyze_match_sequence(family_segments, args.fps)

    player_reports = []
    for player_id, player_name in ((1, args.player_1_name), (2, args.player_2_name)):
        role_counter = player_role_counts[player_id]
        metrics = build_player_metrics(role_counter)
        role_segments = build_role_segments(player_role_timeline[player_id])
        passing = analyze_passing(role_segments, metrics["total_attributed_frames"], args.fps)
        guard_play = analyze_guard_play(role_segments, metrics["total_attributed_frames"], args.fps)
        sequence_info = analyze_player_sequence(player_id, role_segments, label_segments, metrics, args.fps)
        player_reports.append(
            build_player_report(
                player_id, player_name, role_counter, metrics, sequence_info, passing, guard_play, args.fps
            )
        )

    output_dir = Path(args.out_dir) if args.out_dir else base_dir
    ensure_dir(output_dir)

    analysis_json_path = output_dir / "style_analysis.json"
    report_txt_path = output_dir / "style_report.txt"
    report_md_path = output_dir / "style_report.md"
    metrics_csv_path = output_dir / "style_metrics.csv"

    analysis = {
        "input": {
            "position_dir": str(base_dir),
            "summary_json": str(summary_path),
            "positions_csv": str(csv_path),
            "label_column": label_column,
            "fps": args.fps,
            "confidence_filter": confidence_info,
        },
        "source_summary": summary,
        "reliability": {
            "score": round(reliability_score, 3),
            "band": reliability_band,
            "warnings": list(reliability_warnings),
            "details": reliability_details,
        },
        "match_overview": {
            "frames_analyzed": len(rows),
            "seconds_analyzed": frames_to_seconds(len(rows), args.fps),
            "top_position_families": [
                {
                    "family": family,
                    "label": family_name(family),
                    "frames": float(value),
                    "seconds": frames_to_seconds(float(value), args.fps),
                    "share_percent": pct(float(value) / max(1, len(rows))),
                }
                for family, value in top_items(match_families, limit=8)
            ],
        },
        "match_sequence": match_sequence,
        "players": player_reports,
    }

    with analysis_json_path.open("w", encoding="utf-8") as handle:
        json.dump(analysis, handle, indent=2)

    write_metrics_csv(metrics_csv_path, build_metrics_rows(player_reports))

    report_text = build_text_report(
        summary=summary,
        base_dir=base_dir,
        csv_path=csv_path,
        label_column=label_column,
        rows=rows,
        reliability_score=reliability_score,
        reliability_band=reliability_band,
        reliability_warnings=reliability_warnings,
        match_sequence=match_sequence,
        player_reports=player_reports,
        fps=args.fps,
        confidence_info=confidence_info,
    )
    with report_txt_path.open("w", encoding="utf-8") as handle:
        handle.write(report_text)

    if args.markdown:
        report_md = build_markdown_report(
            summary=summary,
            base_dir=base_dir,
            csv_path=csv_path,
            label_column=label_column,
            rows=rows,
            reliability_score=reliability_score,
            reliability_band=reliability_band,
            reliability_warnings=reliability_warnings,
            match_sequence=match_sequence,
            player_reports=player_reports,
            fps=args.fps,
            confidence_info=confidence_info,
        )
        with report_md_path.open("w", encoding="utf-8") as handle:
            handle.write(report_md)

    print("Saved:", analysis_json_path)
    print("Saved:", report_txt_path)
    if args.markdown:
        print("Saved:", report_md_path)
    print("Saved:", metrics_csv_path)
    print(f"Label column used: {label_column}")
    print(f"FPS: {args.fps}")
    if args.min_conf > 0:
        print(f"Confidence filter: {conf_column} >= {args.min_conf} (dropped {dropped_low_conf}/{rows_total})")
    print(f"Attribution reliability: {reliability_band} ({reliability_score:.2f})")
    print(
        "Sequence summary:",
        f"{match_sequence['segment_count']} family phases, avg {match_sequence['average_phase_frames']} frames "
        f"({format_seconds(float(match_sequence.get('average_phase_seconds', 0.0)))})",
    )


if __name__ == "__main__":
    main()
