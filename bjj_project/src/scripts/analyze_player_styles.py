import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple



DEFAULT_POSITION_DIR = Path("outputs/detr_ioutrack_test_images/position_predictions")

GUARD_FAMILIES = {"open_guard", "closed_guard", "half_guard"}
TOP_FAMILIES = {"side_control", "mount", "back", "takedown"}
FINISH_FAMILIES = {"mount", "back"}

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

    if label_column != "pred_fixed" and bool(summary.get("enforce_bottom_2")):
        warnings.append(
            "A role-fixed column appears to be available, but the selected label column is not pred_fixed."
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


def collect_player_counters(rows: Sequence[Dict[str, str]], label_column: str) -> Tuple[Dict[int, Counter], Counter]:
    player_families = {1: Counter(), 2: Counter()}
    match_families: Counter = Counter()

    for row in rows:
        family, player_id = parse_label(row[label_column])
        match_families[family] += 1.0

        if player_id is None:
            player_families[1][family] += 0.5
            player_families[2][family] += 0.5
        else:
            player_families[player_id][family] += 1.0

    return player_families, match_families


def build_player_metrics(counter: Counter) -> Dict[str, float]:
    total = float(sum(counter.values()))
    if total <= 0:
        total = 1.0

    guard = sum(float(counter.get(fam, 0.0)) for fam in GUARD_FAMILIES)
    top = sum(float(counter.get(fam, 0.0)) for fam in TOP_FAMILIES)
    turtle = float(counter.get("turtle", 0.0))
    leg = float(counter.get("5050_guard", 0.0))
    standing = float(counter.get("standing", 0.0))
    mount = float(counter.get("mount", 0.0))
    back = float(counter.get("back", 0.0))
    takedown = float(counter.get("takedown", 0.0))
    side_control = float(counter.get("side_control", 0.0))

    return {
        "total_attributed_frames": total,
        "guard_frames": guard,
        "top_frames": top,
        "turtle_frames": turtle,
        "leg_entanglement_frames": leg,
        "standing_frames": standing,
        "mount_frames": mount,
        "back_frames": back,
        "takedown_frames": takedown,
        "side_control_frames": side_control,
        "guard_ratio": guard / total,
        "top_ratio": top / total,
        "turtle_ratio": turtle / total,
        "leg_ratio": leg / total,
        "standing_ratio": standing / total,
        "finish_ratio": (mount + back) / total,
        "wrestling_ratio": (takedown + 0.4 * standing) / total,
    }


def primary_profile_name(counter: Counter, metrics: Dict[str, float]) -> str:
    guard_pref = dominant_family(counter, GUARD_FAMILIES)

    if metrics["guard_ratio"] >= max(metrics["top_ratio"], metrics["turtle_ratio"]) and metrics["guard_ratio"] >= 0.28:
        if guard_pref == "open_guard":
            return "Guard-first player (open-guard leaning)"
        if guard_pref == "closed_guard":
            return "Guard-first player (closed-guard leaning)"
        if guard_pref == "half_guard":
            return "Guard-first player (half-guard leaning)"
        return "Guard-first positional player"

    if metrics["top_ratio"] >= max(metrics["guard_ratio"], metrics["turtle_ratio"]) and metrics["top_ratio"] >= 0.25:
        if metrics["back_frames"] >= metrics["side_control_frames"] + 0.5 * metrics["mount_frames"]:
            return "Top-control player with back-taking bias"
        if metrics["mount_frames"] + metrics["side_control_frames"] >= 0.55 * metrics["top_frames"]:
            return "Pressure top player"
        if metrics["takedown_frames"] >= max(metrics["mount_frames"], metrics["side_control_frames"]):
            return "Takedown-to-top player"
        return "Top-control player"

    if metrics["turtle_ratio"] >= 0.20:
        return "Scramble / turtle-heavy player"

    if metrics["leg_ratio"] >= 0.15:
        return "50/50 and leg-entanglement player"

    return "Balanced positional player"


def secondary_traits(
    counter: Counter,
    metrics: Dict[str, float],
    primary_profile: str,
    sequence_profile: str,
) -> List[str]:
    traits: List[Tuple[float, str]] = []

    guard_pref = dominant_family(counter, GUARD_FAMILIES)
    if metrics["guard_ratio"] >= 0.20 and "Guard-first" not in primary_profile:
        if guard_pref == "open_guard":
            traits.append((metrics["guard_ratio"], "open-guard preference"))
        elif guard_pref == "closed_guard":
            traits.append((metrics["guard_ratio"], "closed-guard preference"))
        elif guard_pref == "half_guard":
            traits.append((metrics["guard_ratio"], "half-guard preference"))
        else:
            traits.append((metrics["guard_ratio"], "guard preference"))

    if metrics["top_ratio"] >= 0.20 and "Top" not in primary_profile and "Pressure" not in primary_profile:
        if metrics["back_frames"] >= 0.7 * metrics["finish_ratio"] * metrics["total_attributed_frames"]:
            traits.append((metrics["top_ratio"], "back-taking threat"))
        else:
            traits.append((metrics["top_ratio"], "top-control threat"))

    if metrics["turtle_ratio"] >= 0.18 and "Scramble" not in primary_profile:
        traits.append((metrics["turtle_ratio"], "scramble / turtle exchanges"))

    if metrics["leg_ratio"] >= 0.12 and "50/50" not in primary_profile:
        traits.append((metrics["leg_ratio"], "50/50 involvement"))

    if metrics["wrestling_ratio"] >= 0.06:
        traits.append((metrics["wrestling_ratio"], "takedown threat"))

    if metrics["finish_ratio"] >= 0.14 and "back-taking bias" not in primary_profile:
        traits.append((metrics["finish_ratio"], "finishing pressure"))

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


def analyze_match_sequence(family_segments: Sequence[Dict[str, object]]) -> Dict[str, object]:
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
    label_segments: Sequence[Dict[str, object]],
    metrics: Dict[str, float],
) -> Dict[str, object]:
    own_segments = [segment for segment in label_segments if segment["player_id"] == player_id]
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

    avg_duration = float(sum(int(segment["duration_rows"]) for segment in own_segments)) / max(1, len(own_segments))
    longest_duration = max((int(segment["duration_rows"]) for segment in own_segments), default=0)
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
                "label": family_name(str(segment["family"])),
                "duration_rows": int(segment["duration_rows"]),
                "start_frame": int(segment["start_frame"]),
                "end_frame": int(segment["end_frame"]),
            }
            for segment in own_segments
        ],
        key=lambda item: (-item["duration_rows"], item["start_frame"]),
    )[:5]

    return {
        "segment_count": len(own_segments),
        "average_owned_phase_frames": round(avg_duration, 2),
        "longest_owned_phase_frames": int(longest_duration),
        "sequence_profile": profile,
        "narratives": narratives[:4],
        "motifs": motifs,
        "top_transitions": serialize_pair_counts(pair_counts, limit=5),
        "top_chains": serialize_triple_counts(triple_counts, limit=4),
        "longest_owned_phases": longest_segments,
    }


def build_counter_plan(
    counter: Counter,
    metrics: Dict[str, float],
    sequence_info: Dict[str, object],
) -> Dict[str, object]:
    guard_pref = dominant_family(counter, GUARD_FAMILIES)
    motifs = sequence_info["motifs"]
    counter_style = ""
    actions: List[str] = []

    chain_score = motifs["pass_to_finish"] + motifs["finish_cycle"] + motifs["turtle_to_attack"]
    recovery_score = motifs["guard_turtle_loops"] + motifs["guard_reentries"]
    entry_score = motifs["standing_to_takedown"] + motifs["takedown_to_control"] + motifs["takedown_to_reset"]

    if chain_score >= 4 and metrics["top_ratio"] >= 0.20:
        counter_style = "Early guard retention that breaks their pass-to-finish chain"
        actions.extend(
            [
                "Do not concede a settled side-control beat, because they often climb from there into mount or back.",
                "Frame early, turn back inside quickly, and interrupt the second attack before it connects.",
            ]
        )
    elif recovery_score >= 4 and metrics["guard_ratio"] >= 0.20:
        counter_style = "Layered passing that wins the second and third exchange"
        actions.extend(
            [
                "Expect guard or turtle recovery after the first pass attempt and keep chest-to-hips connection.",
                "Finish passes with control, not only movement, so they cannot recycle the exchange.",
            ]
        )
    elif entry_score >= 3 and metrics["wrestling_ratio"] >= 0.05:
        counter_style = "Heavy hand-fighting and reset discipline against repeated entries"
        actions.extend(
            [
                "Make them restart entries from distance instead of allowing one shot to chain into the next phase.",
                "After defending an entry, clear ties and reset stance before chasing the scramble.",
            ]
        )
    elif metrics["guard_ratio"] >= metrics["top_ratio"] and metrics["guard_ratio"] >= 0.25:
        if guard_pref == "open_guard":
            counter_style = "Structured passing with posture and leg-control discipline"
            actions.extend(
                [
                    "Win grips before stepping in, then pin a shin or knee line before committing to the pass.",
                    "Use angle-changing passes instead of driving straight into their open guard.",
                ]
            )
        elif guard_pref == "closed_guard":
            counter_style = "Posture-first passing with safe guard opening"
            actions.extend(
                [
                    "Keep your spine tall, elbows inside, and open the guard before trying to force pressure.",
                    "Do not leave extended arms inside the closed guard where posture breaks and attacks start.",
                ]
            )
        else:
            counter_style = "Patient half-guard passing with knee-line control"
            actions.extend(
                [
                    "Control the inside knee and head position before trying to flatten half guard.",
                    "Clear knee line first, then settle chest pressure once their frames are weaker.",
                ]
            )
    elif metrics["top_ratio"] >= 0.25:
        counter_style = "Mobile guard retention with early frames and wrestle-up threats"
        actions.extend(
            [
                "Frame early and recover distance before they settle chest-to-chest top control.",
                "Make them address off-balancing and wrestle-up threats so they cannot pressure pass freely.",
            ]
        )
    elif metrics["turtle_ratio"] >= 0.20:
        counter_style = "Tight front-headlock pressure and disciplined scramble control"
        actions.extend(
            [
                "Follow hips and shoulders together during scrambles instead of chasing loose hooks.",
                "Secure head-and-arm or near-hip control before attacking the back in turtle exchanges.",
            ]
        )
    elif metrics["leg_ratio"] >= 0.15:
        counter_style = "Leg-entanglement awareness with early knee-line clearing"
        actions.extend(
            [
                "Win the hand fight and clear your knee line before trying to pressure through 50/50 exchanges.",
                "Disengage shallow leg entanglements early instead of stalling in equal positions.",
            ]
        )
    else:
        counter_style = "Force them away from their preferred phase and play first"
        actions.extend(
            [
                "Set the pace early so they spend less time in comfortable neutral exchanges.",
                "Keep transitions tight and avoid loose scrambles that let them re-enter their best positions.",
            ]
        )

    if motifs["guard_turtle_loops"] + motifs["guard_reentries"] >= 2:
        actions.append("Expect a second recovery attempt even after your first pass looks successful.")
    if motifs["pass_to_finish"] + motifs["finish_cycle"] >= 2:
        actions.append("Treat side control as the danger point, because their next step is often mount or back.")
    if motifs["turtle_to_attack"] >= 2:
        actions.append("In scramble phases, connect hips and shoulders before attacking; they counter loose transitions well.")
    if entry_score >= 2:
        actions.append("Keep head position and hand-fighting honest in standing so their entries do not set the pace.")
    if metrics["leg_ratio"] >= 0.12:
        actions.append("Respect 50/50 entries and keep heel-line awareness high during scrambles.")
    if metrics["back_frames"] / max(1.0, metrics["total_attributed_frames"]) >= 0.08:
        actions.append("Hide your back in transitions and keep elbow-knee connection when escaping.")

    return {
        "recommended_style": counter_style,
        "key_actions": dedupe_keep_order(actions)[:5],
    }


def build_player_report(
    player_id: int,
    player_name: str,
    counter: Counter,
    metrics: Dict[str, float],
    sequence_info: Dict[str, object],
) -> Dict[str, object]:
    profile = primary_profile_name(counter, metrics)
    counter_plan = build_counter_plan(counter, metrics, sequence_info)
    traits = secondary_traits(counter, metrics, profile, sequence_info["sequence_profile"])

    top_families_list = [
        {
            "family": family,
            "label": family_name(family),
            "frames": float(value),
            "share_percent": pct(float(value) / max(1.0, metrics["total_attributed_frames"])),
        }
        for family, value in top_items(counter, limit=5)
    ]

    return {
        "player_id": player_id,
        "player_name": player_name,
        "primary_profile": profile,
        "secondary_traits": traits,
        "metrics": {
            "attributed_frames": float(metrics["total_attributed_frames"]),
            "guard_ratio_percent": pct(metrics["guard_ratio"]),
            "top_ratio_percent": pct(metrics["top_ratio"]),
            "turtle_ratio_percent": pct(metrics["turtle_ratio"]),
            "leg_ratio_percent": pct(metrics["leg_ratio"]),
            "wrestling_ratio_percent": pct(metrics["wrestling_ratio"]),
            "finish_ratio_percent": pct(metrics["finish_ratio"]),
        },
        "top_families": top_families_list,
        "sequence": sequence_info,
        "counter_plan": counter_plan,
    }


def build_metrics_rows(player_reports: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    rows = []
    for report in player_reports:
        metrics = report["metrics"]
        sequence_info = report["sequence"]
        motifs = sequence_info["motifs"]
        row = {
            "player_id": report["player_id"],
            "player_name": report["player_name"],
            "primary_profile": report["primary_profile"],
            "secondary_traits": " | ".join(report["secondary_traits"]),
            "sequence_profile": sequence_info["sequence_profile"],
            "attributed_frames": metrics["attributed_frames"],
            "guard_ratio_percent": metrics["guard_ratio_percent"],
            "top_ratio_percent": metrics["top_ratio_percent"],
            "turtle_ratio_percent": metrics["turtle_ratio_percent"],
            "leg_ratio_percent": metrics["leg_ratio_percent"],
            "wrestling_ratio_percent": metrics["wrestling_ratio_percent"],
            "finish_ratio_percent": metrics["finish_ratio_percent"],
            "avg_owned_phase_frames": sequence_info["average_owned_phase_frames"],
            "longest_owned_phase_frames": sequence_info["longest_owned_phase_frames"],
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
) -> str:
    lines: List[str] = []
    lines.append("Player Style Analysis")
    lines.append("")
    lines.append("Input")
    lines.append(f"- Position directory: {base_dir}")
    lines.append(f"- CSV: {csv_path}")
    lines.append(f"- Label column used: {label_column}")
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
        f"- Family phases: {match_sequence['segment_count']} segments, average {match_sequence['average_phase_frames']} frames per phase"
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
        lines.append("")
        lines.append(f"{report['player_name']}")
        lines.append(f"- Primary profile: {report['primary_profile']}")
        lines.append(f"- Sequence profile: {sequence_info['sequence_profile']}")
        if report["secondary_traits"]:
            lines.append(f"- Secondary traits: {', '.join(report['secondary_traits'])}")
        lines.append(
            "- Evidence: "
            f"guard {metrics['guard_ratio_percent']}%, "
            f"top control {metrics['top_ratio_percent']}%, "
            f"turtle/scramble {metrics['turtle_ratio_percent']}%, "
            f"50/50 {metrics['leg_ratio_percent']}%, "
            f"wrestling {metrics['wrestling_ratio_percent']}%"
        )
        lines.append(
            "- Control tempo: "
            f"average owned phase {sequence_info['average_owned_phase_frames']} frames, "
            f"longest owned phase {sequence_info['longest_owned_phase_frames']} frames"
        )
        top_families_text = ", ".join(
            f"{item['label']} {fmt_count(item['frames'])}"
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
    args = parser.parse_args()

    base_dir, summary_path, csv_path = resolve_input_paths(
        position_dir=args.position_dir,
        summary_json=args.summary_json,
        positions_csv=args.positions_csv,
    )

    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    rows = load_rows(csv_path)
    label_column = choose_label_column(rows, args.label_column)

    reliability_score, reliability_band, reliability_warnings, reliability_details = detect_reliability(
        rows,
        summary,
        label_column,
    )

    player_counters, match_families = collect_player_counters(rows, label_column)
    label_segments = build_label_segments(rows, label_column)
    family_segments = build_family_segments(rows, label_column)
    match_sequence = analyze_match_sequence(family_segments)

    player_reports = []
    for player_id, player_name in ((1, args.player_1_name), (2, args.player_2_name)):
        metrics = build_player_metrics(player_counters[player_id])
        sequence_info = analyze_player_sequence(player_id, label_segments, metrics)
        player_reports.append(
            build_player_report(player_id, player_name, player_counters[player_id], metrics, sequence_info)
        )

    output_dir = Path(args.out_dir) if args.out_dir else base_dir
    ensure_dir(output_dir)

    analysis_json_path = output_dir / "style_analysis.json"
    report_txt_path = output_dir / "style_report.txt"
    metrics_csv_path = output_dir / "style_metrics.csv"

    analysis = {
        "input": {
            "position_dir": str(base_dir),
            "summary_json": str(summary_path),
            "positions_csv": str(csv_path),
            "label_column": label_column,
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
            "top_position_families": [
                {
                    "family": family,
                    "label": family_name(family),
                    "frames": float(value),
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
    )
    with report_txt_path.open("w", encoding="utf-8") as handle:
        handle.write(report_text)

    print("Saved:", analysis_json_path)
    print("Saved:", report_txt_path)
    print("Saved:", metrics_csv_path)
    print(f"Label column used: {label_column}")
    print(f"Attribution reliability: {reliability_band} ({reliability_score:.2f})")
    print(
        "Sequence summary:",
        f"{match_sequence['segment_count']} family phases, avg {match_sequence['average_phase_frames']} frames",
    )


if __name__ == "__main__":
    main()