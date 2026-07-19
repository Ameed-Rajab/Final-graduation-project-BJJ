# Player Style Analysis

## Input
- **Position directory:** `C:\Users\Ameed\Downloads\Compressed\bjj_project\bjj_project\outputs_all_clips\position_predictions`
- **CSV:** `C:\Users\Ameed\Downloads\Compressed\bjj_project\bjj_project\outputs_all_clips\position_predictions\positions.csv`
- **Label column:** `pred_fixed`
- **FPS assumed:** 30.0
- **Attribution reliability:** **high** (1.00)

## Match Sequence
- 6 family phases, avg 74.67 frames (2.5s)
- **Opening:** Half guard → Closed guard → Half guard → Closed guard → Half guard → 50/50 guard
- **Ending:** Half guard → Closed guard → Half guard → Closed guard → Half guard → 50/50 guard

### Top transitions
| Pattern | Count |
|---|---|
| Half guard -> Closed guard | 2 |
| Closed guard -> Half guard | 2 |
| Half guard -> 50/50 guard | 1 |

### Longest phases
| Family | Frames | Seconds | Frame range |
|---|---|---|---|
| Half guard | 200 | 6.7s | 19–218 |
| Half guard | 122 | 4.1s | 220–341 |
| 50/50 guard | 106 | 3.5s | 342–447 |
| Half guard | 18 | 0.6s | 0–17 |
| Closed guard | 1 | 0.0s | 18–18 |

### Read
- The match features long control phases once a dominant position is established.

## Player 1
- **Primary profile:** Guard passer (half guard passer)
- **Sequence profile:** Guard/turtle recycler
- **Secondary traits:** recycles exchanges through recovery loops, plays half guard, 50/50 involvement
- **Attributed time:** 448 frames (14.9s)

### Position share (by role)
| Role | Percent |
|---|---|
| Guard passing (top) | 43.8% |
| Dominant top (side/mount/back) | 0.0% |
| Guard playing (bottom) | 32.6% |
| Bottom defense (under control) | 0.0% |
| Turtle / scramble | 0.0% |
| 50/50 (neutral) | 23.7% |
| Overall top / bottom | 43.8% / 32.6% |

### Guard passing breakdown (their top game)
| Guard | Share of top time | Attempts | Reach control | Avg time to pass |
|---|---|---|---|---|
| half guard | 43.5% | 51 | 0.0% | — |
| closed guard | 0.2% | 1 | 0.0% | — |

### Guard defense breakdown (their bottom game)
| Guard | Share of bottom time | Instances | Retained/recovered | Swept up |
|---|---|---|---|---|
| half guard | 32.4% | 50 | 0.0% | 100.0% |
| closed guard | 0.2% | 1 | 0.0% | 100.0% |

### Progression
| Metric | Value |
|---|---|
| Advances | 51 |
| Concessions | 51 |
| Net progression | +1.0 |
| Biggest advance | +2.0 |
| Biggest concession | -2.0 |

### Key positions
| Family | Frames | Seconds | Share |
|---|---|---|---|
| Half guard (passing) | 195 | 6.5s | 43.5% |
| Half guard (bottom) | 145 | 4.8s | 32.4% |
| 50/50 guard | 106 | 3.5s | 23.7% |
| Closed guard (bottom) | 1 | 0.0s | 0.2% |
| Closed guard (passing) | 1 | 0.0s | 0.2% |

- **Control tempo:** avg owned phase 4.31 frames (0.1s), longest 106 frames (3.5s)
- **Sequence read:**
  - Frequently revisits guard or turtle instead of letting the exchange end cleanly.
- **Recurring transitions:** Half guard -> Half guard (98), Half guard -> Closed guard (2), Closed guard -> Half guard (2), Half guard -> 50/50 guard (1)
- **Recurring chains:** Half guard -> Half guard -> Half guard (95), Half guard -> Half guard -> Closed guard (2), Half guard -> Closed guard -> Half guard (2), Closed guard -> Half guard -> Half guard (2)
- **Efficient counter style:** Deny their half guard pass and steer them into a guard they finish less often

**Game plan:**
- They pass half guard most (43.5% of top time, 0.0% reach control) — fight for the underhook and knee-shield early so they cannot clear your knee-line.
- They spend 23.7% in 50/50 — clear your knee-line early and avoid stalling in equal leg positions.

## Player 2
- **Primary profile:** Guard player (half guard bottom)
- **Sequence profile:** Guard/turtle recycler
- **Secondary traits:** recycles exchanges through recovery loops, passes half guard, 50/50 involvement
- **Attributed time:** 448 frames (14.9s)

### Position share (by role)
| Role | Percent |
|---|---|
| Guard passing (top) | 32.6% |
| Dominant top (side/mount/back) | 0.0% |
| Guard playing (bottom) | 43.8% |
| Bottom defense (under control) | 0.0% |
| Turtle / scramble | 0.0% |
| 50/50 (neutral) | 23.7% |
| Overall top / bottom | 32.6% / 43.8% |

### Guard passing breakdown (their top game)
| Guard | Share of top time | Attempts | Reach control | Avg time to pass |
|---|---|---|---|---|
| half guard | 32.4% | 50 | 0.0% | — |
| closed guard | 0.2% | 1 | 0.0% | — |

### Guard defense breakdown (their bottom game)
| Guard | Share of bottom time | Instances | Retained/recovered | Swept up |
|---|---|---|---|---|
| half guard | 43.5% | 51 | 3.9% | 96.1% |
| closed guard | 0.2% | 1 | 0.0% | 100.0% |

### Progression
| Metric | Value |
|---|---|
| Advances | 51 |
| Concessions | 51 |
| Net progression | -1.0 |
| Biggest advance | +2.0 |
| Biggest concession | -2.0 |

### Key positions
| Family | Frames | Seconds | Share |
|---|---|---|---|
| Half guard (bottom) | 195 | 6.5s | 43.5% |
| Half guard (passing) | 145 | 4.8s | 32.4% |
| 50/50 guard | 106 | 3.5s | 23.7% |
| Closed guard (passing) | 1 | 0.0s | 0.2% |
| Closed guard (bottom) | 1 | 0.0s | 0.2% |

- **Control tempo:** avg owned phase 4.31 frames (0.1s), longest 106 frames (3.5s)
- **Sequence read:**
  - Frequently revisits guard or turtle instead of letting the exchange end cleanly.
- **Recurring transitions:** Half guard -> Half guard (98), Half guard -> Closed guard (2), Closed guard -> Half guard (2), Half guard -> 50/50 guard (1)
- **Recurring chains:** Half guard -> Half guard -> Half guard (95), Half guard -> Half guard -> Closed guard (2), Half guard -> Closed guard -> Half guard (2), Closed guard -> Half guard -> Half guard (2)
- **Efficient counter style:** Structured passing that clears their half guard and denies the recovery

**Game plan:**
- They play half guard 43.5% of bottom time, holding or recovering it 3.9% of the time — control the inside knee and head, clear the knee-line, then flatten before settling chest pressure.
- They sweep or wrestle-up from half guard 96.1% of the time — kill the underhook and keep your weight back before passing.
- They spend 23.7% in 50/50 — clear your knee-line early and avoid stalling in equal leg positions.

_Rows analyzed: 448_
