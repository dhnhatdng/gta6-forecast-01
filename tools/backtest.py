"""Leave-one-out backtest for the gta6-twitch-ccu-forecast-01 entry.

The kit ships no local grader (the server grades every entry), and the four practice
cases' answers are sealed on the server. What the kit DOES ship is better for validation:
every practice case carries 38 past launches as `comparables`, 37 of which have a complete
24-hour release-day curve. This script treats each of those 37 launches as a pseudo-target
in turn — hour 1's print as the only thing known, the other 36 launches as the training
set — runs the entry's own `forecast_case` on it, and scores the forecast against the
launch's real hours 2..24 with the contest's pinball loss on the log scale. It also scores
the house persistence baseline the same way, for comparison.

Usage (from the repository root):
    python3 tools/backtest.py [path/to/p00.in]

The kit's practice case is the default input. Exit code 0 when the model beats the
baseline on the rolling-midnight subset (the regime every scored case belongs to).
"""
import json
import math
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from main import baseline_quantiles, day_curve, forecast_case, is_rolling  # noqa: E402

KIT_INPUT = Path(
    "/home/hatch/workspace/makefun/gta6-forecast-01-kit/"
    "gta6-twitch-ccu-forecast-01-kit/input/p00.in"
)
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]


def pinball(y: float, q: float, tau: float) -> float:
    d = math.log(max(1.0, y)) - math.log(max(q, 1e-9))
    return 2 * max(tau * d, (tau - 1) * d)


def loss_of(forecast: dict, actual: dict[int, float]) -> float:
    total, n = 0.0, 0
    rows = {row["hour"]: row["q"] for row in forecast["hours"]}
    for h in range(2, 25):
        for tau, q in zip(LEVELS, rows[h]):
            total += pinball(actual[h], q, tau)
            n += 1
    return total / n


def median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def main() -> int:
    case_path = Path(sys.argv[1]) if len(sys.argv) > 1 else KIT_INPUT
    case = json.loads(case_path.read_text(encoding="utf-8"))
    comparables = case["comparables"]

    usable = []
    for comp in comparables:
        curve = day_curve(comp)
        if curve is not None:
            usable.append((comp, curve))
    print(f"comparables with a complete 24h release-day curve: {len(usable)}")

    results = []
    for target_comp, target_curve in usable:
        others = [c for c in comparables if c is not target_comp]
        fake_case = {
            "game": {"name": target_comp["name"], "twitch_id": "0"},
            "known": [{"hour": 1, "viewers": target_curve[1]}],
            "comparables": others,
        }
        model_loss = loss_of(forecast_case(fake_case), target_curve)
        base = {"hours": [{"hour": h, "q": baseline_quantiles(target_curve[1], h)}
                          for h in range(2, 25)]}
        base_loss = loss_of(base, target_curve)
        results.append((target_comp["name"], is_rolling(target_comp),
                        target_curve[1], model_loss, base_loss))

    for label, subset in (("ROLLING-MIDNIGHT (every scored case is this regime)",
                           [r for r in results if r[1]]),
                          ("ALL", results)):
        m = [r[3] for r in subset]
        b = [r[4] for r in subset]
        wins = sum(1 for r in subset if r[3] < r[4])
        print(f"\n{label}: n={len(subset)}")
        print(f"  model    mean {sum(m)/len(m):.4f}  median {median(m):.4f}")
        print(f"  baseline mean {sum(b)/len(b):.4f}  median {median(b):.4f}")
        print(f"  model beats baseline on {wins}/{len(subset)} launches")

    print("\nper-launch (rolling marked *):")
    for name, rolling, h1, ml, bl in sorted(results, key=lambda r: r[3]):
        mark = "*" if rolling else " "
        print(f"  {mark} {name[:40]:40s} h1={h1:7.0f} model {ml:.4f} baseline {bl:.4f}")

    rolling = [r for r in results if r[1]]
    ok = (sum(r[3] for r in rolling) / len(rolling)
          < sum(r[4] for r in rolling) / len(rolling))
    print("\nVERDICT:", "model beats baseline on rolling subset" if ok
          else "model does NOT beat baseline — do not submit")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
