"""gta6-twitch-ccu-forecast-01 entry: scale-conditioned launch-curve forecaster (Python 3.13, stdlib only).

Forecasts, for each of hours 2..24 of a game's release day (UTC), five quantiles of the Twitch
concurrent-viewer count in the game's category, from hour 1's print and the past big launches
the case carries as `comparables`.

Method
------
1.  Each comparable launch contributes its release-day curve: the 24 hourly Twitch category
    averages aligned to its own `day_start`. Only comparables with all 24 hours are used.
2.  Launches are split by how they unlocked, read from the comparable's own `unlock` text:
    "rolling midnight local" launches (consoles unlock at midnight in each timezone, so hour 1
    already holds the Asia-Pacific + Europe audience) versus everything else (one simultaneous
    global unlock, early-access-contaminated launches, unverified). The targets of this contest
    (the four practice launches and GTA VI itself) are all rolling-midnight console launches,
    so the shape is fitted on the rolling subset only. If fewer than 8 rolling comparables are
    found, all complete curves are used instead (safety net).
3.  Per hour h = 2..24, an OLS regression in logs over the rolling comparables:

        log(viewers_h / viewers_1) = a_h + b_h * log(viewers_1)

    The slope is the finding that matters: it is strongly negative (about -0.28 at hour 6,
    about -0.49 at hour 21 on the practice comparables). A mega launch's hour 1 is already
    enormous, so its curve stays flat or falls (Elden Ring: hour-1 875k, hour-21 ratio 0.90;
    Cyberpunk 2077: 852k, 0.94); a smaller launch's hour 1 is only the early timezones, and
    the US prime-time ramp multiplies it several-fold (Resident Evil Requiem: hour-1 46k,
    hour-21 ratio 11.48; Ghost of Yotei: 27k, 3.62). A pooled median curve ignores this and
    is wrong for both ends. The slope is shrunk 20% toward 0 (toward the pooled median
    curve) and the intercept re-centred, which leave-one-out testing showed beats the
    unshrunk fit.
4.  Quantiles are lognormal around the regression median: q_tau = median * exp(z_tau * s_h),
    z = (-1.645, -0.674, 0, 0.674, 1.645). The spread s_h is the regression's residual
    standard deviation, widened by the standard OLS prediction variance
    sqrt(1 + 1/n + (x - x_mean)^2 / Sxx), so the spread is honest when the target's hour-1
    print sits at or beyond the edge of the fitted range (GTA VI's will be the largest ever).
5.  If no comparable yields a usable curve at all, the forecast falls back to the house
    persistence baseline (every hour at hour 1's print, spread 0.15 * sqrt(h - 1)), so the
    program always writes a valid forecast.

The program is deterministic (pure arithmetic, no randomness, comparables processed in
name order), reads every `<id>.in` in the input directory and writes one `<id>.out` each.
It uses only the standard library and no network.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0) in python:3.13-slim.
-P keeps this file's directory OFF sys.path, so a module shipped beside main.py imports only
after the sys.path line below, which puts the directory first; -s drops user site-packages.

Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR stand in for
/input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py

Validation (see tools/backtest.py): leave-one-out over the 37 past launches with complete
release-day curves in the kit's practice case, scored with the entry's own forecast code.
On the 12 rolling-midnight launches (the regime every scored case belongs to) the model
scores a mean pinball loss of 0.247 against 0.357 for the house persistence baseline
(median 0.167 vs 0.255), beating it on 10 of the 12 launches.
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model` finds a
# model.py shipped beside this file (there is none today; the line stays for that reason).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]
# The house baseline's spread, used by the fallback and as a floor reference.
BASELINE_SIGMA = 0.15
# The regression slope is shrunk this much toward 0 (leave-one-out optimum on the kit data).
SLOPE_SHRINK = 0.8
# Minimum rolling comparables before the fit is trusted; below it, use every complete curve.
MIN_ROLLING = 8


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100 rounds it.

    A whole number comes back as an int, so json.dumps writes 1240 where JavaScript writes
    1240 (not 1240.0).
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def parse_ts(text: str) -> datetime:
    """An ISO 8601 UTC timestamp, with or without a trailing Z."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def day_curve(comp: dict) -> dict[int, float] | None:
    """A comparable's release-day curve: {hour 1..24: viewers}, or None if incomplete.

    The curve is the comparable's hourly points aligned to its own day_start. A point
    counts for the hour its `from` starts; viewers below 1 count as 1 (the scorer's floor).
    """
    try:
        day_start = parse_ts(comp["day_start"])
    except (KeyError, TypeError, ValueError):
        return None
    hours: dict[int, float] = {}
    for point in comp.get("hourly") or []:
        try:
            when = parse_ts(point["from"])
            viewers = float(point["viewers"])
        except (KeyError, TypeError, ValueError):
            continue
        delta_hours = (when - day_start).total_seconds() / 3600
        if delta_hours == int(delta_hours) and 0 <= delta_hours < 24:
            hours[int(delta_hours) + 1] = max(1.0, viewers)
    return hours if len(hours) == 24 else None


def is_rolling(comp: dict) -> bool:
    """True when the comparable unlocked at midnight local time, rolling by timezone.

    Read from the comparable's own `unlock` text, the only unlock information the case
    carries. Rolling-midnight launches are the regime every scored case belongs to.
    """
    text = str(comp.get("unlock") or "").lower()
    return ("midnight local" in text) or ("rolling" in text) or ("12 am local" in text)


def ols(xs: list[float], ys: list[float]) -> tuple[float, float, float, float, float]:
    """Ordinary least squares y ~ a + b*x: (intercept, slope, resid SD, mean x, Sxx)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else 0.0
    intercept = my - slope * mx
    resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    dof = max(1, n - 2)
    resid_sd = math.sqrt(sum(e * e for e in resid) / dof)
    return intercept, slope, resid_sd, mx, sxx


def baseline_quantiles(p1: float, h: int) -> list[float | int]:
    """The house persistence baseline for one hour: every hour at hour 1's print."""
    return [round2(p1 * math.exp(z * BASELINE_SIGMA * math.sqrt(h - 1))) for z in Z]


def forecast_case(case: dict) -> dict:
    """The forecast for one case: {"hours": [{"hour": h, "q": [q05..q95]}, ...]}."""
    first = next((k for k in case.get("known") or [] if k.get("hour") == 1), None)
    p1 = max(1.0, float(first["viewers"])) if first is not None else 1.0
    target_name = str((case.get("game") or {}).get("name") or "").lower()

    # The training set: complete release-day curves from the case's comparables. The
    # target's own curve can never be among them (it does not exist yet), but skip any
    # comparable that names the target's game anyway, so no leakage is possible.
    curves: list[tuple[bool, dict[int, float]]] = []
    for comp in case.get("comparables") or []:
        if str(comp.get("name") or "").lower() == target_name:
            continue
        curve = day_curve(comp)
        if curve is not None:
            curves.append((is_rolling(comp), curve))
    rolling = [curve for is_roll, curve in curves if is_roll]
    train = rolling if len(rolling) >= MIN_ROLLING else [curve for _, curve in curves]

    if len(train) < 3:
        # Too little history to fit anything: fall back to the house baseline.
        return {"hours": [{"hour": h, "q": baseline_quantiles(p1, h)}
                          for h in range(2, 25)]}

    log_p1 = math.log(p1)
    log_h1 = [math.log(curve[1]) for curve in train]
    hours_out = []
    for h in range(2, 25):
        log_ratio = [math.log(curve[h] / curve[1]) for curve in train]
        intercept, slope, resid_sd, mean_x, sxx = ols(log_h1, log_ratio)
        # Shrink the slope toward 0 and re-centre the intercept on the training means,
        # so the shrunk line still passes through the middle of the training data.
        slope *= SLOPE_SHRINK
        intercept = sum(log_ratio) / len(log_ratio) - slope * mean_x
        median_log = intercept + slope * log_p1
        # Honest spread: residual SD widened by the OLS prediction variance, so a
        # target at the edge of (or beyond) the fitted hour-1 range gets a wider band.
        n = len(train)
        leverage = 1.0 + 1.0 / n + ((log_p1 - mean_x) ** 2 / sxx if sxx else 0.0)
        sigma = max(0.10, resid_sd * math.sqrt(leverage))
        quantiles = []
        for z in Z:
            q = p1 * math.exp(median_log + z * sigma)
            quantiles.append(round2(min(1e9, max(1.0, q))))
        # Rounding cannot break the order (exp is monotone, round2 is monotone), but
        # enforce it anyway: a row must be non-decreasing to be valid.
        for i in range(1, 5):
            if quantiles[i] < quantiles[i - 1]:
                quantiles[i] = quantiles[i - 1]
        hours_out.append({"hour": h, "q": quantiles})
    return {"hours": hours_out}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
