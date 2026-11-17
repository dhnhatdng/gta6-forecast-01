# gta6-forecast-01

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/dhnhatdng/gta6-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/dhnhatdng/gta6-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

A forecaster for GTA VI's launch-day Twitch viewers, hour by hour. Given a case — hour 1's
viewer count plus the past big launches it carries as comparables — it writes five quantiles
(5%, 25%, 50%, 75%, 95%) of the viewer count for each of hours 2 to 24 of the release day.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only, no network. Reads every `<id>.in` in the input
directory and writes one `<id>.out` each. Deterministic: two runs write byte-identical
output, in about 0.05 s per case (limit 8 s).

## Method

The house baseline says every hour stays at hour 1's count. That is wrong in a systematic
way: a launch day has a shape, and the shape depends on how big hour 1 already is.

1. **Training curves.** Each comparable launch contributes its release-day curve: 24 hourly
   Twitch category averages aligned to its own `day_start`. Only comparables with all 24
   hours are used (37 of 38 in the kit's practice case).
2. **Regime split.** Launches are split by how they unlocked, read from the comparable's
   own `unlock` text. Rolling-midnight launches (consoles unlock at midnight in each
   timezone, so hour 1 already holds the Asia-Pacific + Europe audience) are fitted
   separately from simultaneous-global-unlock and early-access-contaminated launches. The
   targets of this contest — the four practice launches and GTA VI — are all
   rolling-midnight console launches, so the shape is fitted on that subset only (12
   launches in the kit). Fewer than 8 rolling comparables falls back to all complete curves.
3. **Scale-conditioned shape.** Per hour h = 2..24, an OLS regression in logs over the
   rolling comparables: `log(viewers_h / viewers_1) = a_h + b_h * log(viewers_1)`. The slope
   is strongly negative (about −0.28 at hour 6, about −0.49 at hour 21): a mega launch's
   hour 1 is already enormous, so its curve stays flat or falls (Elden Ring: hour-1 875k,
   hour-21 ratio 0.90; Cyberpunk 2077: 852k, 0.94), while a smaller launch's hour 1 is only
   the early timezones and the US prime-time ramp multiplies it several-fold (Resident Evil
   Requiem: hour-1 46k, hour-21 ratio 11.48). A pooled median curve ignores this and is
   wrong at both ends. The slope is shrunk 20% toward 0 and the intercept re-centred.
4. **Honest spread.** Quantiles are lognormal around the regression median,
   `q_tau = median * exp(z_tau * s_h)`, with the spread from the regression's residual SD
   widened by the OLS prediction variance — so a target at or beyond the edge of the
   fitted hour-1 range (GTA VI's will be the largest ever) gets a wider band. If no
   comparable yields a usable curve, the forecast falls back to the house persistence
   baseline, so the program always writes a valid forecast.

## Validation

The kit ships no local grader (the server grades every entry) and the practice answers are
sealed, so validation is leave-one-out over the 37 past launches with complete release-day
curves (`python3 tools/backtest.py`, which scores the entry's own `forecast_case` code
against each launch's real hours 2..24 with the contest's pinball loss):

| Set | Model mean loss | Baseline mean loss | Model median | Baseline median |
| --- | --- | --- | --- | --- |
| Rolling-midnight launches (n=12, the regime of every scored case) | **0.247** | 0.357 | **0.167** | 0.255 |
| All launches (n=37) | **0.590** | 0.930 | **0.319** | 0.325 |

The model beats the house persistence baseline on 10 of the 12 rolling-midnight launches
and on 26 of all 37. The baseline's published practice loss is about 0.387; the model's
leave-one-out loss on the same regime is 0.247.

Limits, stated plainly: the shape is fitted on 12 rolling launches, so the slope is noisy
and the forecast for GTA VI extrapolates beyond the largest hour-1 print in the data
(875k) — the widened spread is the honest answer to that, not a fix. If launch day has
no dip of attention at all (a flat day), the persistence baseline would win; the model
bets, with the data behind it, that scale predicts the shape.

---

<sub>Entered on [make.fun](https://make.fun/?ref=fs4ajvvm&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
