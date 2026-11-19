# MLB Lineup Optimization Model

A win-probability model that rebuilds every MLB lineup card against an optimal alternative and scores each manager's lineup decisions in wins. Built in Python on Statcast data for the 2026 season, covering roughly 5,000 scored decisions across all 30 teams.

The core question: when a manager writes a lineup card, how much does it cost or gain in win probability compared to the best lineup he could have written from the same roster? And separately, does diverging from the model actually pay off, or just look different?

---

## What the model does

For every game, the model builds the win-probability-optimal batting order given the opposing starter, the projected bullpen behind him, and the hitters available on the roster that day. It then compares that order to the one the manager actually submitted, and scores the gap.

Every projection is point-in-time. For a given date, the model uses only data that existed before that date, so nothing leaks backward from results it could not have known. That constraint runs through the whole pipeline (splits, pitch mix, and the bullpen projection all respect it).

### How a hitter is projected

Each hitter gets an expected wOBA against the specific matchup he faces, built through a four-level regression cascade:

1. **Projection / league baseline** as the starting anchor.
2. **Overall wOBA**, regressed toward that anchor.
3. **Same-handedness wOBA** (vs LHP or vs RHP), regressed toward the overall.
4. **Pitch-type wOBA**, regressed toward the hand value, then blended across the pitches the starter actually throws.

The final projection weights the pitch-type component 70% and the hand component 30%. Each level is shrunk toward the level above it using a Bayesian-style prior, so a small sample pulls toward a trustworthy baseline instead of overreacting (`regress()` in `platoon_core.py`). Defense is folded in through Outs Above Average, scaled to wOBA units, so a glove-first bat is not overvalued on his platoon split alone.

### The bullpen correction

The model does not assume the starter pitches the whole game. It projects how many plate appearances each lineup faces against the starter versus the bullpen, and weights the hitter's value across both:

- The **starter's share** comes from his recent trailing batters-faced (`starter_pa_weight()`), so an opener collapses toward zero and a workhorse approaches ~0.7.
- The **relief share** is re-weighted to the handedness the bullpen actually throws, projected point-in-time from the same relievers' recent usage with a prior-season shrinkage fallback for early in the year (`projected_pen_rhp_share()`).

A lineup that is optimal against a lefty starter can be wrong once you account for the righty-heavy pen behind him. This correction moved the projections more than any other single change; earlier versions that booked the whole game at the starter's hand overstated platoon value by roughly 21 to 35 percent. The blend itself is `bullpen_blend()` in `platoon_core.py`.

### Scoring the decision

Every lineup is scored in win-probability terms, so the output is "how many wins the gap is worth," not "this order looks better." Each managerial divergence (a starter the manager played whom the model would have benched, paired with the model's preferred alternative at the same position) is scored by comparing the manager's realized WPA to the model pick's expected WPA. Pairings are constrained to players who could actually occupy the same lineup spot, so a shortstop is never scored as "replaced by" a catcher.

---

## What it found (2026 season, as of the July sample)

- **~2,906 manager divergences** from the model across all 30 teams, each scored in WPA.
- The spread from the best to the worst dugout was about **7.6 wins of decision value**.
- Pittsburgh's divergences added roughly **4.2 wins** over the model baseline; Colorado's cost about **3.4**.
- **Construction and in-game instinct are separable traits.** Kansas City built suboptimal cards relative to the model, yet its divergences netted strongly positive, it out-managed its own lineups. Detroit and Chicago (AL) did the opposite, diverging often and getting burned.

An honest caveat, stated plainly: WPA over a single season is noisy, and its leverage component is largely non-repeatable. So this ranks decision **outcomes**, not proven manager skill. The model shows which dugouts systematically leave value on the table and which specific bats are being benched into worse matchups, both in theory (the projected gap) and in realized results.

The published charts are in [`charts/`](charts/), including the divergence scatter (how often each manager departs from the model, and whether those departures added wins), the best-vs-worst dugout gap over the season, and the player-level value chart. A sample of the raw output is in [`sample_output/`](sample_output/): the scored decision log and the per-game lineup gaps.

---

## Repository layout

```
src/
  platoon_core.py      The model math. Pure functions, no network calls: the
                       regression cascade, composite scoring, the bullpen blend,
                       optimal-lineup selection, and decision pairing. This is the
                       single source of truth the pipeline is built on.
  backfill_season.py   The point-in-time pipeline. Reconstructs every day of the
                       season using only data available at the time, and produces
                       the scored decision and season logs. This is the complete,
                       bullpen-aware pipeline that generated the published results.
charts/                The published charts from the writeup.
sample_output/         A sample of the generated decision and season logs.
```

---

## Running it

This is a personal research pipeline, not a packaged tool, so a couple of honest notes on running it yourself.

**Dependencies:** Python 3.10+, `pandas`, `requests`, and `pybaseball` (which provides the Statcast pulls). Install with:

```
pip install pandas requests pybaseball
```

**Data sources:** the pipeline pulls live from Baseball Savant and the MLB Stats API. It writes its CSV outputs to a local directory (currently the Desktop; that path is set at the top of each script and is the one thing to change to run it elsewhere). The scripts cache Statcast chunks through `pybaseball`, so a re-run resumes quickly rather than re-downloading.

**A first run:** the backfill accepts a date range, and a single day is the recommended first test:

```
python3 src/backfill_season.py 2026-05-15 2026-05-15
```

The model logic in `platoon_core.py` is pure and has no network dependencies, so its functions can be imported and exercised directly without any data pull.

---

## Design notes a reader might care about

A few decisions in here were deliberate and are worth calling out, because they are the difference between a model that looks right and one that is:

- **Exactly-zero is a real value, not missing data.** A hitter who is 0-for-12 has a wOBA of 0.000, which is information, not a gap. Several functions guard with `is None` / `isna` rather than a falsy test, because an early version treated a true 0.000 as missing and silently discarded real signal (a player's entire defensive value, in one case).
- **A data gap is scored neutrally, not as a penalty.** When a player is missing from the lookups, he is scored at league average rather than zero, so a missing record never fabricates a swing in the lineup gap. An earlier version that scored gaps as zero produced impossible "the actual lineup beat the optimal one" results.
- **The optimal lineup must be able to take the field.** Lineup selection verifies the chosen eight can legally cover every position and backtracks if not, instead of greedily filling slots and stranding a scarce position (which once produced a lineup with three catchers and nobody at second base).
- **Rookies are scored, not skipped.** A model pick with no trailing-year history regresses to zero rather than being dropped. Dropping them biased the backtest, because the model tends to pick bench players, who skew young, so skipping them gave managers a free pass on exactly that category.

The comments throughout `platoon_core.py` and the pipeline scripts document these and the reasoning behind each, including the bugs they fixed.

---

*Built by Kiran Chokshi. A companion LinkedIn writeup walks through the findings and charts for a general audience.*
