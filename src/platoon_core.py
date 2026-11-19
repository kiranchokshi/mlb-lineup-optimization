#!/usr/bin/env python3
"""
platoon_core.py — Shared model logic for the MLB platoon tool.

This module holds the PURE computation (no network calls): the wOBA regression
cascade, composite scoring, and the per-team exp_woba build. Both the daily script
(update_woba_v13.py) and the historical backfill (backfill_season.py) import from
here so the model math has a SINGLE source of truth and can't drift between live
and backfilled data.

Network/fetch orchestration stays in each caller, because it differs:
  - daily  → today's games, live feeds
  - backfill → historical games, point-in-time (data strictly before the target date)

The functions here take already-fetched lookups and return computed values.
"""

import re
import pandas as pd

CORE_VERSION = "bullpen-v1 (BF-weighted starter hand + pen-hand mix; forked from core v7)"

# ─────────────────────────────────────────────────────────────────────────────
# Constants (single source of truth — callers import these)
# ─────────────────────────────────────────────────────────────────────────────
PITCH_TYPES = [
    "4-Seam Fastball","Sinker","Cutter","Slider","Sweeper","Curveball",
    "Knuckle Curve","Changeup","Splitter","Forkball","Other"
]
SLOT_MULT = {1:1.03, 2:1.05, 3:1.01, 4:1.04, 5:1.01, 6:0.98, 7:0.97, 8:0.96, 9:0.95}
REG_K = {'overall': 150.0, 'hand': 250.0, 'pitch': 200.0}

POS_GROUPS = {
    'C':['C'],'1B':['1B'],'2B':['2B'],'3B':['3B'],'SS':['SS'],
    'OF':['LF','CF','RF','OF'],
    'UTIL':['1B','2B','3B','SS','LF','CF','RF','OF','DH','UTIL','P'],
}

# ─────────────────────────────────────────────────────────────────────────────
# Scoring helpers
# ─────────────────────────────────────────────────────────────────────────────
def oaa_cap(oaa_val):
    """OAA -> wOBA-scale run contribution. No cap.

    NOTE: `is None` / isna, not a falsy test. An OAA of exactly 0 is a real value
    (a dead-average defender), not missing data.
    """
    if oaa_val is None or pd.isna(oaa_val):
        return 0.0
    return float(oaa_val) * (10/162) / 0.320

def composite_score(exp_woba, oaa_val):
    """Composite = ExpwOBA + OAA contribution (OAA scaled to wOBA units * 0.01).

    The old guard was `if not exp_woba`, which treats an exp_woba of exactly 0.0 as
    missing and returns 0.0 — silently discarding the player's ENTIRE defensive
    value. exp_woba is set to 0.0 whenever the cascade finds no data (a callup, or
    anyone absent from Statcast), so those players lost their glove entirely and
    could never win a lineup spot on defense. Test for None/NaN instead.
    """
    if exp_woba is None or pd.isna(exp_woba):
        return 0.0
    return float(exp_woba) + oaa_cap(oaa_val) * 0.01

def regress(raw_woba, pa, anchor, k):
    """Bayesian-style shrinkage: (raw*pa + anchor*k) / (pa + k)."""
    if raw_woba is None or pd.isna(raw_woba):
        return anchor
    pa = float(pa or 0)
    return (raw_woba * pa + anchor * k) / (pa + k)

# ─────────────────────────────────────────────────────────────────────────────
# THE CASCADE — regressed exp_woba for one player vs one pitcher matchup
# ─────────────────────────────────────────────────────────────────────────────
def compute_player_woba(name, sp_name, p_hand, mix,
                        woba_lookup, pa_lookup, sp_loaded,
                        proj_woba, league_avg_woba):
    """Run the 4-level regression cascade for a single player against a single
    pitcher's hand + pitch mix. Returns a dict with:
        exp_woba, reg_vs_LHP, reg_vs_RHP, reg_pitch_by_type (11 types), had_data(bool)

    projection(xwOBA) → OVERALL → SAME-HAND → PITCH-TYPE, each anchoring to the
    regressed value above it. Identical to the daily script's inline cascade.

    Args:
      name       : roster name key ("witt, bobby")
      sp_name    : resolved savant name key
      p_hand     : opposing pitcher hand 'L'/'R'
      mix        : {pitch_type: pct} for today's opposing starter
      woba_lookup: {(name, 'OVERALL'|'L'|'R'): raw_woba}
      pa_lookup  : {(name, 'OVERALL'|'L'|'R'): pa}
      sp_loaded  : DataFrame of savant splits (player_name, p_throws, pitch_type, woba, pa)
      proj_woba  : {name: xwOBA projection}
      league_avg_woba: float fallback
    """
    # Step 1 — regressed OVERALL (anchor = projection or league avg)
    # `is not None` throughout, not `or`: a wOBA of exactly 0.000 is a real value
    # (an 0-for-12 start) and must regress from 0.000, not fall through to the next
    # lookup or to league average.
    def _first(*vals):
        for v in vals:
            if v is not None and not (isinstance(v, float) and pd.isna(v)):
                return v
        return None
    overall_anchor = _first(proj_woba.get(sp_name), proj_woba.get(name), league_avg_woba)
    raw_overall = _first(woba_lookup.get((sp_name, 'OVERALL')),
                         woba_lookup.get((name, 'OVERALL')))
    overall_pa  = _first(pa_lookup.get((sp_name, 'OVERALL')),
                         pa_lookup.get((name, 'OVERALL')), 0)
    if raw_overall is not None:
        reg_overall = regress(raw_overall, overall_pa, overall_anchor, REG_K['overall'])
        has_overall = True
    else:
        reg_overall = overall_anchor
        has_overall = (overall_anchor != league_avg_woba)

    # Step 2 — regressed vs-hand for BOTH hands (anchor = reg_overall)
    def reg_hand_for(hand):
        rh = _first(woba_lookup.get((sp_name, hand)), woba_lookup.get((name, hand)))
        hp = _first(pa_lookup.get((sp_name, hand)), pa_lookup.get((name, hand)), 0)
        if rh is not None:
            return regress(rh, hp, reg_overall, REG_K['hand']), True
        return reg_overall, False
    reg_hand_L, has_hand_L = reg_hand_for('L')
    reg_hand_R, has_hand_R = reg_hand_for('R')
    reg_hand = reg_hand_L if p_hand == 'L' else reg_hand_R
    has_hand = has_hand_L if p_hand == 'L' else has_hand_R

    # Step 3 — regressed pitch-mix (anchor = reg_hand), capture every pitch type
    pitch_component = 0.0
    pitch_weight_seen = 0.0
    pitch_anchor = reg_hand
    reg_pitch_by_type = {}
    total_mix = sum(mix.values())
    rows_p = sp_loaded[(sp_loaded['player_name']==sp_name) &
                       (sp_loaded['p_throws']==p_hand)]
    for pt in PITCH_TYPES:
        prow = rows_p[rows_p['pitch_type']==pt]
        if not prow.empty:
            raw_pt = float(prow['woba'].iloc[0])
            pt_pa  = float(prow['pa'].iloc[0]) if 'pa' in prow else 0
            reg_pitch_by_type[pt] = regress(raw_pt, pt_pa, pitch_anchor, REG_K['pitch'])
        else:
            reg_pitch_by_type[pt] = pitch_anchor
    if total_mix > 0:
        for pt, pct in mix.items():
            if pct <= 0:
                continue
            pitch_component   += reg_pitch_by_type[pt] * pct
            pitch_weight_seen += pct
        if pitch_weight_seen > 0:
            pitch_component /= pitch_weight_seen
    has_pitch = (total_mix > 0)

    # Step 4 — blend 70% pitch / 30% hand (overall is the anchor, not a term)
    weights = {'pitch': 0.70, 'hand': 0.30}
    comp = {}
    if has_pitch: comp['pitch'] = pitch_component
    if has_hand:  comp['hand']  = reg_hand
    if comp:
        tw = sum(weights[k] for k in comp)
        ew = sum(comp[k] * weights[k] / tw for k in comp)
        had_data = True
    elif has_overall:
        ew = reg_overall
        had_data = True
    else:
        ew = league_avg_woba
        had_data = False

    return {
        'exp_woba': ew,
        'reg_vs_LHP': reg_hand_L,
        'reg_vs_RHP': reg_hand_R,
        'reg_pitch_by_type': reg_pitch_by_type,
        'had_data': had_data,
    }

# ─────────────────────────────────────────────────────────────────────────────
# BULLPEN-AWARE projection (Piece 1 + Piece 2)
# ─────────────────────────────────────────────────────────────────────────────
LINEUP_PA_PER_GAME = 38.0     # ~ team PAs in a 9-inning game

def bullpen_blend(res, p_hand_starter, w_start, pen_rhp_share,
                  use_pen_hand=True, tto_delta=0.0):
    """Blend a player's projection across the STARTER's innings and the RELIEF
    innings, instead of booking the full game at the starter's hand.

    res            : the dict returned by compute_player_woba (has reg_vs_LHP/RHP)
    p_hand_starter : 'L' or 'R' — the starting pitcher's hand
    w_start        : fraction of the lineup's PAs the starter is projected to face
                     (starter_batters_faced / LINEUP_PA_PER_GAME, capped at 1.0)
    pen_rhp_share  : projected fraction of RELIEF PAs thrown by RHP for this game's
                     opposing bullpen (Piece 2). Ignored if use_pen_hand=False.
    use_pen_hand   : Piece 2 toggle. If False, the relief portion uses a NEUTRAL
                     baseline (the average of the player's vs-L and vs-R), i.e. no
                     assumed pen handedness — the conservative Piece-1-only model.
    tto_delta      : times-through-order adjustment (wOBA points) added to the
                     STARTER portion only. Computed upstream from the pitcher's own
                     sample-regressed TTO curve, weighted by how deep he's projected
                     to pitch. Positive = the batter gains as the starter tires. It
                     applies only to the starter's innings; relievers reset the count.

    Returns a new exp_woba that replaces res['exp_woba'] for the composite.

    Why this is correct: compute_player_woba already produced reg_vs_LHP and
    reg_vs_RHP (fully regressed, pitch-mix-blended on the starter's hand for the
    matched one). The STARTER portion keeps the starter-hand value (now plus the TTO
    penalty for facing him multiple times); the RELIEF portion is re-weighted to the
    hand(s) the pen actually throws.
    """
    vL = res.get('reg_vs_LHP'); vR = res.get('reg_vs_RHP')
    if vL is None or vR is None:
        return res['exp_woba']                    # no split data — leave as-is
    w_start = max(0.0, min(1.0, w_start))
    w_relief = 1.0 - w_start

    starter_val = (vL if p_hand_starter == 'L' else vR) + tto_delta

    if use_pen_hand:
        # Piece 2: relief value = pen_rhp_share vs RHP + (1-share) vs LHP
        s = max(0.0, min(1.0, pen_rhp_share))
        relief_val = s * vR + (1 - s) * vL
    else:
        # Piece 1 only: neutral baseline for relief (no assumed hand)
        relief_val = 0.5 * (vL + vR)

    return w_start * starter_val + w_relief * relief_val


def starter_pa_weight(starter_bf, lineup_pa=LINEUP_PA_PER_GAME):
    """w_start from a starter's projected batters-faced. An opener with tiny BF
    collapses toward ~0 automatically; a workhorse approaches 0.7-0.75."""
    if starter_bf is None or starter_bf <= 0:
        return 0.64                                # league-avg fallback
    return max(0.0, min(1.0, float(starter_bf) / lineup_pa))

# ─────────────────────────────────────────────────────────────────────────────
# Lineup-gap (season log) — actual vs model-optimal, total-vs-total composite
# ─────────────────────────────────────────────────────────────────────────────
def lineup_gap(actual_names, optimal_names, composite_lookup, fallback=None):
    """Total value of ACTUAL lineup minus MODEL-OPTIMAL lineup, each = sum over 9
    slots of (composite × slot multiplier). Compares TOTALS (not slot pairs), so
    the optimal (value-maximizing) lineup dominates and the gap is <= 0.

    A player missing from composite_lookup used to contribute 0 to his side's total.
    That is not a neutral default — it is the value of a player who cannot bat. One
    missing player swung the gap by ~0.33, roughly 4x the real signal, and when the
    missing player was on the OPTIMAL side the gap went POSITIVE, producing the
    "actual beat optimal" games that should be impossible by construction.

    Now an unknown player is scored at `fallback` (league average when supplied), so
    a data gap is neutral rather than a fabricated penalty. Returns (gap, n_missing)
    so callers can flag games built on incomplete data.
    """
    actual_total = optimal_total = 0.0
    missing = 0
    for i in range(9):
        mult = SLOT_MULT.get(i+1, 1.0)
        if i < len(actual_names):
            a = composite_lookup.get(actual_names[i])
            if a is None:
                a = fallback; missing += 1
            if a is not None: actual_total += mult * a
        if i < len(optimal_names):
            o = composite_lookup.get(optimal_names[i])
            if o is None:
                o = fallback; missing += 1
            if o is not None: optimal_total += mult * o
    return round(actual_total - optimal_total, 4), missing

# ─────────────────────────────────────────────────────────────────────────────
# Decision pairing — benched starters ↔ model additions, globally optimal, no reuse
# ─────────────────────────────────────────────────────────────────────────────
def _pair_overlap(a_pos, b_pos):
    """Position overlap for deciding whether two players are interchangeable.

    DH is not a fielding position, it is the absence of one: ANY hitter can DH, so
    a lineup spot held by a DH-listed player is contestable by anyone. Treat DH (and
    UTIL, which is explicitly 'anywhere') as compatible with everything. Otherwise
    require a genuinely shared fielding position.
    """
    WILD = {'DH', 'UTIL'}
    if a_pos & WILD or b_pos & WILD:
        return 1
    return len(a_pos & b_pos)


def pair_decisions(actual_lineup, optimal_lineup, pos_lookup):
    """Pair each benched starter (in actual, not in optimal) with a model addition
    (in optimal, not in actual). Globally-optimal by position overlap, best-first,
    each addition used at most once. Returns list of (manager_pick, model_pick).

    A pair is only emitted when the two players could plausibly occupy the same
    lineup spot, i.e. they share a fielding position (or either is a DH/UTIL, which
    anyone can fill). A "platoon decision" means the manager could have played B
    instead of A in the same spot; a shortstop and a catcher is not a decision the
    manager ever faced, it is an artifact of the optimizer restructuring several
    slots at once.

    Without this constraint every leftover benched starter got matched to whatever
    addition remained, producing nonsense like a shortstop "replaced by" a catcher.
    That inflated decision counts for durable everyday players (always in the actual
    lineup, so always available to absorb a leftover pairing) and polluted ~40% of
    the log.
    """
    actual_set, opt_set = set(actual_lineup), set(optimal_lineup)
    benched   = [p for p in actual_lineup  if p and p not in opt_set]
    additions = [p for p in optimal_lineup if p and p not in actual_set]

    candidate_pairs = []
    for starter in benched:
        s_pos = pos_lookup.get(starter, set())
        for add in additions:
            overlap = _pair_overlap(s_pos, pos_lookup.get(add, set()))
            if overlap == 0:
                continue          # not a swap the manager could have made
            candidate_pairs.append((overlap, starter, add))
    candidate_pairs.sort(key=lambda x: (-x[0], x[1], x[2]))

    used_starters, used_additions, pairs = set(), set(), []
    for overlap, starter, add in candidate_pairs:
        if starter in used_starters or add in used_additions:
            continue
        used_starters.add(starter)
        used_additions.add(add)
        pairs.append((starter, add))
    return pairs

# ─────────────────────────────────────────────────────────────────────────────
# Position eligibility + optimal-lineup selection (pure; moved from daily script)
# ─────────────────────────────────────────────────────────────────────────────
def can_play(player_positions, required_slot):
    """Whether a player can fill a required slot given all their positions."""
    eligible = POS_GROUPS.get(required_slot, [required_slot])
    return any(pos in eligible for pos in player_positions)

def _best_legal_eight(player_scores):
    """Choose the 8 fielders that maximize total composite SUBJECT TO covering
    C, 1B, 2B, 3B, SS, OF, OF, OF.

    The old code filled each slot greedily with the best eligible player and never
    checked that the resulting GROUP could take the field. On real 2026 rosters that
    produced illegal lineups: DET would pick three catchers (two of them arriving via
    the DH step and the fallback, neither of which checked eligibility) while Hao-Yu
    Lee (2B/3B) sat on the bench, leaving 2B/SS/3B with only two capable bodies for
    three slots.

    This does it exactly instead: try players best-composite-first and backtrack, so
    the first complete assignment found is the highest-value legal one. Only 8 slots,
    so the search is small and terminates fast.

    Returns (list_of_8_players_in_slot_order, slots) or (None, None) if the roster
    genuinely cannot field a legal eight.
    """
    slots = ['C', '1B', '2B', '3B', 'SS', 'OF', 'OF', 'OF']
    n = len(player_scores)

    def search(i, used_idx):
        if i == len(slots):
            return []
        for j in range(n):
            if j in used_idx:
                continue
            if not can_play(player_scores[j]['positions'], slots[i]):
                continue
            rest = search(i + 1, used_idx | {j})
            if rest is not None:
                return [j] + rest
        return None

    order = search(0, frozenset())
    if order is None:
        return None, None
    return [player_scores[j] for j in order], slots


def select_optimal_lineup(players_df, oaa_lookup, woba_lookup, obp_lookup,
                          exp_woba_lookup, pitcher_hand, confirmed_starters=None,
                          resolve_fn=None):
    """Pick + order the model's optimal 9. Identical logic to the daily script:
    select 8 fielders + DH by composite, then order slots 2,1,4,3,5,6,7,8,9.
    Returns [(player_name, slot), ...]."""
    if players_df.empty:
        return []
    if resolve_fn is None:
        resolve_fn = lambda n: n

    player_scores = []
    for _, row in players_df.iterrows():
        name        = row['player_name']
        primary_pos = str(row.get('position', 'UTIL'))
        if primary_pos == 'P':
            continue
        secondary_str = str(row.get('secondary_positions', ''))
        secondary = [p.strip() for p in secondary_str.split(',') if p.strip()] \
                    if secondary_str and secondary_str != 'nan' else []
        all_positions = list(set([primary_pos] + secondary))

        rname        = resolve_fn(name.lower()) if name else name
        exp_w        = exp_woba_lookup.get(name)
        woba_hand    = woba_lookup.get((rname, pitcher_hand)) or woba_lookup.get((name, pitcher_hand))
        woba_overall = woba_lookup.get((rname, 'OVERALL'))   or woba_lookup.get((name, 'OVERALL'))
        oaa          = oaa_lookup.get(rname)
        if oaa is None:
            oaa = oaa_lookup.get(name, 0.0)

        if exp_w is not None:
            exp_woba_val = exp_w
        elif woba_hand is not None:
            exp_woba_val = woba_hand
        elif woba_overall is not None:
            exp_woba_val = woba_overall
        else:
            exp_woba_val = 0.0
        comp = composite_score(exp_woba_val, oaa)

        player_scores.append({
            'player_name': name, 'positions': all_positions,
            'exp_woba': exp_woba_val, 'composite': comp,
        })

    player_scores.sort(key=lambda x: x['composite'], reverse=True)

    # Choose the eight fielders. The old code filled each slot greedily with the best
    # eligible player, which USUALLY works but can strand a scarce position: a bat
    # that qualifies for an early slot gets consumed there, and a later slot (or the
    # DH step, which had no eligibility check at all) leaves the nine unable to take
    # the field. On real 2026 rosters that produced illegal lineups in ~0.6% of cases
    # (e.g. DET choosing three catchers with nobody able to play 2B).
    #
    # Fix: fill greedily as before, then VERIFY the eight can legally cover
    # C/1B/2B/3B/SS/OF/OF/OF. If they cannot, repair by trying the best available
    # replacement for each chosen fielder until the eight are legal. Scarce positions
    # therefore win over marginal bat upgrades, which is the constraint a real
    # manager actually faces.
    # Eight fielders, chosen so the group can actually take the field (see
    # _best_legal_eight). player_scores is already sorted by composite descending, so
    # the search returns the highest-value legal eight.
    filled, _slots = _best_legal_eight(player_scores)
    if filled is None:
        # Roster genuinely cannot field a legal eight (very thin / bad position data).
        # Fall back to the old greedy fill so we still return a nine rather than
        # nothing, since a short lineup breaks the lineup_gap comparison entirely.
        FIELD_SLOTS = ['C', '1B', '2B', '3B', 'SS', 'OF', 'OF', 'OF']
        filled, used = [], set()
        for req in FIELD_SLOTS:
            for p in player_scores:
                if p['player_name'] in used: continue
                if can_play(p['positions'], req):
                    filled.append(p); used.add(p['player_name']); break
        NON_FIELDING = {'DH', 'UTIL'}
        rest = [p for p in player_scores if p['player_name'] not in used]
        fielders  = [p for p in rest if set(p['positions']) - NON_FIELDING]
        bats_only = [p for p in rest if not (set(p['positions']) - NON_FIELDING)]
        for p in fielders + bats_only:
            if len(filled) >= 8: break
            filled.append(p); used.add(p['player_name'])
    else:
        used = {p['player_name'] for p in filled}

    # DH: best remaining bat. The eight above are guaranteed fieldable, so the DH is
    # a free choice — anyone can DH, no eligibility check needed.
    dh_candidates = [p for p in player_scores if p['player_name'] not in used]
    if dh_candidates:
        dh_player = max(dh_candidates, key=lambda p: p['exp_woba'])
        used.add(dh_player['player_name'])
    else:
        dh_player = None

    nine = filled + ([dh_player] if dh_player else [])
    if not nine:
        return []

    players_remaining = nine[:]
    slot_assignments  = {}
    if players_remaining:
        best = max(players_remaining, key=lambda p: p['exp_woba'])
        slot_assignments[2] = best; players_remaining.remove(best)
    if players_remaining:
        leadoff = max(players_remaining,
                      key=lambda p: obp_lookup.get(p['player_name'], 0.0) or 0.0)
        slot_assignments[1] = leadoff; players_remaining.remove(leadoff)
    for slot in [4, 3, 5, 6, 7, 8, 9]:
        if not players_remaining: break
        best = max(players_remaining, key=lambda p: p['exp_woba'])
        slot_assignments[slot] = best; players_remaining.remove(best)

    return [(slot_assignments[s]['player_name'], s)
            for s in sorted(slot_assignments.keys())]

# ─────────────────────────────────────────────────────────────────────────────
# Recency-weighted trailing-year WPA/game (model pick's expected contribution)
# ─────────────────────────────────────────────────────────────────────────────
# Using season-to-date average is unstable early in the year (1-3 game samples are
# noise). A trailing-365-day window always has a full season of games behind it, so
# the estimate is stable from opening day. Recency weighting within that window
# leans on last year early (stable) and shifts toward recent games as the current
# season accumulates (current) — best of both.
WPA_HALFLIFE_DAYS = 120   # a game's weight halves every ~120 days into the past

# WPA is one of the noisiest stats in baseball — the leverage ("clutch") component
# is largely non-repeatable, so a raw per-game average is unreliable at small
# samples and never fully trustworthy even at large ones. We shrink each player's
# weighted average toward 0 (league-average WPA/game is ~0 by construction) using
# a games-played prior:  reg = avg * n / (n + K).
#
# K=25 games is the chosen balance:
#   - a 1-game rookie with a +0.400 debut regresses to ~+0.015 (noise killed)
#   - a 150-game everyday player at +0.050 only moves to ~+0.043 (signal kept —
#     ~86% of the spread between the best and worst full-season players survives)
# Players with NO trailing-year games regress to exactly 0 rather than being
# dropped from the backtest, which removes the old bias where decisions involving
# rookies were silently skipped.
WPA_REG_K = 25.0

# Plate appearances in a typical start. Used to convert the per-PA WPA rate back
# to a per-game (per-start) scale, and to denominate effective sample size in
# start-equivalents so WPA_REG_K keeps meaning "games". ~4.2 PA/game is the
# league norm for a lineup regular; 4.0 is a clean, defensible round number.
PA_PER_START = 4.0

def wpa_recency_weighted(df, as_of_date, window_days=365):
    """{batter_id: recency-weighted avg WPA/game} over the trailing `window_days`
    ending the day BEFORE as_of_date (point-in-time). Each game is weighted by
    0.5 ** (age_in_days / WPA_HALFLIFE_DAYS), so recent games count more.

    df must have: batter, game_pk, game_date, inning_topbot, delta_home_win_exp.
    """
    import pandas as _pd
    out = {}
    if df is None or df.empty or 'delta_home_win_exp' not in df.columns:
        return out
    need = {'batter','game_pk','inning_topbot','delta_home_win_exp'}
    if not need.issubset(df.columns):
        return out

    from datetime import datetime as _dt, timedelta as _td
    asof = _dt.strptime(as_of_date, '%Y-%m-%d')
    lo   = asof - _td(days=window_days)

    w = df.copy()
    # date column: pybaseball statcast uses 'game_date'
    if 'game_date' not in w.columns:
        return out
    w = w[['batter','game_pk','game_date','inning_topbot','delta_home_win_exp']].dropna(
        subset=['batter','game_pk','game_date','delta_home_win_exp'])
    w['game_date'] = _pd.to_datetime(w['game_date'], errors='coerce')
    w = w.dropna(subset=['game_date'])
    # strictly before as_of_date, within the trailing window
    w = w[(w['game_date'] < asof) & (w['game_date'] >= lo)]
    if w.empty:
        return out

    # signed WPA (home batters +, away −), summed to one value per batter-game
    w['wpa'] = w['delta_home_win_exp'].where(w['inning_topbot']=='Bot',
                                             -w['delta_home_win_exp'])
    # PA count per batter-game: distinct at_bat_number (pitch-level data -> many
    # rows per PA). Falls back to 1 PA/game if the column is unavailable.
    if 'at_bat_number' in w.columns:
        pa = (w.groupby(['batter','game_pk','game_date'])['at_bat_number']
                .nunique().reset_index(name='pa'))
    else:
        pa = (w.groupby(['batter','game_pk','game_date'])
                .size().reset_index(name='pa'))
        pa['pa'] = 1
    per_game = w.groupby(['batter','game_pk','game_date'])['wpa'].sum().reset_index()
    per_game = per_game.merge(pa, on=['batter','game_pk','game_date'], how='left')
    per_game['pa'] = per_game['pa'].fillna(1).clip(lower=1)

    # The model recommends a player START, but a raw per-game average blends his
    # starts with one-PA pinch-hit cameos — games where he had little chance to
    # move win probability in either direction. That dilutes bench players'
    # estimates toward zero, and bench players are exactly who the model picks,
    # so the bias runs one way (it understates the model's own recommendations).
    #
    # Fix: build a per-PA rate and weight each game by its PA count (the evidence
    # it actually carries), then scale back to a start. A 4-PA start counts 4x a
    # 1-PA cameo. NOTE: we weight the RATE, not the game's WPA sum — a 4-PA game
    # already has ~4x the raw WPA of a cameo, so weighting the sum by PA would
    # double-count playing time.
    #
    # Weights are recency * PA, normalized by PA_PER_START, so:
    #   - the output stays on a per-GAME (per-start) scale, directly comparable to
    #     the manager's actual per-game WPA, and
    #   - n_eff stays denominated in start-equivalents, so WPA_REG_K keeps its
    #     meaning (a 1-PA cameo is ~1/4 of a game of evidence, not a full one).
    age_days = (asof - per_game['game_date']).dt.days.clip(lower=0)
    recency = 0.5 ** (age_days / float(WPA_HALFLIFE_DAYS))
    per_game['rate'] = per_game['wpa'] / per_game['pa']          # WPA per PA
    per_game['wt']   = recency * per_game['pa']                  # evidence weight
    per_game['wwpa'] = per_game['rate'] * per_game['wt']
    agg = per_game.groupby('batter').agg(wwpa=('wwpa','sum'), wt=('wt','sum'))
    for bid, r in agg.iterrows():
        if r['wt'] > 0:
            # r['wt'] is in recency-weighted PA units; convert to start-equivalents
            # so n_eff means "how many starts' worth of evidence do we have" and
            # WPA_REG_K stays interpretable as a number of games.
            n_eff = float(r['wt']) / float(PA_PER_START)
            rate_per_pa = float(r['wwpa']) / float(r['wt'])
            # Scale the per-PA rate up to a full start.
            per_start = rate_per_pa * float(PA_PER_START)
            # Shrink toward 0 by effective starts. Equivalent to (avg * n)/(n + K):
            # thin samples collapse to ~0, full seasons barely move.
            out[int(bid)] = per_start * n_eff / (n_eff + WPA_REG_K)
    return out
