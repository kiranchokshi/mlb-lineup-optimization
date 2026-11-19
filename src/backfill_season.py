#!/usr/bin/env python3
"""
backfill_season.py — Point-in-time historical backfill for the MLB platoon model.

Reconstructs season_log.csv and decisions_log.csv for EVERY day from Opening Day
2026 through yesterday, using ONLY data available at the time (point-in-time).

Shares all model math with the daily script via platoon_core.py (single source of
truth). This script owns the FETCH orchestration, with historical date bounds:
  - Statcast for a target date is pulled with end_dt = target_date - 1 (no lookahead).
  - Rosters are fetched AS OF the target date (?date=), so team membership is correct.
  - Lineups come from that date's FINAL boxscore (battingOrder X00 = starters).
  - WPA comes from that date's completed games (delta_home_win_exp), so decisions are
    captured AND resolved in one pass (games are final - no pending queue).

HONEST LIMITATIONS (read these)
  - Splits/xwOBA/pitch-mix are all derived from the SAME point-in-time Statcast pull,
    so they're internally consistent and lookahead-free. This uses Statcast raw wOBA
    (woba_value/woba_denom) rather than Savant's separate xwOBA leaderboard - a slight
    difference from the daily script's xwOBA anchor, but it avoids a second fetch per
    day and keeps each day self-consistent.
  - Model-expected WPA for a decision uses the model pick's WPA IN THAT GAME (the only
    per-game WPA cheaply available in a one-pass historical run), not a trailing season
    average. Decisions where the model pick didn't record WPA that game are skipped
    (your chosen "clean comparison" behavior).
  - OAA / catcher framing: fetched ONCE (current 2025+2026 numbers) and reused for
    every date. Defensive skill is stable across a season, so the current number is a
    sound point-in-time approximation. This matches the daily script's OAA source.
  - Roster-as-of-date is reliable for the active roster; exact transaction-day timing
    can occasionally include/omit a bench player (~95%+ correct).
  - This is a faithful RECONSTRUCTION, not a bit-identical replay of live runs.

USAGE
  python3 backfill_season.py                       # Opening Day -> yesterday
  python3 backfill_season.py 2026-05-15            # from a date -> yesterday
  python3 backfill_season.py 2026-05-15 2026-05-15 # a SINGLE day (RECOMMENDED FIRST TEST)

platoon_core.py must be in the SAME folder as this script.
"""

import os, sys, time, requests, unicodedata, re
from io import StringIO
from datetime import datetime, timedelta
import pandas as pd

import platoon_core as pc
from platoon_core import (
    PITCH_TYPES, SLOT_MULT, REG_K,
    composite_score, regress, compute_player_woba,
    select_optimal_lineup, lineup_gap, pair_decisions,
    wpa_recency_weighted,
)

# ─────────────────────────────────────────────────────────────────────────────
DESKTOP        = os.path.expanduser("~/Desktop")
SEASON_LOG_OUT = os.path.join(DESKTOP, "season_log_bullpen.csv")
DECISIONS_OUT  = os.path.join(DESKTOP, "decisions_log_bullpen.csv")

SCRIPT_VERSION = "bullpen-v3clean (Piece1 projected starter-BF + Piece2 prior-season pen decay; table-builds WIRED; no TTO; point-in-time)"


OPENING_DAY = "2026-03-27"       # adjust if 2026 Opening Day differs
STATCAST_WINDOW_DAYS = 1095      # 3-year trailing window (matches daily script)

PITCH_NAME_MAP = {
    '4-Seam Fastball':'4-Seam Fastball','Fastball':'4-Seam Fastball',
    'Sinker':'Sinker','Two-Seam Fastball':'Sinker',
    'Cutter':'Cutter','Slider':'Slider','Sweeper':'Sweeper',
    'Curveball':'Curveball','Knuckle Curve':'Knuckle Curve','Slow Curve':'Curveball',
    'Changeup':'Changeup','Screwball':'Changeup',
    'Split-Finger':'Splitter','Splitter':'Splitter','Forkball':'Forkball',
    'Eephus':'Other','Slurve':'Other','Knuckleball':'Other',
    'Pitch Out':'Other','Unknown':'Other',
}
TEAM_ID_MAP = {
    109:"ARI",110:"BAL",111:"BOS",112:"CHC",113:"CIN",114:"CLE",
    115:"COL",116:"DET",117:"HOU",118:"KC",119:"LAD",120:"WSH",
    121:"NYM",133:"OAK",134:"PIT",135:"SD",136:"SEA",137:"SF",
    138:"STL",139:"TB",140:"TEX",141:"TOR",142:"MIN",143:"PHI",
    144:"ATL",145:"CWS",146:"MIA",147:"NYY",158:"MIL",108:"LAA"
}

def normalize_name(name):
    if not name or not isinstance(name, str):
        return name
    nfkd = unicodedata.normalize('NFKD', name)
    a = ''.join(c for c in nfkd if not unicodedata.combining(c)).strip()
    a = re.sub(r',?\s*\b(Jr\.?|Sr\.?|III|II|IV|V)\b\.?', '', a, flags=re.IGNORECASE)
    a = re.sub(r'\.{2,}', '.', a)
    a = re.sub(r'\s+', ' ', a).strip().rstrip(',').strip()
    return a.lower()

def to_last_first(name):
    name = name.strip()
    suf = re.compile(r'^(Jr\.?|Sr\.?|II|III|IV|V)$', re.IGNORECASE)
    parts = name.split()
    if len(parts) < 2:
        return name
    sfx = []
    while len(parts) > 2 and suf.match(parts[-1]):
        sfx.insert(0, parts.pop())
    last, first = parts[-1], ' '.join(parts[:-1])
    return f"{last}, {first}" + (f" {' '.join(sfx)}" if sfx else "")

def daterange(start, end):
    d = datetime.strptime(start, '%Y-%m-%d'); e = datetime.strptime(end, '%Y-%m-%d')
    while d <= e:
        yield d.strftime('%Y-%m-%d'); d += timedelta(days=1)

def normalize_savant_name(row, cols):
    """Extract + normalize a player name from a Savant leaderboard CSV row."""
    if 'last_name' in cols and 'first_name' in cols:
        return normalize_name(to_last_first(f"{row['first_name']} {row['last_name']}"))
    combined = next((c for c in cols if c.strip().lower() == 'last_name, first_name'), None)
    if combined:
        return normalize_name(str(row[combined]).strip())
    for c in cols:
        if 'name' in c.lower() and 'team' not in c.lower() and 'pos' not in c.lower():
            val = str(row[c]).strip()
            return normalize_name(val if ',' in val else to_last_first(val))
    return None

# ── OAA (fetched ONCE for the whole backfill; current 2025+2026) ─────────────
# OAA/framing is stable over a season, so using the CURRENT number as the point-in-
# time value for every date is a sound approximation (per design decision). Fetched
# a single time and reused across all dates — far faster than per-date, and correct.
def fetch_oaa_once(year=2026):
    """Return {normalized_name: oaa} combining fielding OAA + catcher framing for
    2025+year. Mirrors the daily script's OAA build. Empty dict on failure."""
    def fetch_fielding(yr):
        for url in [
            (f"https://baseballsavant.mlb.com/leaderboard/outs_above_average"
             f"?year={yr}&team=&min=1&pos=&roles=&viz=Show+Stats&csv=true"),
            (f"https://baseballsavant.mlb.com/leaderboard/outs_above_average"
             f"?type=Fielding&year={yr}&team=&min=1&pos=&roles=&viz=Show+Stats&csv=true"),
        ]:
            try:
                resp = requests.get(url, timeout=30, headers={"User-Agent":"Mozilla/5.0"})
                if resp.status_code != 200 or len(resp.text) < 200 or resp.text.strip().startswith('<'):
                    continue
                df_o = pd.read_csv(StringIO(resp.text)); cols = df_o.columns.tolist()
                oaa_col = next((c for c in cols if c.lower()=='outs_above_average'), None)
                if not oaa_col:
                    continue
                df_o['player_name'] = df_o.apply(lambda r: normalize_savant_name(r, cols), axis=1)
                df_o['oaa'] = pd.to_numeric(df_o[oaa_col].astype(str).str.strip(), errors='coerce')
                df_o = df_o[['player_name','oaa']].dropna()
                if len(df_o) > 10:
                    return df_o
            except Exception:
                time.sleep(0.3)
        return None

    def fetch_catcher(yr):
        try:
            resp = requests.get(
                f"https://baseballsavant.mlb.com/leaderboard/catcher-framing?year={yr}&csv=true",
                timeout=30, headers={"User-Agent":"Mozilla/5.0"})
            if resp.status_code != 200 or len(resp.text) < 500 or resp.text.strip().startswith('<'):
                return None
            df_c = pd.read_csv(StringIO(resp.text)); cols = df_c.columns.tolist()
            if 'name' not in cols or 'rv_tot' not in cols:
                return None
            df_c['player_name'] = df_c['name'].apply(
                lambda n: normalize_name(str(n) if ',' in str(n) else to_last_first(str(n))))
            df_c['framing_runs'] = pd.to_numeric(df_c['rv_tot'].astype(str).str.strip(), errors='coerce')
            df_c = df_c.dropna(subset=['player_name','framing_runs'])
            df_c['oaa'] = df_c['framing_runs'] / 0.75
            return df_c[['player_name','oaa']].copy()
        except Exception:
            return None

    years = [2025] if year == 2025 else [2025, year]
    field = [r for yr in years for r in [fetch_fielding(yr)] if r is not None]
    catch = [r for yr in years for r in [fetch_catcher(yr)]  if r is not None]
    fielding_df = pd.concat(field).groupby('player_name', as_index=False)['oaa'].sum() if field else pd.DataFrame(columns=['player_name','oaa'])
    catcher_df  = pd.concat(catch).groupby('player_name', as_index=False)['oaa'].sum() if catch else pd.DataFrame(columns=['player_name','oaa'])
    catcher_names = set(catcher_df['player_name'])
    non_catchers = fielding_df[~fielding_df['player_name'].isin(catcher_names)]
    combined = pd.concat([non_catchers, catcher_df], ignore_index=True)
    combined['oaa'] = pd.to_numeric(combined['oaa'], errors='coerce').fillna(0)
    return dict(zip(combined['player_name'], combined['oaa']))

# ── POINT-IN-TIME FETCHES ────────────────────────────────────────────────────
def load_statcast_range(overall_start, overall_end, max_retries=4):
    """Pull ALL Statcast pitch data from overall_start..overall_end ONCE, in
    monthly chunks (so no single query is enormous and memory stays reasonable),
    and return one concatenated DataFrame. Slicing per date is then done in memory
    — this is the big speedup vs re-pulling a 3-year window for every date.

    overall_start should be (earliest target date − 3yr window); overall_end is
    yesterday. We only download this once for the whole backfill.

    Chunk failures are RETRIED and, if still failing, RAISED. Baseball Savant
    regularly truncates a large response ("Connection broken: IncompleteRead"),
    which is transient. The old code caught the exception, printed it, and carried
    on — so a dropped month left a hole in the frame, and every downstream date
    silently backfilled against missing data (an April gap also removes April from
    every later date's trailing window). A partial frame is worse than no run at
    all, because the output looks complete.
    """
    import pybaseball; pybaseball.cache.enable()
    frames = []
    cur = datetime.strptime(overall_start, '%Y-%m-%d')
    end = datetime.strptime(overall_end, '%Y-%m-%d')
    failed = []
    while cur <= end:
        # chunk = this calendar month (cur .. min(month-end, end))
        if cur.month == 12:
            nxt = cur.replace(year=cur.year+1, month=1, day=1)
        else:
            nxt = cur.replace(month=cur.month+1, day=1)
        chunk_end = min(nxt - timedelta(days=1), end)
        s = cur.strftime('%Y-%m-%d'); e = chunk_end.strftime('%Y-%m-%d')
        print(f"  loading Statcast {s} .. {e} ...")
        got = None
        for attempt in range(1, max_retries + 1):
            try:
                df = pybaseball.statcast(start_dt=s, end_dt=e)
                got = df if df is not None else pd.DataFrame()
                break
            except Exception as ex:
                wait = 5 * attempt
                if attempt < max_retries:
                    print(f"    chunk {s}..{e} attempt {attempt}/{max_retries} failed "
                          f"({type(ex).__name__}); retrying in {wait}s")
                    time.sleep(wait)
                else:
                    print(f"    chunk {s}..{e} FAILED after {max_retries} attempts: {ex}")
                    failed.append((s, e))
        if got is not None and not got.empty:
            frames.append(got)
        cur = nxt
    if failed:
        raise RuntimeError(
            "Statcast chunks failed to download: "
            + ", ".join(f"{a}..{b}" for a, b in failed)
            + ". Refusing to backfill on an incomplete frame — every date's trailing "
              "window would silently be missing this data. Re-run; pybaseball caches "
              "the chunks that already succeeded, so it will resume quickly.")
    if not frames:
        return pd.DataFrame()
    allc = pd.concat(frames, ignore_index=True)
    # Ensure game_date is datetime for fast slicing
    if 'game_date' in allc.columns:
        allc['game_date'] = pd.to_datetime(allc['game_date'], errors='coerce')
    return allc

def slice_before(all_df, target_date, window_days=STATCAST_WINDOW_DAYS):
    """In-memory slice: rows strictly before target_date, within the trailing
    window. Replaces a live 3-year query per date."""
    if all_df is None or all_df.empty or 'game_date' not in all_df.columns:
        return all_df if all_df is not None else pd.DataFrame()
    asof = datetime.strptime(target_date, '%Y-%m-%d')
    lo   = asof - timedelta(days=window_days)
    m = (all_df['game_date'] < asof) & (all_df['game_date'] >= lo)
    return all_df.loc[m]

def slice_on(all_df, target_date):
    """In-memory slice: rows on exactly target_date (for that day's WPA)."""
    if all_df is None or all_df.empty or 'game_date' not in all_df.columns:
        return all_df if all_df is not None else pd.DataFrame()
    asof = datetime.strptime(target_date, '%Y-%m-%d')
    return all_df.loc[all_df['game_date'] == asof]

def pull_roster_asof(target_date):
    roster_lk = {}
    for team_id, team_abbr in TEAM_ID_MAP.items():
        try:
            r = requests.get(
                f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster"
                f"?rosterType=active&date={target_date}", timeout=15).json()
            for p in r.get('roster', []):
                fn  = p.get('person',{}).get('fullName','')
                pos = p.get('position',{}).get('abbreviation','UTIL')
                if fn:
                    nm = normalize_name(to_last_first(fn))
                    roster_lk[(nm, team_abbr)] = {
                        'player_name': nm, 'team': team_abbr,
                        'position': pos, 'secondary_positions': '',
                    }
            time.sleep(0.12)
        except Exception as e:
            print(f"      roster {team_abbr} failed: {e}")
    return roster_lk

def pull_schedule(target_date):
    try:
        s = requests.get(
            f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={target_date}",
            timeout=15).json()
        return [g for d in s.get('dates', []) for g in d.get('games', [])]
    except Exception as e:
        print(f"      schedule failed: {e}")
        return []

def get_final_feed(game_pk):
    try:
        live = requests.get(
            f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live",
            timeout=20).json()
    except Exception:
        return None
    status = live.get('gameData',{}).get('status',{}).get('detailedState','')
    return live if status in ('Final','Game Over','Completed Early') else None

def starters_from_feed(live, side):
    """Starting nine (frozen) for a side. For each of the 9 batting slots, take the
    player with the LOWEST battingOrder value in that slot (X00 = the starter; X01,
    X02 = substitutes who came later). This is robust to same-inning substitutions
    that can leave a slot without a clean X00 entry. Returns (names, {name: id})."""
    players = live.get('liveData',{}).get('boxscore',{}).get('teams',{}).get(side,{}).get('players',{})
    by_slot = {}   # slot (1-9) -> (battingOrder_int, name, mlbam_id)
    for pid, pdata in players.items():
        bo = str(pdata.get('battingOrder','') or '')
        if not bo.isdigit():
            continue
        boi = int(bo)
        if boi <= 0:
            continue
        slot = boi // 100          # 100->1, 205->2, etc.
        if slot < 1 or slot > 9:
            continue
        fn = pdata.get('person',{}).get('fullName','')
        mlbam = pdata.get('person',{}).get('id')
        if not fn:
            continue
        # keep the lowest battingOrder seen for this slot (the starter)
        if slot not in by_slot or boi < by_slot[slot][0]:
            by_slot[slot] = (boi, normalize_name(to_last_first(fn)), mlbam)
    ordered = [by_slot[s] for s in sorted(by_slot)]
    names = [n for _,n,_ in ordered]
    ids   = {n: i for _,n,i in ordered}
    return names, ids

def positions_from_feed(live, side):
    """{normalized_name: 'POS'} from the boxscore. Used to place a player who
    actually started but is missing from the active-roster snapshot for that date
    (transaction timing), so the optimizer can still consider him. Prefers the
    primary listed position; falls back to any position he appeared at."""
    players = live.get('liveData',{}).get('boxscore',{}).get('teams',{}).get(side,{}).get('players',{})
    out = {}
    for pid, pdata in players.items():
        fn = pdata.get('person',{}).get('fullName','')
        if not fn:
            continue
        abbr = (pdata.get('position',{}) or {}).get('abbreviation','')
        if not abbr:
            allp = pdata.get('allPositions') or []
            if allp:
                abbr = (allp[0] or {}).get('abbreviation','')
        if abbr:
            out[normalize_name(to_last_first(fn))] = abbr
    return out

def roster_ids_from_feed(live, side):
    """Every player on a side in the boxscore -> {normalized_name: mlbam_id}.
    Covers bench players too, so model picks (bench) can be resolved to IDs."""
    players = live.get('liveData',{}).get('boxscore',{}).get('teams',{}).get(side,{}).get('players',{})
    out = {}
    for pid, pdata in players.items():
        fn = pdata.get('person',{}).get('fullName','')
        mlbam = pdata.get('person',{}).get('id')
        if fn and mlbam:
            out[normalize_name(to_last_first(fn))] = mlbam
    return out

_PITCH_HAND_CACHE = {}

# ─── TRAILING PROJECTIONS (point-in-time) for the bullpen model ──────────────
# Instead of using a game's REALIZED starter batters-faced and REALIZED pen hand
# mix (which is mildly circular — using the outcome to weight the projection), these
# build EXPECTED values from each pitcher's / team's recent history, strictly before
# the game date. This is the point-in-time version a manager could actually form at
# card-time.
_PITCHER_GAME_BF = None   # cached per-run: DataFrame [pitcher, game_pk, game_date, bf, is_starter, p_throws, team_side]

def _build_pitcher_game_table(df):
    """One row per (pitcher, game): batters faced, whether he started, his hand.
    Batters faced = distinct plate appearances = count of unique (game_pk, at_bat_number).
    Starter = the pitcher whose first pitch in the game came earliest (min pitch index).
    """
    global _PITCHER_GAME_BF
    need = {'pitcher','game_pk','game_date','p_throws','at_bat_number','inning'}
    if df is None or df.empty or not need.issubset(df.columns):
        _PITCHER_GAME_BF = pd.DataFrame(
            columns=['pitcher','game_pk','game_date','bf','is_starter','p_throws'])
        return _PITCHER_GAME_BF
    d = df[['pitcher','game_pk','game_date','p_throws','at_bat_number','inning',
            'inning_topbot']].dropna(subset=['pitcher','game_pk','at_bat_number']).copy()
    d['game_date'] = pd.to_datetime(d['game_date'], errors='coerce')
    # batters faced per pitcher-game = distinct at-bats
    bf = (d.groupby(['pitcher','game_pk'])['at_bat_number']
            .nunique().reset_index().rename(columns={'at_bat_number':'bf'}))
    # starter = pitcher who threw in the 1st inning for his side AND has the most
    # early workload; simplest robust proxy: pitched in inning 1.
    first_inning = (d[d['inning']==1].groupby(['pitcher','game_pk']).size()
                    .reset_index().rename(columns={0:'p1'}))
    meta = (d.groupby(['pitcher','game_pk'])
              .agg(game_date=('game_date','first'), p_throws=('p_throws','first'))
              .reset_index())
    t = bf.merge(meta, on=['pitcher','game_pk'], how='left')
    t = t.merge(first_inning, on=['pitcher','game_pk'], how='left')
    t['is_starter'] = t['p1'].notna()      # threw in inning 1 -> started
    t = t.drop(columns=['p1'])
    _PITCHER_GAME_BF = t
    return t

def projected_starter_bf(pitcher_id, as_of_date, n_starts=8, half_life=4):
    """Recency-weighted trailing average batters-faced over a starter's recent
    STARTS before as_of_date. Weights decay by half-life (in starts). Falls back to
    the league-average starter load if we have no history yet."""
    LEAGUE_AVG_STARTER_BF = 22.0
    if _PITCHER_GAME_BF is None or pitcher_id is None:
        return LEAGUE_AVG_STARTER_BF
    asof = pd.to_datetime(as_of_date)
    h = _PITCHER_GAME_BF
    rows = h[(h['pitcher']==int(pitcher_id)) & (h['is_starter']) &
             (h['game_date'] < asof)].sort_values('game_date', ascending=False).head(n_starts)
    if rows.empty:
        return LEAGUE_AVG_STARTER_BF
    import math
    ws = [0.5 ** (i/half_life) for i in range(len(rows))]
    vals = rows['bf'].tolist()
    return sum(w*v for w,v in zip(ws,vals)) / sum(ws)

def projected_pen_rhp_share(team_pitcher_ids, as_of_date,
                            recent_days=45, prior_lo_days=45, prior_hi_days=900,
                            K=60, half_life=12):
    """Expected fraction of RELIEF batters-faced thrown by RHP for a team's bullpen,
    projected point-in-time with prior-season -> current-season SHRINKAGE.

    The problem with a flat league-average fallback early in the year is that you DO
    know the pen — you know which relievers are on the roster and each one's hand.
    You just don't have many CURRENT-season relief appearances yet. So instead of
    falling back to league average, we fall back to how these SAME relievers were
    deployed previously (handedness doesn't change year to year), then let the
    current season take over as its sample grows:

        recent : RHP share of these relievers' relief BF in the last `recent_days`
                 (this season) — the same signal as before.
        prior  : RHP share of the SAME relievers' relief BF in everything from
                 `prior_lo_days` back to `prior_hi_days` back. This window starts
                 just past the recent window and reaches ~2.5 years, so it captures
                 all of LAST season no matter what month it currently is (a rolling
                 "365 days back" misses last summer when you're in April).
        w      : n_recent / (n_recent + K)  — current data earns weight as it grows.

        share = w * recent + (1 - w) * prior

    Early April n_recent ~ 0, so the estimate is essentially these relievers' prior
    deployment (real and team-specific, not league-average). By midseason w -> 1 and
    it's driven by current usage. League average (0.68) is used ONLY when a reliever
    has no relief history in EITHER window (a genuine rookie pen).
    """
    LEAGUE_AVG_PEN_RHP = 0.68
    if _PITCHER_GAME_BF is None or not team_pitcher_ids:
        return LEAGUE_AVG_PEN_RHP
    asof = pd.to_datetime(as_of_date)
    ids = [int(x) for x in team_pitcher_ids]
    h = _PITCHER_GAME_BF
    pen = h[(~h['is_starter']) & (h['pitcher'].isin(ids))]

    # current-season window (recency-weighted within it)
    recent_win = pen[(pen['game_date'] < asof) &
                     (pen['game_date'] >= asof - pd.Timedelta(days=recent_days))].copy()
    if not recent_win.empty:
        recent_win = recent_win.sort_values('game_date', ascending=False).reset_index(drop=True)
        recent_win['w'] = [0.5 ** (i / half_life) for i in range(len(recent_win))]
        rmask = recent_win['p_throws'] == 'R'
        rw = (recent_win.loc[rmask,'bf'] * recent_win.loc[rmask,'w']).sum()
        lw = (recent_win.loc[~rmask,'bf'] * recent_win.loc[~rmask,'w']).sum()
        recent = (rw / (rw + lw)) if (rw + lw) > 0 else None
    else:
        recent = None
    n_recent = recent_win['bf'].sum() if not recent_win.empty else 0

    # prior window: from just past the recent window back ~2.5 years. Captures all of
    # last season regardless of the current month.
    prior_win = pen[(pen['game_date'] < asof - pd.Timedelta(days=prior_lo_days)) &
                    (pen['game_date'] >= asof - pd.Timedelta(days=prior_hi_days))]
    if not prior_win.empty:
        pr = prior_win[prior_win['p_throws']=='R']['bf'].sum()
        pl = prior_win[prior_win['p_throws']=='L']['bf'].sum()
        prior = (pr / (pr + pl)) if (pr + pl) > 0 else None
    else:
        prior = None

    if recent is None and prior is None:
        return LEAGUE_AVG_PEN_RHP
    if prior is None:
        return recent
    if recent is None:
        return prior
    w = n_recent / (n_recent + K)
    return w * recent + (1 - w) * prior


def seed_pitcher_hands_from_statcast(df):
    """Fill the pitcher-hand cache from Statcast, which carries `p_throws` on every
    pitch alongside the `pitcher` id. This is the authoritative source we already
    have in memory, so it costs no API calls and covers anyone who has thrown a
    pitch in the loaded window.
    """
    if df is None or df.empty:
        return
    if 'pitcher' not in df.columns or 'p_throws' not in df.columns:
        return
    d = df[['pitcher','p_throws']].dropna()
    d = d[d['p_throws'].isin(['L','R'])]
    if d.empty:
        return
    # a pitcher's hand is constant; take the most common value per id defensively
    for pid, grp in d.groupby('pitcher')['p_throws']:
        try:
            _PITCH_HAND_CACHE[int(pid)] = grp.mode().iloc[0]
        except Exception:
            continue


def pitcher_hand(pid):
    """'L' or 'R' for a pitcher's throwing hand.

    Prefers the Statcast-seeded cache; falls back to the MLB people endpoint for a
    starter with no pitches in the loaded window (rare, e.g. a debut).

    The game feed's boxscore carries only a THIN person object (id, fullName, link);
    `pitchHand` is NOT on it. The old code read
        pdata['person']['pitchHand']['code']
    which therefore missed every time and fell through to the 'R' default, so every
    row in season_log came back right-handed. That silently disabled the platoon
    premise: batters were always projected against a righty, so vs-LHP splits never
    influenced a single lineup.
    """
    if pid is None:
        return 'R'
    pid = int(pid)
    if pid in _PITCH_HAND_CACHE:
        return _PITCH_HAND_CACHE[pid]
    hand = 'R'
    try:
        j = requests.get(f"https://statsapi.mlb.com/api/v1/people?personIds={pid}",
                         timeout=20).json()
        people = j.get('people', [])
        if people:
            code = (people[0].get('pitchHand') or {}).get('code')
            if code in ('L', 'R'):
                hand = code
    except Exception as e:
        print(f"      ⚠ pitchHand lookup failed for {pid}: {e}")
    _PITCH_HAND_CACHE[pid] = hand
    return hand


def opp_starter_from_feed(live, side, as_of_date=None):
    """Returns (starter_id, starter_hand, starter_name, starter_bf, pen_rhp_share).

    Two NEW fields vs the base model:
      starter_bf     : batters the starter actually faced this game. We use the
                       game's OWN starter BF as the point-in-time estimate of the
                       starter's workload (a manager knows a pitcher's typical
                       length pregame; the realized BF is our proxy for that
                       expectation in a historical backfill).
      pen_rhp_share  : of the RELIEF batters faced (every opposing pitcher after the
                       starter), the fraction thrown by RHP. This is the game's
                       actual bullpen hand mix — the Piece-2 input.
    """
    opp = 'away' if side == 'home' else 'home'
    box = live.get('liveData',{}).get('boxscore',{}).get('teams',{}).get(opp,{})
    pitchers = box.get('pitchers', [])
    if not pitchers:
        return None, 'R', '', None, 0.68
    sid = pitchers[0]
    pdata = box.get('players',{}).get(f'ID{sid}',{})
    name = pdata.get('person',{}).get('fullName','')
    hand = pitcher_hand(sid)

    def _bf(pid):
        pp = box.get('players',{}).get(f'ID{pid}',{})
        st = pp.get('stats',{}).get('pitching',{})
        return st.get('battersFaced', st.get('battersFaced', 0)) or 0

    # PROJECTED (point-in-time), not realized. w_start from the starter's recent
    # trailing BF; pen mix from the opposing bullpen's recent relief usage. The
    # relievers who appear AFTER the starter in THIS game identify the pen, but we
    # score their hand share from their OWN trailing relief history, not today's line.
    proj_bf = projected_starter_bf(sid, as_of_date) if as_of_date is not None else (_bf(sid) or 22.0)
    pen_ids = [pid for pid in pitchers[1:]]
    pen_share = (projected_pen_rhp_share(pen_ids, as_of_date)
                 if as_of_date is not None else 0.68)
    return sid, hand, name, proj_bf, pen_share

# ── LOOKUP BUILDERS FROM A STATCAST FRAME ────────────────────────────────────
_BATTER_NAME_CACHE = {}

def build_batter_name_map(df):
    """{batter_id: 'last, first'} for every batter in the Statcast frame.

    Raw statcast() rows describe a PITCH: `player_name` is the PITCHER, `batter` and
    `pitcher` are mlbam ids. Batter names are not in the frame at all, so resolve ids
    to names via the MLB people endpoint. Results are cached process-wide, so a full
    season backfill only pays for ids it has not already seen.
    """
    if df.empty or 'batter' not in df.columns:
        return dict(_BATTER_NAME_CACHE)
    ids = {int(x) for x in df['batter'].dropna().unique()}
    missing = sorted(ids - set(_BATTER_NAME_CACHE))
    CHUNK = 400
    for i in range(0, len(missing), CHUNK):
        chunk = missing[i:i+CHUNK]
        url = ("https://statsapi.mlb.com/api/v1/people?personIds="
               + ",".join(str(x) for x in chunk))
        try:
            j = requests.get(url, timeout=30).json()
            for p in j.get('people', []):
                pid, full = p.get('id'), p.get('fullName', '')
                if pid and full:
                    _BATTER_NAME_CACHE[int(pid)] = normalize_name(to_last_first(full))
        except Exception as e:
            print(f"      ⚠ batter name lookup failed for chunk {i//CHUNK}: {e}")
    return dict(_BATTER_NAME_CACHE)


def build_woba_lookups(df, batter_names):
    empty_sp = pd.DataFrame(columns=['player_name','p_throws','pitch_type','woba','pa'])
    if df.empty or 'woba_value' not in df.columns or 'batter' not in df.columns:
        return {}, {}, empty_sp, {}, {}
    d = df.copy().dropna(subset=['batter'])
    d['pitch_name'] = d['pitch_name'].map(lambda x: PITCH_NAME_MAP.get(x, 'Other'))

    # Key every batter lookup by the BATTER'S name, resolved from his mlbam id.
    # `player_name` in raw statcast() output is the PITCHER, so keying on it built
    # batter wOBA profiles under pitcher names (and made every pitcher look like a
    # "name collision", since one pitcher faces many batter ids).
    d['name_norm'] = d['batter'].map(
        lambda b: batter_names.get(int(b)) if pd.notna(b) else None)
    d = d.dropna(subset=['name_norm'])

    # True same-name batters (2026: two Max Muncys, LAD and OAK) would still merge,
    # because the roster CSV has only a name to look up. Suffix the less-used one so
    # each gets his own profile; resolve_batter_key() maps a roster player to the
    # right key using his mlbam id from the game feed. This fires for ~1 player a
    # season, and only for genuine duplicates — a pitcher can never trigger it now.
    ids_per_name = d.groupby('name_norm')['batter'].nunique()
    dupes = set(ids_per_name[ids_per_name > 1].index)
    collisions = {}
    if dupes:
        pa_rank = (d[d['name_norm'].isin(dupes)]
                   .groupby(['name_norm','batter'])['woba_denom'].sum()
                   .reset_index().sort_values('woba_denom', ascending=False))
        for nm, grp in pa_rank.groupby('name_norm'):
            for i, (_, row) in enumerate(grp.iterrows()):
                bid = int(row['batter'])
                collisions.setdefault(nm, {})[bid] = nm if i == 0 else f"{nm}#{bid}"
            print(f"      name shared by {len(collisions[nm])} batters: {nm}")
        d['name_norm'] = d.apply(
            lambda r: collisions.get(r['name_norm'], {}).get(int(r['batter']), r['name_norm']),
            axis=1)

    if 'game_year' in d.columns:
        yr = pd.to_numeric(d['game_year'], errors='coerce').fillna(0).astype(int)
        maxyr = yr.max() if len(yr) else 0
        d['w'] = (maxyr - yr).map(lambda g: {0:5,1:4,2:3}.get(g,2))
    else:
        d['w'] = 1
    d['wv'] = pd.to_numeric(d['woba_value'], errors='coerce') * d['w']
    d['wd'] = pd.to_numeric(d['woba_denom'], errors='coerce') * d['w']
    d = d.dropna(subset=['wd'])

    woba_lookup, pa_lookup = {}, {}
    g = d.groupby('name_norm').agg(wv=('wv','sum'), wd=('wd','sum'),
                                   pa=('woba_denom','sum')).reset_index()
    for _,r in g.iterrows():
        if r['wd'] > 0:
            woba_lookup[(r['name_norm'],'OVERALL')] = r['wv']/r['wd']
            pa_lookup[(r['name_norm'],'OVERALL')]   = r['pa']
    gh = d.groupby(['name_norm','p_throws']).agg(wv=('wv','sum'), wd=('wd','sum'),
                                                 pa=('woba_denom','sum')).reset_index()
    for _,r in gh.iterrows():
        if r['wd'] > 0 and r['p_throws'] in ('L','R'):
            woba_lookup[(r['name_norm'], r['p_throws'])] = r['wv']/r['wd']
            pa_lookup[(r['name_norm'], r['p_throws'])]   = r['pa']
    gp = d.groupby(['name_norm','p_throws','pitch_name']).agg(
        wv=('wv','sum'), wd=('wd','sum'), pa=('woba_denom','sum')).reset_index()
    sp_rows = [{'player_name': r['name_norm'], 'p_throws': r['p_throws'],
                'pitch_type': r['pitch_name'], 'woba': r['wv']/r['wd'], 'pa': r['pa']}
               for _,r in gp.iterrows() if r['wd'] > 0 and r['p_throws'] in ('L','R')]
    sp_loaded = pd.DataFrame(sp_rows) if sp_rows else empty_sp
    proj_woba = {nm: round(v,4) for (nm,h),v in woba_lookup.items() if h=='OVERALL'}
    obp_lookup = {nm: v for (nm,h),v in woba_lookup.items() if h=='OVERALL'}
    return woba_lookup, pa_lookup, sp_loaded, proj_woba, obp_lookup, collisions

def resolve_batter_key(name, mlbam_id, collisions):
    """Map a roster player to his lookup key. Only matters when two active
    batters genuinely share a name (~1 case a season); otherwise returns the
    name unchanged."""
    if not collisions or name not in collisions:
        return name
    try:
        return collisions[name].get(int(mlbam_id), name)
    except (TypeError, ValueError):
        return name

PITCH_MIX_WINDOW_DAYS = 365      # a pitcher's ARSENAL is current-state, not career

def build_pitcher_mix(df, as_of_date=None, window_days=PITCH_MIX_WINDOW_DAYS):
    """{pitcher_id: {pitch_type: pct}} from the pitcher's RECENT arsenal.

    Window matters here in a way it does not for hitter talent. A batter's ability is
    reasonably stable across three years, which is why the wOBA cascade uses the long
    window. A pitcher's MIX is not: arsenals turn over constantly (a sweeper added,
    a slider shelved, usage reshuffled after a pitching-coach change). Averaging three
    years of pitches projects hitters against pitches the guy has stopped throwing.

    The mix feeds the pitch-type leg of the cascade, which carries 70% of the weight,
    so this is not a rounding detail: a pitcher who swapped his slider for a sweeper
    shows ~26% slider over three years versus ~5% today.

    The daily script already used the most recent 1,500 pitches per pitcher; the
    backfill used the full 3-year frame, so the two disagreed on the same pitcher for
    the same matchup. This aligns them on a trailing-year window (point-in-time: only
    pitches strictly before as_of_date).
    """
    if df is None or df.empty or 'pitcher' not in df.columns:
        return {}
    d = df
    if as_of_date is not None and 'game_date' in d.columns:
        asof = pd.to_datetime(as_of_date)
        lo = asof - pd.Timedelta(days=window_days)
        d = d[(d['game_date'] < asof) & (d['game_date'] >= lo)]
    if d.empty:
        return {}
    d = d.copy()
    d['pt'] = d['pitch_name'].map(lambda x: PITCH_NAME_MAP.get(x, 'Other'))
    mix = {}
    for pid, grp in d.groupby('pitcher'):
        counts = grp['pt'].value_counts(); total = counts.sum()
        if total > 0:
            mix[int(pid)] = {pt: round(counts.get(pt,0)/total, 4) for pt in PITCH_TYPES}
    return mix

def wpa_by_batter_game(df):
    out = {}
    if df.empty or 'delta_home_win_exp' not in df.columns:
        return out
    w = df[['batter','game_pk','inning_topbot','delta_home_win_exp']].dropna(
        subset=['batter','game_pk','delta_home_win_exp']).copy()
    w['wpa'] = w['delta_home_win_exp'].where(w['inning_topbot']=='Bot',
                                             -w['delta_home_win_exp'])
    for (bid,gpk), v in w.groupby(['batter','game_pk'])['wpa'].sum().items():
        out[(int(bid), int(gpk))] = float(v)
    return out

def name_to_id_from(batter_names):
    """{normalized_batter_name: mlbam_id} — the inverse of build_batter_name_map.

    Used only as a fallback when the game feed's box score does not yield an id for
    a player. The old version built this from Statcast's `player_name` column, which
    is the PITCHER on every row, so it mapped pitcher names to whatever batter they
    happened to face last. Every id it returned was wrong.

    When two batters share a name the id is ambiguous; those are skipped rather than
    guessed, since the caller already prefers box-score ids and a wrong id silently
    attributes another player's WPA to the decision.
    """
    seen, dupes = {}, set()
    for bid, nm in (batter_names or {}).items():
        if nm in seen and seen[nm] != bid:
            dupes.add(nm)
        else:
            seen[nm] = int(bid)
    for nm in dupes:
        seen.pop(nm, None)
    return seen

# ── PROCESS ONE DATE ─────────────────────────────────────────────────────────
def process_date(target_date, oaa_lookup, all_statcast, debug=False):
    games = pull_schedule(target_date)
    if not games:
        return [], []
    hist_df  = slice_before(all_statcast, target_date)   # in-memory, no re-fetch
    today_df = slice_on(all_statcast, target_date)       # in-memory, no re-fetch
    roster_lk = pull_roster_asof(target_date)

    combo = pd.concat([hist_df, today_df], ignore_index=True) if (not hist_df.empty or not today_df.empty) else pd.DataFrame()
    batter_names = build_batter_name_map(combo)   # cached; covers today's debutants
    (woba_lookup, pa_lookup, sp_loaded,
     proj_woba, obp_lookup, collisions) = build_woba_lookups(hist_df, batter_names)
    # trailing-year arsenal, not 3-year career (hist_df is already point-in-time)
    pitcher_mix = build_pitcher_mix(hist_df, as_of_date=target_date)
    wpa_lk = wpa_by_batter_game(today_df)          # that-day WPA (for manager pick)
    # Model pick's expected WPA: recency-weighted trailing-YEAR avg (point-in-time).
    # Stable from opening day (always a full year behind it) and leans on recent games.
    wpa_avg = wpa_recency_weighted(hist_df, target_date, window_days=365)
    name_to_id = name_to_id_from(batter_names)
    league_avg = (sorted(proj_woba.values())[len(proj_woba)//2] if proj_woba else 0.320)

    # diagnostic tallies
    dbg = {'pairs':0, 'no_mgr_id':0, 'no_mgr_wpa':0, 'no_model_id':0, 'no_hist_zerofill':0, 'ok':0}

    season_rows, decision_rows = [], []
    for game in games:
        gpk = game.get('gamePk')
        live = get_final_feed(gpk)
        if live is None:
            continue
        home_id = game.get('teams',{}).get('home',{}).get('team',{}).get('id')
        away_id = game.get('teams',{}).get('away',{}).get('team',{}).get('id')
        home_abbr = TEAM_ID_MAP.get(home_id, str(home_id))
        away_abbr = TEAM_ID_MAP.get(away_id, str(away_id))

        for side, team_abbr in [('home', home_abbr), ('away', away_abbr)]:
            actual, lineup_ids = starters_from_feed(live, side)
            if len(actual) < 9:
                continue
            box_ids = roster_ids_from_feed(live, side)   # name->id incl. bench
            opp_pid, p_hand, opp_name, starter_bf, pen_rhp_share = opp_starter_from_feed(live, side, as_of_date=target_date)
            mix = pitcher_mix.get(int(opp_pid), {pt:0.0 for pt in PITCH_TYPES}) if opp_pid else {pt:0.0 for pt in PITCH_TYPES}
            w_start = pc.starter_pa_weight(starter_bf)   # Piece 1: starter's PA share

            # Guard against the CSV header importing as a data row (a player
            # literally named 'player_name' on a team named 'team').
            team_roster = [info for (nm,tm),info in roster_lk.items()
                           if tm == team_abbr and info.get('position','') != 'P'
                           and info.get('player_name') != 'player_name']
            if not team_roster:
                continue

            # The optimizer's candidate pool is the active-roster snapshot for this
            # date, but the MLB API's roster-as-of endpoint can miss a player who
            # actually started (transaction processed late, call-up not yet
            # reflected). When that happens he's absent from the pool while still
            # appearing in the actual lineup, so the "optimal" lineup is built from
            # a strictly smaller set than the manager had — and the actual lineup can
            # score HIGHER than the optimum, which is impossible by construction.
            # (Observed in 26 of 2,866 team-games — 0.9% — clustered in SF's early
            # April, i.e. one missing player across a run of games.)
            #
            # Fix: anyone who actually started is by definition available, so union
            # them into the pool. Their position comes from the game feed's box
            # score. This makes the optimum a true superset of the actual lineup.
            _have = {info['player_name'] for info in team_roster}
            _missing = [nm for nm in actual if nm not in _have]
            if _missing:
                box_pos = positions_from_feed(live, side)
                for _nm in _missing:
                    _pos = box_pos.get(_nm, '') or 'DH'   # DH = no fielding constraint
                    if _pos == 'P':                        # never platoon pitchers
                        continue
                    team_roster.append({
                        'player_name': _nm,
                        'team': team_abbr,
                        'position': _pos,
                        'secondary_positions': '',
                        'mlbam_id': box_ids.get(_nm, ''),
                    })
                    _have.add(_nm)

            exp_lookup, comp_lookup, pos_lookup = {}, {}, {}
            for info in team_roster:
                nm = info['player_name']
                key = resolve_batter_key(nm, box_ids.get(nm), collisions)
                res = compute_player_woba(key, key, p_hand, mix, woba_lookup,
                                          pa_lookup, sp_loaded, proj_woba, league_avg)
                # BULLPEN MODEL (Piece 1 + Piece 2 only): re-weight the projection
                # across starter innings (at the starter's hand) and relief innings
                # (at the projected pen hand mix). No TTO adjustment in this build.
                blended = pc.bullpen_blend(res, p_hand, w_start, pen_rhp_share,
                                           use_pen_hand=True)
                exp_lookup[nm] = blended
                comp_lookup[nm] = composite_score(blended, oaa_lookup.get(nm,0))
                prim = info.get('position',''); sec = [p for p in str(info.get('secondary_positions','')).split(',') if p]
                pos_lookup[nm] = set([prim]+sec) if prim else set(sec)

            roster_df = pd.DataFrame(team_roster)
            optimal = select_optimal_lineup(roster_df, oaa_lookup, woba_lookup,
                                            obp_lookup, exp_lookup, p_hand,
                                            confirmed_starters=set(actual))
            opt_dict = {s:n for n,s in optimal}
            opt_names = [opt_dict.get(i,'') for i in range(1,10)]

            # league_avg as the fallback for a player with no composite: scoring him
            # 0 would penalize (or credit) a lineup for a DATA GAP rather than a
            # managerial decision, and was a source of impossible positive gaps.
            gap, n_missing = lineup_gap(actual, opt_names, comp_lookup,
                                        fallback=league_avg)
            if n_missing:
                dbg['gap_missing_players'] = dbg.get('gap_missing_players', 0) + n_missing
            platoon_opps = len(set(opt_names) - set(actual) - {''})
            season_rows.append({
                'date': target_date, 'team': team_abbr, 'opp_pitcher': opp_name,
                'opp_pitcher_hand': p_hand, 'platoon_opps': platoon_opps,
                'lineup_gap': gap, 'lineup_confirmed': 1,
                # NEW bullpen-model columns:
                'starter_bf': starter_bf if starter_bf is not None else '',
                'w_start': round(w_start, 3),
                'pen_rhp_share': round(pen_rhp_share, 3),
            })

            for mgr_pick, model_pick in pair_decisions(actual, opt_names, pos_lookup):
                dbg['pairs'] += 1
                mgr_id   = lineup_ids.get(mgr_pick) or box_ids.get(mgr_pick) or name_to_id.get(mgr_pick)
                model_id = box_ids.get(model_pick) or name_to_id.get(model_pick)
                if mgr_id is None:
                    dbg['no_mgr_id'] += 1; continue
                if (int(mgr_id), int(gpk)) not in wpa_lk:
                    dbg['no_mgr_wpa'] += 1; continue
                mgr_wpa = wpa_lk[(int(mgr_id), int(gpk))]
                # Model pick's expected WPA = recency-weighted trailing-year avg,
                # shrunk toward 0 by games played (WPA_REG_K in platoon_core).
                # A player with NO trailing-year history (true rookie, long injury
                # absence) gets 0 — the same value full regression would give him —
                # rather than being dropped. Dropping used to bias the backtest:
                # the model picks bench players, who skew young/low-PA, so decisions
                # where the model wanted a rookie were silently never scored, giving
                # the manager a free pass on exactly that category.
                if model_id is None:
                    dbg['no_model_id'] += 1; continue   # can't identify the player at all
                model_exp = wpa_avg.get(int(model_id), 0.0)
                if int(model_id) not in wpa_avg:
                    dbg['no_hist_zerofill'] += 1
                dbg['ok'] += 1
                impact = round(mgr_wpa - model_exp, 4)
                decision_rows.append({
                    'game_date': target_date, 'game_pk': int(gpk), 'team': team_abbr,
                    'manager_pick': mgr_pick, 'model_pick': model_pick,
                    'manager_actual_wpa': round(mgr_wpa,4),
                    'model_expected_wpa': round(model_exp,4),
                    'decision_impact': impact,
                    # An exact tie is neither side winning; calling it model_better
                    # silently credited the model for ~4 non-events per season.
                    'verdict': ('manager_helped' if impact > 0
                                else 'tie' if impact == 0
                                else 'model_better'),
                })
        time.sleep(0.15)
    if debug:
        print(f"      [debug] pairs={dbg['pairs']} no_mgr_id={dbg['no_mgr_id']} "
              f"no_mgr_wpa={dbg['no_mgr_wpa']} no_model_id={dbg['no_model_id']} "
              f"zero_filled(no history)={dbg['no_hist_zerofill']} ok={dbg['ok']}")
    return season_rows, decision_rows

def _append_dedup(path, rows, keys):
    """Merge `rows` into the CSV at `path`, keeping the LAST row per key.

    Keys are normalized to strings before comparing. read_csv infers dtypes, so a
    game_pk written as int can come back as float64 the moment any row has a null
    (822729 vs 822729.0 then compare as different keys and the dedup silently
    fails). Rows with a null key are dropped: they can never dedup against anything
    (NaN != NaN) so they accumulate forever, and a decision with no game_pk is
    unusable anyway.
    """
    if not rows:
        return
    new = pd.DataFrame(rows)
    combined = pd.concat([pd.read_csv(path), new], ignore_index=True) \
               if os.path.exists(path) else new
    for k in keys:
        if k not in combined.columns:
            print(f"    ⚠ dedup key '{k}' missing from {os.path.basename(path)}")
            combined.to_csv(path, index=False)
            return
    before = len(combined)
    combined = combined.dropna(subset=keys)
    if len(combined) < before:
        print(f"    dropped {before-len(combined)} rows with null keys from "
              f"{os.path.basename(path)}")
    norm = {}
    for k in keys:
        col = combined[k]
        if pd.api.types.is_numeric_dtype(col):
            norm[k] = col.astype('int64').astype(str)
        else:
            norm[k] = col.astype(str).str.strip()
    key_df = pd.DataFrame(norm, index=combined.index)
    combined = combined.loc[~key_df.duplicated(keep='last')]
    combined.to_csv(path, index=False)

def main():
    start = sys.argv[1] if len(sys.argv) > 1 else OPENING_DAY
    end   = sys.argv[2] if len(sys.argv) > 2 else \
            (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')
    print(f"Point-in-time backfill: {start} -> {end}\n")

    # CLEAN-SLATE on a full-season rebuild.
    # _append_dedup MERGES new rows into the existing CSVs and dedups on
    # (game_pk, team, manager_pick). That is correct for INCREMENTAL runs (adding
    # today's games), but WRONG when re-running the whole season after a logic fix:
    # rows written by an OLD run have different manager_pick values (the old pairing
    # bug paired a shortstop with a catcher; the fixed code pairs him correctly), so
    # they do NOT dedup against the new rows and survive forever. That is exactly how
    # a 2,600-decision rebuild left 9,000+ rows in decisions_log.
    #
    # So: if this run starts at OPENING_DAY (a full rebuild), delete the output CSVs
    # first and rebuild from scratch. A partial-range run still appends.
    if start == OPENING_DAY:
        for _f in (DECISIONS_OUT, SEASON_LOG_OUT,
                   os.path.join(DESKTOP, "pending_decisions_bullpen.csv")):
            if os.path.exists(_f):
                os.remove(_f)
                print(f"  full rebuild: cleared {os.path.basename(_f)}")
        # pending_decisions.csv is written by the DAILY script, not the backfill, so
        # a full rebuild would leave it missing until the next daily run — and Power
        # Query errors with DataSource.NotFound on refresh. Recreate an empty one
        # (header only) so the workbook can always refresh; the daily script fills it.
        _pend = os.path.join(DESKTOP, "pending_decisions_bullpen.csv")
        pd.DataFrame(columns=['game_date','game_pk','team','manager_pick',
                              'model_pick','model_expected_wpa']).to_csv(_pend, index=False)
        print("  full rebuild: recreated empty pending_decisions.csv (daily script fills it)")
        print()

    # OAA is stable over a season — fetch the CURRENT 2025+2026 numbers ONCE and
    # reuse for every date (per design decision: current OAA is a sound point-in-
    # time approximation for defensive skill).
    print("Fetching OAA (2025+2026, once)...")
    oaa_lookup = fetch_oaa_once(2026)
    print(f"  OAA loaded for {len(oaa_lookup)} players\n")

    # Load ALL Statcast data ONCE (monthly chunks), covering the 3-year window
    # before `start` through `end`. Every date then slices this in memory instead
    # of re-querying a 3-year window — the main speedup (~9hr -> well under 1hr).
    overall_start = (datetime.strptime(start,'%Y-%m-%d')
                     - timedelta(days=STATCAST_WINDOW_DAYS)).strftime('%Y-%m-%d')
    print(f"Loading Statcast {overall_start} .. {end} once (monthly chunks)...")
    print(f"\n=== backfill_season {SCRIPT_VERSION} ===")
    _cv = getattr(pc, 'CORE_VERSION', 'UNKNOWN (stale platoon_core.py!)')
    print(f"    platoon_core: {_cv}")
    if _cv.startswith('UNKNOWN'):
        raise SystemExit("STOP: platoon_core.py on disk is an OLD version missing "
                         "CORE_VERSION. The pairing/gap fixes live there. Copy the "
                         "current platoon_core.py next to this script and re-run.")
    import platoon_core as _pc
    print(f"    platoon_core: WPA_REG_K={_pc.WPA_REG_K} PA_PER_START={_pc.PA_PER_START}")
    all_statcast = load_statcast_range(overall_start, end)

    # Coverage check: a month that downloaded "successfully" but came back thin is
    # just as damaging as a failed chunk, and silent. In-season months should have
    # tens of thousands of pitches; flag anything that looks truncated.
    if not all_statcast.empty and 'game_date' in all_statcast.columns:
        _by_month = all_statcast.groupby(
            all_statcast['game_date'].dt.to_period('M')).size()
        _thin = [(str(m), int(n)) for m, n in _by_month.items()
                 if n < 5000 and 4 <= m.month <= 9]
        if _thin:
            print("    ⚠ WARNING: in-season months with suspiciously few pitches:")
            for m, n in _thin:
                print(f"        {m}: {n:,} pitches")
            print("        A truncated download leaves a hole in every later date's")
            print("        trailing window. Consider re-running before trusting output.")
        print(f"    Statcast loaded: {len(all_statcast):,} pitches across "
              f"{len(_by_month)} months")

    seed_pitcher_hands_from_statcast(all_statcast)
    _lh = sum(1 for v in _PITCH_HAND_CACHE.values() if v == 'L')
    print(f"    pitcher hands seeded from Statcast: {len(_PITCH_HAND_CACHE)} "
          f"({_lh} LHP, {len(_PITCH_HAND_CACHE)-_lh} RHP)")
    _build_pitcher_game_table(all_statcast)   # Piece 1/2: trailing BF + pen-mix
    print(f"    pitcher-game table built: {len(_PITCHER_GAME_BF)} pitcher-games")
    print(f"  Statcast loaded: {len(all_statcast):,} rows\n")

    # Debug diagnostics when running a short (<= 2 day) range, so decision-gate
    # failures are visible during testing.
    debug = False
    try:
        span = (datetime.strptime(end,'%Y-%m-%d') - datetime.strptime(start,'%Y-%m-%d')).days
        debug = span <= 2
    except Exception:
        pass

    tot_s = tot_d = 0
    for target_date in daterange(start, end):
        try:
            sr, dr = process_date(target_date, oaa_lookup, all_statcast, debug=debug)
            if sr or dr:
                print(f"  {target_date}: {len(sr)} season rows, {len(dr)} decisions")
                _append_dedup(SEASON_LOG_OUT, sr, keys=['date','team'])
                # Dedup on (game_pk, team, manager_pick) — NOT including model_pick.
                # Each benched starter yields at most ONE decision per game. Keying on
                # model_pick too meant a re-run that paired the same starter with a
                # different replacement (because lineups/model state changed between
                # runs) ADDED a second row instead of replacing the first, leaving one
                # game with contradictory pairings (e.g. tena->nunez and tena->mead).
                _append_dedup(DECISIONS_OUT, dr, keys=['game_pk','team','manager_pick'])
                tot_s += len(sr); tot_d += len(dr)
            else:
                print(f"  {target_date}: no games / nothing to log")
        except Exception as e:
            print(f"  {target_date}: ERROR {e}")
        time.sleep(0.3)
    print(f"\nDone. {tot_s} season rows, {tot_d} decisions written.")

    # Sanity check the platoon premise itself. Roughly 25-30% of MLB starts are made
    # by left-handers. If this comes back ~0% or ~100%, the hand is not being read
    # and every batter is being projected against the same side, which silently
    # neutralizes the whole model (this exact failure produced 2,866 straight 'R'
    # rows before pitchHand was resolved properly).
    try:
        _chk = pd.read_csv(SEASON_LOG_OUT)
        if 'opp_pitcher_hand' in _chk.columns and len(_chk):
            _vc = _chk['opp_pitcher_hand'].value_counts()
            _l = int(_vc.get('L', 0)); _r = int(_vc.get('R', 0))
            _pct = _l / max(_l + _r, 1) * 100
            print(f"    opposing starter hands: {_l} LHP / {_r} RHP ({_pct:.1f}% left)")
            if _pct < 10 or _pct > 50:
                print(f"    ⚠ WARNING: {_pct:.1f}% left-handed is implausible "
                      f"(expect roughly 25-30%). Pitcher hand may not be resolving.")
    except Exception as _e:
        print(f"    (hand sanity check skipped: {_e})")

    print("Refresh the season_log and decisions_log tabs in Excel.")

if __name__ == '__main__':
    main()
