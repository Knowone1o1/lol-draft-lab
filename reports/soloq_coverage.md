# Pro solo-queue history: coverage

Built 2026-11-12 by `scripts/soloq_ttp.py build` from cached TrackingThePros pages.

## Source and terms

- **TrackingThePros** (trackingthepros.com): pro list with known accounts and a feed of recent ranked games. robots.txt allows all crawling. The Terms of Service link returned 404 on 2026-11-12, so no terms could be read. Crawled one request at a time, 5 s apart, every response cached under `data/soloq/raw/ttp/`. It's a personal, non-commercial project; ask the site owner before publishing the raw data.
- Tried first and dropped: protracker.gg (parked domain), dpm.lol (Cloudflare challenge on every page, not bypassed), Leaguepedia API (anonymous queries rate-limited), official Riot API (no key).

## Coverage per league

Players = everyone who played in that league in 2025-26 per Oracle's Elixir (by the league of their latest game).
Mapped = found on TrackingThePros by name. With games = at least one solo-queue game collected.

| League | OE players | Mapped | % | With games | Games |
|---|---:|---:|---:|---:|---:|
| LCK | 33 | 33 | 100% | 33 | 7700 |
| LPL | 79 | 62 | 78% | 45 | 9480 |
| LEC | 48 | 46 | 96% | 43 | 8840 |
| LCS | 51 | 34 | 67% | 29 | 5540 |
| LTA N | 14 | 9 | 64% | 7 | 1494 |
| LTA S | 22 | 14 | 64% | 2 | 320 |
| LTA | 3 | 3 | 100% | 3 | 380 |
| LCP | 82 | 48 | 59% | 28 | 4940 |
| CBLOL | 67 | 39 | 58% | 27 | 5440 |
| WLDs | 115 | 114 | 99% | 108 | 23410 |
| EWC | 9 | 7 | 78% | 5 | 1220 |
| LCKC | 74 | 47 | 64% | 44 | 880 |
| PRM | 78 | 48 | 62% | 41 | 820 |
| LFL | 90 | 39 | 43% | 31 | 620 |
| NACL | 107 | 35 | 33% | 18 | 360 |
| LVP SL | 32 | 18 | 56% | 13 | 260 |
| TCL | 93 | 30 | 32% | 21 | 420 |
| NLC | 219 | 45 | 21% | 35 | 700 |
| VCS | 113 | 18 | 16% | 5 | 100 |
| PCS | 135 | 21 | 16% | 5 | 100 |
| LJL | 267 | 9 | 3% | 2 | 40 |
| EM | 367 | 176 | 48% | 156 | 3120 |
| **Total** | 2098 | 895 | 43% | 701 | 76184 |

- 76184 games, 701 pros, 973 accounts, from 2026-01-02 to 2026-11-13.
- Games per pro: median 20, history span per pro: median 31 days.
- Games by server: KR 38812, EUW 15471, NA 14495, BR 6505, JP 839.
- LPL pros' games by server: KR 9213, NA 267.

## What's missing

- **Chinese servers.** TrackingThePros, like every Riot-API-based site, can't see Chinese-server games. LPL pros are covered only when they also play on Korean (or other) accounts, mostly imports and bootcamps.
- **Depth.** The feed is paged 20 games at a time, newest first. This pull keeps only recent games, not two years: enough for current comfort and pocket picks, not for back-testing old drafts.
- **Smaller leagues.** EM, LJL, VCS, NLC and PCS players are often not on TrackingThePros.
- **Name matching.** Players are matched on their handle. A handle shared by two pros is resolved by region; matches with an unexpected region are flagged (`region_ok=0` in `ttp_players.csv`) and skipped.

## Files

- `data/soloq/ttp_players.csv`: Oracle's Elixir player ↔ TrackingThePros id, region, rank, top-3 solo-queue champions.
- `data/soloq/pro_accounts.csv`: accounts seen in the games (server, Riot ID, games, first/last seen).
- `data/soloq/matches.csv`: one row per game (date, account, champion, win, KDA, length).
- `data/soloq/oe_players_match_status.csv`: every OE training-league player and whether they were mapped.

## Next step (not done)

Turn this into draft features: per player, the share of recent solo-queue games on each champion (as of the day before each pro game), plus a 'practised recently, never played on stage' flag for pocket picks. Test whether it improves pick top-5 in the backtest before keeping it.
