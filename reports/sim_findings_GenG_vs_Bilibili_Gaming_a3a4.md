# GEN vs BLG: 1,000 simulated Bo5 fearless series (A3 + A4 win model)

Generated 2026-11-13 22:16 with the app's models (built 2026-11-12, data to 2026-11-12, patch 16.20). Each team drafts like itself (imitation model, sampling among its 8 likeliest choices at temperature 0.7); fearless locks every champion already picked in the series; the A3 strength model (strength_g15_tier_cal: tier-weighted gold-at-15 ratings, side and 2026 first-pick terms, temperature-scaled) plus the A4 draft layer (5 champion features, coefficient-1 offset, refitted through 2026-11) scores each finished draft; each series gets a random form swing (SD 0.4 on the win logit, about ±70 rating points); sides / pick order follow how 2026 series actually went. A3 pre-draft chance for GEN: blue side 60.3% (blue picks first) / 61.0% (red picks first); red side 56.2% / 55.4%.

## 1. Who wins

- **GEN wins 61.1% [58.1%, 64.1%]** of series; BLG 38.9%.
- Average series length: 4.03 games; goes to game 5 in 32%.

| Final score (GEN–BLG) | Share |
|---|---|
| 0-3 | 10.0% |
| 1-3 | 13.3% |
| 2-3 | 15.6% |
| 3-0 | 19.8% |
| 3-1 | 24.4% |
| 3-2 | 16.9% |

Per game, GEN wins 56.6% of all simulated games; the draft model alone gives GEN 57.2% on average (before form).

## 2. Does fearless change the matchup as the series goes on?

Draft-model chance for GEN by game number (before form), and how many champions were already locked.

| Game | Games played | GEN chance (draft) | Range (5th–95th pct) | Locked before the game |
|---|---|---|---|---|
| 1 | 1,000 | 57.6% | 49%–66% | 0 |
| 2 | 1,000 | 57.4% | 49%–66% | 10 |
| 3 | 1,000 | 56.8% | 48%–65% | 20 |
| 4 | 702 | 56.9% | 47%–66% | 30 |
| 5 | 325 | 57.0% | 48%–66% | 40 |

## 3. Side and first pick

| Setup for GEN | Games | GEN chance (draft) |
|---|---|---|
| Red side, second pick | 1,400 | 55.3% |
| Red side, first pick | 615 | 54.7% |
| Blue side, second pick | 622 | 59.6% |
| Blue side, first pick | 1,390 | 59.1% |

## 4. Comebacks

| Situation | Series | Came back to win the series |
|---|---|---|
| GEN down 0-1 | 413 | 37.8% [33.1%, 42.4%] |
| GEN down 0-2 | 198 | 16.7% [11.5%, 21.9%] |
| GEN down 1-2 | 295 | 27.8% [22.7%, 32.9%] |
| BLG down 0-1 | 587 | 22.5% [19.1%, 25.9%] |
| BLG down 0-2 | 339 | 8.3% [5.3%, 11.2%] |
| BLG down 1-2 | 407 | 18.7% [14.9%, 22.5%] |

## 5. The champions the series revolves around

Presence = share of series where the champion is picked or banned at least once.

| Champion | Presence | Picked by GEN (series) | Picked by BLG (series) | Banned (games) |
|---|---|---|---|---|
| Cassiopeia | 100% | 29% | 16% | 2,485 |
| Lee Sin | 100% | 32% | 48% | 1,274 |
| Orianna | 100% | 1% | 7% | 3,742 |
| Nocturne | 100% | 21% | 3% | 3,071 |
| Nautilus | 100% | 37% | 56% | 565 |
| Rumble | 100% | 81% | 11% | 771 |
| Seraphine | 100% | 56% | 31% | 617 |
| Jarvan IV | 100% | 47% | 42% | 890 |
| Vi | 100% | 16% | 29% | 2,378 |
| Qiyana | 100% | 38% | 7% | 2,469 |
| Ezreal | 100% | 37% | 51% | 902 |
| Akali | 100% | 34% | 37% | 1,362 |

## 6. Game 1 first bans and first picks

- **GEN** first ban: Orianna 60%, Vi 8%, Lee Sin 7%, Ezreal 7%; first pick: Rumble 14%, Ezreal 11%, Nautilus 9%, Jarvan IV 9%
- **BLG** first ban: Nocturne 27%, Qiyana 26%, Cassiopeia 16%, Olaf 15%; first pick: Nautilus 19%, Ezreal 13%, Ambessa 12%, Lee Sin 12%

## 7. Player pools under fearless

Each player's most-played champion in these simulations, and how often it's still available in the last game of a series that goes to 5.

| Team | Role | Player | Most played here | Share of their games | Still open in game 5 |
|---|---|---|---|---|---|
| GEN | Top | Kiin | Rumble | 20% | 8% |
| GEN | Jungle | Canyon | Wukong | 13% | 14% |
| GEN | Mid | Chovy | Syndra | 14% | 15% |
| GEN | Bot | Ruler | Jhin | 19% | 4% |
| GEN | Support | Duro | Leona | 15% | 27% |
| BLG | Top | Bin | Gnar | 20% | 11% |
| BLG | Jungle | Xun | Naafiri | 17% | 17% |
| BLG | Mid | Knight | Locke | 15% | 34% |
| BLG | Bot | Viper | Kai'Sa | 15% | 30% |
| BLG | Support | ON | Nautilus | 14% | 7% |

## 8. Picks that move the draft model most (GEN's chance, before form)

Average chance when the team picks the champion minus when it doesn't (champions in 150+ games).

| Team | Champion | Games picked | Chance with | Without | Difference |
|---|---|---|---|---|---|
| GEN | Ashe | 589 | 61.3% | 56.5% | +4.8 pp |
| GEN | Seraphine | 555 | 61.3% | 56.5% | +4.7 pp |
| GEN | Qiyana | 385 | 60.6% | 56.8% | +3.8 pp |
| GEN | Lucian | 188 | 52.7% | 57.4% | -4.8 pp |
| GEN | Milio | 161 | 52.2% | 57.4% | -5.2 pp |
| BLG | Ashe | 325 | 47.4% | 42.4% | +5.0 pp |
| BLG | Seraphine | 311 | 47.0% | 42.5% | +4.6 pp |
| BLG | Cassiopeia | 165 | 45.9% | 42.7% | +3.2 pp |
| BLG | Lucian | 392 | 39.3% | 43.2% | -3.9 pp |
| BLG | Milio | 349 | 38.7% | 43.2% | -4.4 pp |

(Chance shown from that team's side.)

## Caveats

- The draft adds little to the win model (A4: draft champion features improve log-loss by ~0.004); most of the per-game chance is the rating gap. Draft-driven differences above are small by design.
- Win model: A3 strength + A4 draft layer, fitted on all games up to the match (a forecast; nothing is scored). The app itself still uses its older win model until A6.
- Teams draft like their past selves; real finals prep (pocket picks, targeted bans) is not modelled.
- The frozen holdout (from 2026-05-12) was not evaluated; the models are only used to forecast.
