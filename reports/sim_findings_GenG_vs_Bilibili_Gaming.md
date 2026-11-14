# GEN vs BLG: 1,000 simulated Bo5 fearless series

Generated 2026-11-13 21:33 with the app's models (built 2026-11-12, data to 2026-11-12, patch 16.20). Each team drafts like itself (imitation model, sampling among its 8 likeliest choices at temperature 0.7); fearless locks every champion already picked in the series; the app's draft win model scores each finished draft; each series gets a random form swing (SD 0.4 on the win logit, about ±70 rating points); sides / pick order follow how 2026 series actually went. Ratings used: GEN 2050, BLG 1923.

## 1. Who wins

- **GEN wins 73.3% [70.6%, 76.0%]** of series; BLG 26.7%.
- Average series length: 3.97 games; goes to game 5 in 30%.

| Final score (GEN–BLG) | Share |
|---|---|
| 0-3 | 4.6% |
| 1-3 | 10.2% |
| 2-3 | 11.9% |
| 3-0 | 28.0% |
| 3-1 | 27.2% |
| 3-2 | 18.1% |

Per game, GEN wins 63.9% of all simulated games; the draft model alone gives GEN 65.0% on average (before form).

## 2. Does fearless change the matchup as the series goes on?

Draft-model chance for GEN by game number (before form), and how many champions were already locked.

| Game | Games played | GEN chance (draft) | Range (5th–95th pct) | Locked before the game |
|---|---|---|---|---|
| 1 | 1,000 | 65.0% | 57%–72% | 0 |
| 2 | 1,000 | 65.2% | 56%–73% | 10 |
| 3 | 1,000 | 65.0% | 57%–72% | 20 |
| 4 | 674 | 64.6% | 56%–72% | 30 |
| 5 | 300 | 64.6% | 57%–72% | 40 |

## 3. Side and first pick

| Setup for GEN | Games | GEN chance (draft) |
|---|---|---|
| Red side, second pick | 1,286 | 61.2% |
| Red side, first pick | 595 | 61.3% |
| Blue side, second pick | 634 | 68.1% |
| Blue side, first pick | 1,459 | 68.4% |

## 4. Comebacks

| Situation | Series | Came back to win the series |
|---|---|---|
| GEN down 0-1 | 352 | 51.1% [45.9%, 56.4%] |
| GEN down 0-2 | 133 | 21.8% [14.8%, 28.8%] |
| GEN down 1-2 | 268 | 40.3% [34.4%, 46.2%] |
| BLG down 0-1 | 648 | 14.7% [11.9%, 17.4%] |
| BLG down 0-2 | 423 | 5.2% [3.1%, 7.3%] |
| BLG down 1-2 | 406 | 15.0% [11.5%, 18.5%] |

## 5. The champions the series revolves around

Presence = share of series where the champion is picked or banned at least once.

| Champion | Presence | Picked by GEN (series) | Picked by BLG (series) | Banned (games) |
|---|---|---|---|---|
| Orianna | 100% | 0% | 7% | 3,731 |
| Rumble | 100% | 80% | 11% | 732 |
| Olaf | 100% | 49% | 8% | 1,944 |
| Nocturne | 100% | 22% | 3% | 2,984 |
| Nautilus | 100% | 38% | 55% | 582 |
| Lee Sin | 100% | 31% | 46% | 1,303 |
| Jarvan IV | 100% | 48% | 40% | 951 |
| Cassiopeia | 100% | 28% | 16% | 2,531 |
| Qiyana | 100% | 35% | 5% | 2,591 |
| Ezreal | 100% | 35% | 50% | 988 |
| Vi | 100% | 17% | 30% | 2,305 |
| Akali | 100% | 35% | 37% | 1,285 |

## 6. Game 1 first bans and first picks

- **GEN** first ban: Orianna 60%, Vi 10%, Ezreal 8%, Lee Sin 7%; first pick: Rumble 17%, Seraphine 8%, Ezreal 8%, Jarvan IV 8%
- **BLG** first ban: Qiyana 26%, Nocturne 25%, Cassiopeia 19%, Olaf 15%; first pick: Nautilus 18%, Ezreal 15%, Lee Sin 13%, Ambessa 10%

## 7. Player pools under fearless

Each player's most-played champion in these simulations, and how often it's still available in the last game of a series that goes to 5.

| Team | Role | Player | Most played here | Share of their games | Still open in game 5 |
|---|---|---|---|---|---|
| GEN | Top | Kiin | Rumble | 20% | 8% |
| GEN | Jungle | Canyon | Wukong | 13% | 16% |
| GEN | Mid | Chovy | Syndra | 15% | 13% |
| GEN | Bot | Ruler | Jhin | 20% | 2% |
| GEN | Support | Duro | Leona | 16% | 23% |
| BLG | Top | Bin | Gnar | 20% | 12% |
| BLG | Jungle | Xun | Naafiri | 17% | 21% |
| BLG | Mid | Knight | Locke | 15% | 27% |
| BLG | Bot | Viper | Kai'Sa | 15% | 31% |
| BLG | Support | ON | Nautilus | 14% | 6% |

## 8. Picks that move the draft model most (GEN's chance, before form)

Average chance when the team picks the champion minus when it doesn't (champions in 150+ games).

| Team | Champion | Games picked | Chance with | Without | Difference |
|---|---|---|---|---|---|
| GEN | Ashe | 576 | 67.9% | 64.4% | +3.5 pp |
| GEN | Seraphine | 557 | 67.8% | 64.5% | +3.3 pp |
| GEN | Xayah | 232 | 67.0% | 64.8% | +2.2 pp |
| GEN | Lucian | 193 | 62.0% | 65.1% | -3.1 pp |
| GEN | Milio | 173 | 61.3% | 65.1% | -3.9 pp |
| BLG | Ashe | 340 | 38.4% | 34.7% | +3.7 pp |
| BLG | Seraphine | 323 | 38.3% | 34.8% | +3.5 pp |
| BLG | Gnar | 810 | 36.6% | 34.7% | +1.9 pp |
| BLG | Lucian | 358 | 32.5% | 35.3% | -2.7 pp |
| BLG | Milio | 312 | 32.0% | 35.3% | -3.3 pp |

(Chance shown from that team's side.)

## Caveats

- The draft adds little to the win model (A4: draft champion features improve log-loss by ~0.004); most of the per-game chance is the rating gap. Draft-driven differences above are small by design.
- The win model here is the app's current one; the better-calibrated strength model from A3 (strength_g15_tier_cal) isn't in the app yet (A6).
- Teams draft like their past selves; real finals prep (pocket picks, targeted bans) is not modelled.
- The frozen holdout (from 2026-05-12) was not evaluated; the models are only used to forecast.
