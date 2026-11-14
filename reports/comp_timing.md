# Comp timing, style and win conditions

Generated 2026-11-12 16:16. 41,302 team-games (training leagues, 2023 onward, with gold at 15), each game seen from both teams. Comp timing and style mix come from the team's champion tag sheet (Early = 0, Mid = 1, Late = 2; style points primary = 2, secondary = 1). The style-matchup score multiplies the two style mixes through the sheet's Strong vs / Weak vs columns.

## 1. Which model predicts best (last 120 days, fitted on earlier games)

| Variant | What it adds | Log-loss | AUC |
|---|---|---|---|
| timing | Gold at 15, both comps' timing, gold × timing, Elo | 0.5154 | 0.824 |
| timing+style | + the sheet's style-matchup score | 0.5148 | 0.825 |
| timing+style+gold | + style-matchup score and style score × gold | 0.5148 | 0.825 |
| **timing+learned** | + style-matchup score and learned weights on the style mix | 0.5116 | 0.827 |

Used in the app: **timing+learned** (0.0037 better than timing only).

| Term | Coefficient |
|---|---|
| gold@15 | +0.561 |
| own timing | +0.097 |
| enemy timing | -0.097 |
| gold × own timing | +0.040 |
| gold × enemy timing | +0.040 |
| Elo diff | +1.423 |
| style-matchup score | +0.246 |
| style mix: Front-to-back | +0.244 |
| style mix: Hard engage | +0.310 |
| style mix: Dive | -0.315 |
| style mix: Poke / siege | -0.996 |
| style mix: Pick | -0.096 |
| style mix: Split push | +0.570 |
| style mix: Protect the carry | +0.160 |
| style mix: Disengage | +0.511 |
| style mix: Skirmish | -0.389 |

## 2. Does the data agree with the sheet's style matchups?

For each pair of main styles the sheet calls strong or weak (150+ games), the average result above or below what gold at 15, timing and Elo predict. 'Clear' = more than two standard errors from zero.

| Your main style | Enemy main style | Sheet says | Games | Result vs expected | Agrees | Clear |
|---|---|---|---|---|---|---|
| Front-to-back | Poke / siege | weak | 1021 | +5.3% | no | yes |
| Dive | Poke / siege | strong | 332 | +5.0% | yes | no |
| Hard engage | Poke / siege | strong | 917 | +4.6% | yes | yes |
| Front-to-back | Pick | strong | 917 | +0.9% | yes | no |
| Hard engage | Front-to-back | strong | 4232 | +0.9% | yes | no |
| Front-to-back | Dive | strong | 1412 | +0.1% | yes | no |
| Dive | Front-to-back | weak | 1412 | -0.1% | yes | no |
| Front-to-back | Hard engage | weak | 4232 | -0.9% | yes | no |
| Pick | Front-to-back | weak | 917 | -0.9% | yes | no |
| Poke / siege | Hard engage | weak | 917 | -4.6% | yes | yes |
| Poke / siege | Dive | weak | 332 | -5.0% | yes | no |
| Poke / siege | Front-to-back | strong | 1021 | -5.3% | no | yes |

The data agrees with the sheet on 10 of 12 pairs; 2 agree clearly and 2 disagree clearly.

## 3. Break-even gold at 15 by timing (equal teams, neutral style matchup)

| Your comp ↓ / enemy comp → | Early | Mid / tempo | Late |
|---|---|---|---|
| **Early** | -0 | +186 | +345 |
| **Mid / tempo** | -186 | -0 | +161 |
| **Late** | -345 | -161 | -0 |

Effect of the style matchup at equal timing (Mid vs Mid): break-even gold with a style score of -0.30: +132, +0.00: -0, +0.30: -132.

## 4. Which champions need a lead

Team win rate when the champion's lane is 500+ gold ahead at 15 vs 500+ behind (last two years, shrunk towards the role average). 'vs role' is that gap minus the role's usual gap: how much MORE the game rides on this champion's lane than on a typical champion in the same role. Support and jungle gaps are large for every champion because their gold mostly mirrors the whole game, so the app names win conditions by damage share plus 'vs role', not by the raw gap.

### Top (role average gap: 32%)

| Champion | Games | Win when ahead | Win when behind | Gap | vs role | Damage share | Gold share |
|---|---|---|---|---|---|---|---|
| Irelia | 79 | 73% | 31% | +43% | +11% | 24% | 24% |
| Kled | 83 | 71% | 31% | +41% | +9% | 22% | 21% |
| Aatrox | 721 | 68% | 29% | +40% | +8% | 23% | 21% |
| Kennen | 178 | 66% | 27% | +39% | +7% | 26% | 21% |
| Gangplank | 71 | 69% | 31% | +38% | +6% | 28% | 25% |
| Ornn | 648 | 74% | 37% | +38% | +6% | 17% | 17% |

### Jungle (role average gap: 49%)

| Champion | Games | Win when ahead | Win when behind | Gap | vs role | Damage share | Gold share |
|---|---|---|---|---|---|---|---|
| Ivern | 144 | 81% | 27% | +54% | +5% | 10% | 18% |
| Zaahen | 135 | 80% | 26% | +54% | +5% | 18% | 21% |
| Aatrox | 230 | 73% | 20% | +53% | +4% | 18% | 21% |
| Olaf | 78 | 75% | 22% | +53% | +4% | 20% | 22% |
| Trundle | 907 | 80% | 27% | +53% | +4% | 14% | 20% |
| Qiyana | 429 | 70% | 18% | +52% | +3% | 21% | 24% |

### Mid (role average gap: 44%)

| Champion | Games | Win when ahead | Win when behind | Gap | vs role | Damage share | Gold share |
|---|---|---|---|---|---|---|---|
| Zoe | 142 | 78% | 24% | +54% | +10% | 27% | 21% |
| Yone | 865 | 77% | 27% | +50% | +6% | 23% | 24% |
| Tristana | 205 | 71% | 22% | +49% | +5% | 26% | 26% |
| Galio | 941 | 79% | 30% | +49% | +5% | 20% | 20% |
| Sylas | 709 | 76% | 27% | +48% | +4% | 22% | 22% |
| Orianna | 1875 | 75% | 27% | +48% | +4% | 26% | 22% |

### Bot (role average gap: 44%)

| Champion | Games | Win when ahead | Win when behind | Gap | vs role | Damage share | Gold share |
|---|---|---|---|---|---|---|---|
| Zeri | 431 | 75% | 21% | +54% | +9% | 28% | 29% |
| Yunara | 1477 | 76% | 23% | +53% | +9% | 28% | 29% |
| Ezreal | 2749 | 75% | 27% | +48% | +4% | 31% | 26% |
| Mel | 124 | 64% | 18% | +46% | +2% | 25% | 26% |
| Lucian | 1390 | 71% | 26% | +46% | +1% | 29% | 27% |
| Sivir | 1119 | 74% | 29% | +45% | +1% | 28% | 28% |

### Support (role average gap: 52%)

| Champion | Games | Win when ahead | Win when behind | Gap | vs role | Damage share | Gold share |
|---|---|---|---|---|---|---|---|
| Rakan | 1889 | 80% | 22% | +58% | +6% | 6% | 10% |
| Milio | 769 | 77% | 20% | +57% | +5% | 6% | 10% |
| Thresh | 216 | 80% | 23% | +57% | +5% | 6% | 10% |
| Shen | 355 | 77% | 20% | +57% | +5% | 7% | 11% |
| Nami | 708 | 80% | 23% | +57% | +5% | 9% | 10% |
| Lulu | 1186 | 81% | 24% | +56% | +5% | 7% | 10% |

## 5. Objective paths and game length: why everything is compared at the same gold at 15

Season 2026 only (11,360 team-games), because objectives change between seasons (Atakhan was in 2025 and is gone in 2026). Objectives in this season's data: Dragon soul (4+ drakes), Baron, Elder dragon.

Early leads cause objectives and short games: a team ahead at 15 takes drakes, towers and Baron because it is ahead, and stomps end early. So raw 'objectives in wins' says little about a comp's needs. The table shows how strongly the objectives in wins depend on gold at 15:

| Gold at 15 | Team-games | Win rate | Dragon soul (4+ drakes) in wins | Baron in wins | Elder dragon in wins | Median win length |
|---|---|---|---|---|---|---|
| below -3k | 1,547 | 9% | 27% | 87% | 14% | 35.5 min |
| -3k to -1.5k | 1,677 | 24% | 29% | 88% | 8% | 33.9 min |
| -1.5k to -0.5k | 1,590 | 36% | 37% | 92% | 12% | 34.1 min |
| even | 1,732 | 50% | 41% | 90% | 10% | 33.3 min |
| +0.5k to +1.5k | 1,594 | 64% | 42% | 89% | 8% | 32.2 min |
| +1.5k to +3k | 1,674 | 76% | 44% | 91% | 7% | 30.9 min |
| above +3k | 1,546 | 91% | 39% | 89% | 4% | 27.7 min |

In the app, a comp's game-length and objective tendencies are its similar comps' results MINUS what an average comp with the same gold at 15 did. Only differences larger than the noise (1.5 standard errors, and at least 4 points) are called out. This removes the early-lead effect, but it is still an association: what happens after minute 15 (a Baron throw, a soul fight) is not in the data.

Caveat: comp timing and style are expert labels from the sheet, not learned from the data.