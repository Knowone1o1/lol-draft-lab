# Early game: forecasting the state at 15 minutes

Generated 2026-11-12 15:23. Every month, early-game statistics are rebuilt from games played *before* that month; models are fit on earlier months and scored on the most recent 4 months (2026-08 to 2026-11). Gold difference is from the blue side's point of view; each lane is that player against the opposite laner.

## 1. Gold difference at 15, per lane

MAE = average miss in gold. R² = share of the variation explained. Direction = how often the predicted leader was right, on lanes where someone led by 300+ gold.

| Lane | Test lanes | Model MAE | Elo-only MAE | Model R² | Direction (model) | Direction (Elo only) |
|---|---|---|---|---|---|---|
| Top | 1611 | 729 | 789 | 0.150 | 68.8% | 57.0% |
| Jungle | 1611 | 634 | 668 | 0.156 | 63.5% | 58.2% |
| Mid | 1611 | 648 | 676 | 0.128 | 64.8% | 61.5% |
| Bot | 1611 | 866 | 886 | 0.089 | 62.5% | 57.8% |
| Support | 1611 | 359 | 371 | 0.109 | 66.8% | 62.3% |

Gold per unit of each feature (features are in thousands of gold; Elo difference in units of 400):

| Lane | champ_strength_diff | matchup_lift | player_diff | team_early_diff | elo_diff |
|---|---|---|---|---|---|
| Top | +978 | +769 | +659 | +14 | +306 |
| Jungle | +990 | +16 | +533 | +33 | +339 |
| Mid | +952 | +466 | +798 | -8 | +346 |
| Bot | +724 | +363 | +678 | -43 | +551 |
| Support | +933 | +339 | +510 | +3 | +188 |

## 2. Team gold difference at 15

| Model | MAE | R² | Direction |
|---|---|---|---|
| Draft + players + teams | 2025 | 0.153 | 64.2% |
| Elo only | 2106 | 0.082 | 60.9% |
| Predict 0 | 2210 | 0 | — |

Weights: champ_strength_sum +1116, matchup_sum +721, player_sum +653, team_early_diff -6, elo_diff +1719

## 3. Early objectives

| Objective | Test games | Base rate (blue) | Model log-loss | Base-rate log-loss | Elo-only log-loss | Model AUC | Accuracy |
|---|---|---|---|---|---|---|---|
| First dragon | 1611 | 38.4% | 0.6565 | 0.6679 | 0.6630 | 0.592 | 61.6% |
| Void grubs majority | 1609 | 63.5% | 0.6511 | 0.6585 | 0.6594 | 0.566 | 63.1% |
| First blood | 1611 | 52.0% | 0.6845 | 0.6907 | 0.6900 | 0.559 | 55.0% |
| First tower | 1611 | 55.2% | 0.6406 | 0.6803 | 0.6524 | 0.663 | 63.6% |

## 4. Team profiles (2026, training leagues, 20+ games)

Across all these games, a team 1,500+ gold ahead at 15 won **83%**; a team 1,500+ behind won **17%**. Full table: `reports/team_early_profiles.csv`.

### Strongest early games (average gold lead at 15)

| Team | League | Games | Win rate | Avg gold at 15 | Won when ahead | Won when behind |
|---|---|---|---|---|---|---|
| Inferno Drive Tokyo | LJL | 40 | 70% | +3,052 | 92% (24) | 0% (4) |
| Rising Gaming | LJL | 56 | 66% | +2,327 | 90% (31) | 0% (6) |
| Misa Esports | TCL | 63 | 79% | +2,278 | 90% (39) | 71% (7) |
| FENNEL | LJL | 53 | 79% | +2,185 | 91% (32) | 20% (5) |
| Galions | LFL | 131 | 72% | +1,775 | 94% (67) | 22% (18) |
| Verdant | NLC | 73 | 70% | +1,766 | 88% (41) | 25% (12) |
| MVK Esports Academy | VCS | 54 | 65% | +1,590 | 79% (24) | 30% (10) |
| Team Heretics Academy | EM | 34 | 62% | +1,492 | 87% (15) | 0% (3) |
| SillySilly Gaming | PCS | 32 | 66% | +1,478 | 100% (16) | 14% (7) |
| French Flair | LFL | 40 | 70% | +1,365 | 83% (18) | 0% (5) |

### Most reliable closers (win rate when 1,500+ ahead, 8+ such games)

| Team | League | Games | Win rate | Avg gold at 15 | Won when ahead | Won when behind |
|---|---|---|---|---|---|---|
| SillySilly Gaming | PCS | 32 | 66% | +1,478 | 100% (16) | 14% (7) |
| CCG Esports | NACL | 51 | 53% | -689 | 100% (8) | 29% (14) |
| Uwinks | LJL | 60 | 52% | -48 | 100% (17) | 10% (21) |
| Frank Esports | PCS | 35 | 63% | +1,275 | 100% (13) | 0% (5) |
| DMG Esports | NLC | 47 | 47% | -325 | 100% (16) | 7% (15) |
| UCAM Esports | EM | 41 | 61% | +101 | 100% (14) | 21% (14) |
| Contingent Esports | NACL | 35 | 51% | -645 | 100% (8) | 13% (15) |
| NOVEX | LJL | 37 | 57% | -93 | 100% (10) | 21% (14) |
| Deer Gaming | NLC | 44 | 55% | +156 | 100% (11) | 25% (12) |
| TLN Pirates | LFL | 80 | 57% | -784 | 100% (12) | 36% (25) |

### Weakest closers: throw risk (win rate when 1,500+ ahead, 8+ such games)

| Team | League | Games | Win rate | Avg gold at 15 | Won when ahead | Won when behind |
|---|---|---|---|---|---|---|
| Team Heretics | LEC | 59 | 24% | -640 | 50% (8) | 5% (21) |
| Shifters | LEC | 61 | 31% | -235 | 53% (15) | 24% (17) |
| Bomba Team | EM | 28 | 39% | +416 | 58% (12) | 12% (8) |
| Blue Otter | NACL | 51 | 33% | -205 | 58% (12) | 0% (14) |
| EDward Gaming | LPL | 64 | 33% | -92 | 60% (15) | 6% (17) |
| Fukuoka SoftBank HAWKS gaming | LCP | 70 | 44% | +23 | 61% (18) | 20% (15) |
| Gen.G Global Academy | LCKC | 74 | 22% | -1,353 | 64% (11) | 5% (39) |
| HANJIN BRION | LCK | 99 | 39% | -230 | 64% (22) | 12% (26) |
| Ground Zero Gaming | LCP | 68 | 41% | -508 | 64% (11) | 21% (24) |
| Fnatic | LEC | 67 | 40% | -383 | 64% (14) | 18% (22) |

### Best comeback teams (win rate when 1,500+ behind, 8+ such games)

| Team | League | Games | Win rate | Avg gold at 15 | Won when ahead | Won when behind |
|---|---|---|---|---|---|---|
| Gen.G | LCK | 130 | 75% | +791 | 92% (48) | 58% (19) |
| RED Canids | CBLOL | 86 | 55% | +683 | 88% (32) | 38% (16) |
| Barça eSports | EM | 26 | 58% | +295 | 78% (9) | 38% (8) |
| G2 Esports | LEC | 136 | 65% | +605 | 84% (49) | 38% (24) |
| TLN Pirates | LFL | 80 | 57% | -784 | 100% (12) | 36% (25) |
| Unicorns of Love Sexy Edition | PRM | 105 | 59% | -797 | 85% (20) | 36% (42) |
| Bilibili Gaming | LPL | 165 | 71% | +1,001 | 90% (61) | 35% (17) |
| Dplus Kia Challengers | LCKC | 91 | 57% | -288 | 95% (21) | 35% (26) |
| New Meta | LJL | 74 | 61% | +747 | 93% (28) | 33% (15) |
| Dplus Kia | LCK | 153 | 56% | +622 | 84% (49) | 33% (27) |

## Caveats

- Gold at 15 is the outcome of draft *and* play; a draft that is 'ahead on paper' still depends on execution.
- Player early ratings mix player skill with the champions they were given; the champion and matchup terms absorb part of that.
- Some LPL games lack 15-minute data in Oracle's Elixir; they are left out of the early-game targets.
