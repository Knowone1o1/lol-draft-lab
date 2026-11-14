# strength_gold25 tuning (2024 walk-forward only)

10,204 games, 2024-01-01 to 2025-01-01. The report window (2025 onward) was not used. elo_league on the same games: log-loss 0.6270.
**Edge warning:** the chosen value sits at the edge of the searched range for at least one parameter.


| form half-life (games) | weight half-life (days) | lam_team | lam_league | margin k | log-loss | calibration slope |
|---|---|---|---|---|---|---|
| — | 90 | 0.1 | 0.1 | 0.16 | 0.6151 **(chosen)** | 0.92 |
| — | 90 | 0.1 | 0.1 | 0.08 | 0.6155 | 0.89 |
| — | 90 | 0.1 | 0.1 | 0.04 | 0.6165 | 0.88 |
| — | 90 | 0.1 | 0.1 | 0.02 | 0.6173 | 0.88 |
| — | 90 | 0.1 | 0.1 | 0.01 | 0.6180 | 0.88 |