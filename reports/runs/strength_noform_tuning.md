# strength_noform tuning (2024 walk-forward only)

10,204 games, 2024-01-01 to 2025-01-01. The report window (2025 onward) was not used. elo_league on the same games: log-loss 0.6270.
**Edge warning:** the chosen value sits at the edge of the searched range for at least one parameter.


| form half-life (games) | weight half-life (days) | lam_team | lam_league | log-loss | calibration slope |
|---|---|---|---|---|---|
| — | 90 | 0.1 | 0.1 | 0.6216 **(chosen)** | 1.10 |
| — | 90 | 0.1 | 0.01 | 0.6232 | 0.98 |
| — | 90 | 0.03 | 0.1 | 0.6233 | 0.86 |
| — | 90 | 0.1 | 1 | 0.6260 | 1.14 |
| — | 180 | 0.1 | 0.1 | 0.6276 | 1.24 |
| — | 90 | 0.3 | 0.1 | 0.6305 | 1.46 |
| — | 365 | 0.1 | 0.1 | 0.6351 | 1.26 |
| — | 730 | 0.1 | 0.1 | 0.6401 | 1.23 |
| — | 90 | 1 | 0.1 | 0.6497 | 1.83 |
| — | 90 | 3 | 0.1 | 0.6655 | 1.55 |
| — | 180 | 3 | 0.1 | 0.6675 | 1.72 |
| — | 365 | 3 | 0.1 | 0.6697 | 1.69 |
| — | 730 | 3 | 0.1 | 0.6712 | 1.61 |
| — | 90 | 10 | 0.1 | 0.6747 | 1.13 |
| — | 90 | 30 | 0.1 | 0.6779 | 0.98 |