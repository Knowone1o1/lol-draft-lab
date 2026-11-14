# strength_g15_tier tuning (2024 walk-forward only)

10,204 games, 2024-01-01 to 2025-01-01. The report window (2025 onward) was not used. elo_league on the same games: log-loss 0.6270.

| form half-life (games) | weight half-life (days) | lam_team | lam_league | margin k | log-loss | calibration slope |
|---|---|---|---|---|---|---|
| — | 90 | 0.1 | 0.1 | 0.16 / maj 0.25 | 0.6157 **(chosen)** | 0.91 |
| — | 90 | 0.1 | 0.1 | 0.16 / maj 0.5 | 0.6158 | 0.91 |
| — | 90 | 0.1 | 0.1 | 0.16 / maj 0.0 | 0.6158 | 0.91 |
| — | 90 | 0.1 | 0.1 | 0.32 / maj 0.0 | 0.6160 | 0.96 |
| — | 90 | 0.1 | 0.1 | 0.32 / maj 0.25 | 0.6160 | 0.96 |
| — | 90 | 0.1 | 0.1 | 0.16 / maj 1.0 | 0.6161 | 0.91 |
| — | 90 | 0.1 | 0.1 | 0.08 / maj 0.25 | 0.6167 | 0.88 |
| — | 90 | 0.1 | 0.1 | 0.08 / maj 0.0 | 0.6168 | 0.88 |
| — | 90 | 0.1 | 0.1 | 0.64 / maj 0.0 | 0.6185 | 1.04 |
| — | 90 | 0.1 | 0.1 | 0.64 / maj 0.25 | 0.6185 | 1.04 |
| — | 90 | 0.1 | 0.1 | 1.0 / maj 0.0 | 0.6213 | 1.10 |
| — | 90 | 0.1 | 0.1 | 1.0 / maj 0.25 | 0.6214 | 1.10 |

**Temperature scaling (2024 predictions only):** raw slope 0.907 -> applied, logit x 0.906, slope after 1.002, log-loss 0.6157 -> 0.6151