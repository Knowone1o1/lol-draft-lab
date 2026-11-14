# Plan card: first Dragon and Grubs forecasts, backtested

Generated 2026-11-13 21:19. Walk-forward from 2024-01 to 2026-05-12 (holdout not used): each month, early stats are rebuilt from earlier games and the app's objective model (logistic regression) is refit on the 12 months before, then scores that month. Probabilities are from Blue's side; a team's chance on the card is the same number from its side. 95% intervals resample whole series.

## 1. Overall

Accuracy = the side given over 50% got it. Log-loss: lower is better; Δ = model minus the baseline (negative = the model is better). Calibration slope 1 = probabilities are the right size; below 1 = too confident.

| Objective | From | Games | Blue got it | Accuracy | AUC | Log-loss | Δ vs base rate | Δ vs Elo only | Calibration slope |
|---|---|---|---|---|---|---|---|---|---|
| First Dragon | 2024-01 | 24,084 | 39% | 61.2% [60.6%, 61.8%] | 0.579 [0.572, 0.588] | 0.6597 | -0.0100 [-0.0118, -0.0083] | -0.0076 [-0.0092, -0.0059] | 0.94 [0.85, 1.04] |
| Grubs (majority of void grubs) | 2024-03 | 19,538 | 64% | 64.3% [63.6%, 64.9%] | 0.594 [0.586, 0.603] | 0.6397 | -0.0142 [-0.0167, -0.0117] | -0.0099 [-0.0121, -0.0078] | 0.90 [0.82, 0.98] |

## 2. Calibration: when the card says X%, how often did the team get it?

| Objective | Card's chance | Team-games | Average card chance | Actually got it |
|---|---|---|---|---|
| First Dragon | 0%–30% | 1,765 | 27% | 26% [24%, 28%] |
| First Dragon | 30%–40% | 11,397 | 36% | 35% [34%, 36%] |
| First Dragon | 40%–50% | 10,922 | 44% | 45% [44%, 46%] |
| First Dragon | 50%–60% | 10,922 | 56% | 55% [54%, 56%] |
| First Dragon | 60%–70% | 11,397 | 64% | 65% [64%, 66%] |
| First Dragon | 70%–100% | 1,765 | 73% | 74% [72%, 76%] |
| Grubs (majority of void grubs) | 0%–30% | 5,679 | 25% | 27% [26%, 28%] |
| Grubs (majority of void grubs) | 30%–40% | 9,018 | 35% | 36% [35%, 37%] |
| Grubs (majority of void grubs) | 40%–50% | 4,841 | 44% | 45% [44%, 46%] |
| Grubs (majority of void grubs) | 50%–60% | 4,841 | 56% | 55% [54%, 56%] |
| Grubs (majority of void grubs) | 60%–70% | 9,018 | 65% | 64% [63%, 65%] |
| Grubs (majority of void grubs) | 70%–100% | 5,679 | 75% | 73% [72%, 74%] |

## 3. The card's warning line

The card warns when the forecast gives the team under 40% for the objective it plans to take.

| Objective | Warned | Average card chance | Actually got it |
|---|---|---|---|
| First Dragon | 13,162 (27% of team-games) | 35% | 34% [33%, 35%] |
| Grubs (majority of void grubs) | 14,697 (38% of team-games) | 31% | 33% [32%, 33%] |

## Caveats

- Grubs = the team with more void grubs; Oracle's Elixir has counts, not who took the first camp.
- These are the app's numbers before any draft-specific rule; the card's Dragon / Grubs choice comes from the rules (R11, R12, R19) and is tested in reports/rule_tests.md.
