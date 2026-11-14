# Plan card accuracy: were the lane calls right?

Generated 2026-11-13 21:15. Games from 2024-01-01 to 2026-05-12 (holdout not used), each scored with stats and forecast models built only from earlier games, using the same code as the card. One row per game (Blue's view). "Ahead" = gold lead at 15 in that lane (mid-jungle: mid + jungle). "Clearly" = 500+ gold. 95% intervals resample whole series.

## 1. Strong / Weak calls: how often the favoured side was ahead at 15

| Call | Strong/Weak calls | Favoured side ahead |
|---|---|---|
| Top | 4,507 | 80% [79%, 81%] |
| Mid | 4,687 | 74% [73%, 75%] |
| Bot | 4,119 | 74% [72%, 75%] |
| Mid-jungle | 4,627 | 74% [72%, 75%] |

For reference, the forecast's sign on every lane (not just Strong / Weak calls):

| Lane | Lanes | Forecast leader was ahead |
|---|---|---|
| Top | 22,371 | 64% [64%, 65%] |
| Mid | 22,374 | 61% [61%, 62%] |
| Bot | 22,368 | 61% [60%, 61%] |
| Jungle | 22,370 | 62% [61%, 63%] |
| Mid-jungle | 22,374 | 61% [61%, 62%] |

## 2. All three calls, with what actually happened

| Lane | Call | Games (share) | Ahead at 15 | Clearly ahead | Clearly behind | Average gold at 15 |
|---|---|---|---|---|---|---|
| Top | Strong | 2,807 (13%) | 80% [79%, 82%] | 58% [57%, 60%] | 8% [7%, 9%] | +742 |
| Top | Even | 17,874 (80%) | 51% [51%, 52%] | 29% [28%, 30%] | 26% [26%, 27%] | +41 |
| Top | Weak | 1,700 (8%) | 20% [18%, 22%] | 9% [7%, 10%] | 59% [56%, 61%] | -751 |
| Mid | Strong | 3,097 (14%) | 74% [72%, 75%] | 50% [48%, 51%] | 10% [9%, 12%] | +555 |
| Mid | Even | 17,694 (79%) | 51% [51%, 52%] | 27% [27%, 28%] | 25% [24%, 26%] | +27 |
| Mid | Weak | 1,590 (7%) | 25% [23%, 27%] | 10% [9%, 12%] | 50% [48%, 53%] | -539 |
| Bot | Strong | 2,782 (12%) | 73% [71%, 75%] | 54% [52%, 56%] | 14% [13%, 15%] | +671 |
| Bot | Even | 18,262 (82%) | 51% [50%, 51%] | 31% [30%, 31%] | 29% [28%, 30%] | +17 |
| Bot | Weak | 1,337 (6%) | 25% [23%, 28%] | 13% [11%, 14%] | 56% [53%, 58%] | -735 |
| Mid-jungle | Strong | 3,359 (15%) | 74% [72%, 75%] | 60% [59%, 62%] | 15% [14%, 16%] | +918 |
| Mid-jungle | Even | 17,754 (79%) | 52% [51%, 53%] | 36% [35%, 37%] | 33% [32%, 34%] | +45 |
| Mid-jungle | Weak | 1,268 (6%) | 26% [24%, 28%] | 15% [13%, 17%] | 62% [59%, 64%] | -945 |

## 3. Strong / Weak calls by tier and season (top, mid, bot together)

| Group | Calls | Favoured side ahead |
|---|---|---|
| Major | 2,758 | 74% [72%, 76%] |
| ERL | 4,454 | 77% [75%, 78%] |
| Other | 6,093 | 76% [75%, 77%] |
| 2024 | 5,620 | 78% [77%, 79%] |
| 2025 | 4,301 | 73% [72%, 75%] |
| 2026 | 3,384 | 76% [74%, 77%] |

## 4. Team gold at 15 (the number behind the break-even line)

- Forecast leader was ahead at 15 in 63% [62%, 63%] of 22,376 games.
- Average miss: 2,030 gold.

## How to read this

- 50% would be a coin flip. A Strong or Weak call is only made for about the top / bottom 10% of forecasts for each lane, so these are the card's most confident lane calls.
- Even calls are not a prediction of an even lane; they mean the forecast isn't confident either way.
- Laning tags are empty, so every call here comes from the gold forecast; filled tags would override it.
- The rule labels (Supported / Unclear) are in reports/rule_tests.md; this report only checks the lane calls the rules are built on.
