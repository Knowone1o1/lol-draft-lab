# Early-game rules: tests on past games (Part 3)

Generated 2026-11-13 21:08. Window: 2024-01-01 to 2026-05-12 (24,660 games); the frozen holdout is not used, not even as history.

**Method.** For each month, early-game stats are rebuilt from games played before it and the lane gold-at-15 forecast is refit on the 12 months before it; then every game's facts are computed with the same code as the plan card, and each rule's condition is evaluated. Win outcomes use the walk-forward team-strength logit (strength_g15_tier_cal) as a fixed offset with coefficient 1 (the A4 approach); gold and plate outcomes control for it. 95% intervals resample whole series (B = 1,000). 

**Labels.** Supported = the interval excludes zero in the direction the rule claims. Contradicted = it excludes zero in the other direction. Unclear = it includes zero, or the rule held (or didn't hold) in fewer than 150 games.

**What is tested.** Oracle's Elixir shows outcomes, not what teams tried. Most tests check the rule's premise (e.g. "this lane is behind at 15") or whether the objective it prioritises is worth more for the comps it names; none can show that following the plan wins more.

| Rule | Test | Effect | 95% CI | Games tested | Rule held in | Label |
|---|---|---|---|---|---|---|
| R19 | Games where one team took first Dragon and the other took the Grubs majority: does the Dragon team win more than team strength predicts? | +1.9 pp | [+1.0 pp, +2.9 pp] | 12,838 | 12,838 | **Supported** |
| R11 | Do early comps with bot/mid priority gain MORE from first Dragon + Grubs majority than other comps? (interaction: rule fires x share of the two objectives taken) | +1.7 pp | [-6.4 pp, +10.1 pp] | 20,982 | 2,250 | **Unclear** |
| R12 | Do late comps facing early comps lose LESS from conceding first Dragon + Grubs majority than other comps? (interaction: rule fires x share of the two objectives taken) | +5.0 pp | [-9.9 pp, +19.0 pp] | 20,982 | 647 | **Unclear** |
| R15 | Premise: a facilitator jungler facing a carry jungler is behind in jungle gold at 15 (the plan itself, ganking instead of farming, is not visible in the data). | -214 gold | [-236 gold, -194 gold] | 22,262 | 9,214 | **Supported** |
| R17 | Drafts with no Strong lane and at least one Weak lane (forecast): do they win less than team strength predicts? | -2.1 pp | [-3.6 pp, -0.8 pp] | 24,660 | 10,457 | **Supported** |
| R24 | Top lanes forecast Weak: are they behind in top gold at 15 (beyond team strength)? | -640 gold | [-673 gold, -609 gold] | 22,381 | 4,507 | **Supported** |
| R25 | Early comps with mid and bot priority: more tower plates than strength and the team gold forecast predict? | +0.62 plates | [-0.47 plates, +1.64 plates] | 22,383 | 412 | **Unclear** |
| R29 | Lane gold at 15: is blind-picking a matchup-dependent champion worse than blind-picking others (interaction blind x matchup-dependent)? | +23 gold | [-4 gold, +49 gold] | 22,011 | 12,757 | **Unclear** |
| R30 | Mid-jungle duos forecast Weak: behind in mid + jungle gold at 15 (beyond team strength)? | -553 gold | [-601 gold, -508 gold] | 22,381 | 4,627 | **Supported** |
| R40 | Against an enemy star laner: does our laner do better at 15 when we counterpicked the star (star picked first) than when the star counterpicked us? (Jungle pressure is not visible.) | +13 gold | [-15 gold, +44 gold] | 10,842 | 7,034 | **Unclear** |

Win effects are shown in percentage points at a 50% base chance; for R11 and R12 that is the change in the value of taking both objectives (vs neither).

## Not tested

| Rule | Why |
|---|---|
| R05 | Needs Lvl 1 tags (empty in the sheet). |
| R06 | Needs Lvl 1 tags and level-1 vision data. |
| R07 | Needs Lvl 1 tags (empty in the sheet). |
| R08 | Needs the jungler's start side (not in Oracle's Elixir). |
| R13 | Needs Early spike tags and objective fight data. |
| R18 | Needs vision at 15 (not loaded). |
| R27 | Describes a habit after the lead exists (comeback rate is context, not a rule effect). |
| R31 | Needs Laning / Roam / Peel / Dive tags (empty in the sheet). |
| R32 | Needs Laning tags for bot and support (empty in the sheet). |
| R38 | Needs the order of dragons (not in Oracle's Elixir). |
| R39 | Scouting checklist, not a rule effect. |
| R09 | Not testable from match data |
| R10 | Live / timeline information |
| R14 | Live / timeline information |
| R16 | Live / timeline information |
| R20 | Not testable from match data |
| R21 | Live / timeline information |
| R22 | Live / timeline information |
| R23 | Live / timeline information |
| R26 | Not testable from match data |
| R28 | Not testable from match data |
| R33 | Not testable from match data |
| R34 | Live / timeline information |
| R35 | Live / timeline information |
| R36 | Live / timeline information |
| R37 | Live / timeline information |

## How often each draft rule fires (share of team-games where it is known)

| Rule | Fires | Known in |
|---|---|---|
| R05 | – | 0% (needs labels) |
| R06 | – | 0% (needs labels) |
| R07 | – | 0% (needs labels) |
| R08 | 100% | 100% |
| R09.bot | 9% | 100% |
| R09.top | 10% | 100% |
| R11 | 5% | 100% |
| R12 | 1% | 100% |
| R13 | – | 0% (needs labels) |
| R15 | 22% | 93% |
| R17 | 21% | 100% |
| R18 | 17% | 100% |
| R19 | 100% | 100% |
| R20.bot | 9% | 100% |
| R20.mid | 10% | 100% |
| R20.top | 10% | 100% |
| R24 | 10% | 100% |
| R25 | 1% | 100% |
| R26 | 2% | 100% |
| R29.bot | 5% | 99% |
| R29.mid | 8% | 99% |
| R29.top | 22% | 98% |
| R30 | 10% | 100% |
| R31 | – | 0% (needs labels) |
| R32 | – | 0% (needs labels) |
| R33 | 20% | 100% |
| R40.bot | 12% | 100% |
| R40.mid | 13% | 100% |
| R40.top | 12% | 100% |

## Caveats

- Champion tags (comp timing, primary style) are today's sheet, not labels as of each game: a small hindsight risk.
- Grubs = the team with more void grubs (Oracle's Elixir has counts, not who took the first camp).
- Ten tests at 95%: about one in twenty labels could be wrong by chance alone.
- R24 and R30 partly check the forecast that defines Weak lanes; they confirm the card's lane calls, not the advice.
- Lane priority: Strong / Weak = the lane's forecast is in the top / bottom 10% of forecasts over the 12 months before each month (per lane; same for the mid-jungle 2v2). Cuts per month: data/rules/prio_cuts.json.
