# Analytics — the basketball reading

`METHODOLOGY.md` says how each number is computed. This document says what it
means on a basketball court, how to read it, and where it misleads.

---

## 1. The unit of analysis: chance, not possession

A **possession** is one span of team control. A **chance** is one scoring
opportunity inside it. An offensive rebound keeps the possession alive but
starts a new chance.

This matters for every efficiency number. "Points per possession" in common
basketball usage almost always means points per *chance* — each shot attempt or
turnover is its own opportunity. In this dataset there are 2,055 chances across 1,564 possessions — 31 % more
chances than possessions — so mixing them up moves a team's offensive rating by
nearly a quarter.

**Everything in this platform is per chance** unless it explicitly says
otherwise. League baseline in these ten games: **0.84 points per chance**.

Reading the derived fields:

| Field | Reading |
|---|---|
| `start_type` | how the chance began — `DEFOB` (dead-ball inbound) and `FGDRB` (defensive rebound) dominate; `FGORB` is a second chance |
| `transition` | SkillCorner's flag. 17 % of chances |
| `quality_index` / `usable` | tracking confidence 0–5; `usable` means ≥ 3. Filter on it for anything spatial |
| `frontcourt_frame` | the first frame the ball is possessed in the front court — the difference from `start_frame` is how long the team took to get organised |

---

## 2. Shots

### The three shot-quality numbers, and when to use which

| Number | What it is | Use it for |
|---|---|---|
| `contest_level` | SkillCorner's 5-level classification at release (`open`, `light`, `average`, `plus`, `blocked`) | Fast filtering and grouping. Behaves monotonically: 54.8 % → 51.6 % → 42.0 % → 36.8 % → 0 % |
| `shot_quality` | SkillCorner's own 0–100 index | The most accurate single quality number here (AUC 0.711). Use it when you want *their* answer |
| `xfg` | Our logistic model | Use it when you need to see *why* — the coefficients are inspectable, and it is the only one that can be decomposed by feature |

Never present `xfg` as a SkillCorner metric, and never average the three.

### Reading a shot profile

The three quantities that actually separate shooters, in order of reliability at
this sample size:

1. **Shot mix** (`three_rate`, `rim_rate`, `zone` distribution) — stable at
   small n, because it is a decision, not an outcome.
2. **Shot quality generated** (`shot_quality`, `xfg`, `open_rate`) — the shots a
   player gets. Moderately stable.
3. **Conversion versus expectation** (`shot_value_added`) — the least stable.
   At 10–30 attempts this is mostly variance. Read it as a description of what
   happened, never as a skill estimate.

`is_box_score_fga` implements the FIBA convention: a miss while fouled is a free
throw trip, not a field goal attempt. Use it whenever you compare to a box
score; use raw shot events when you want every attempt the tracking saw.

### Creation lineage

Each shot is labelled with the action detected in the 4 seconds before it
(`creation_type`). The largest group is **unstructured** — no named action
detected — and it is also the most efficient. That is a taxonomy fact, not a
tactical one: it captures transition, putbacks and rebound finishes. Any
shot-creation study built only on named actions silently drops the most
efficient third of an offence. See case study 8.

---

## 3. Drives

A drive is an advantage-creation event. Judge it by what it *produced*, not by
whether the driver scored.

```
points_per_drive = (own field goal points + assisted field goal points) / drives
```

Free throws are excluded so the number is comparable with the season aggregate's
`points` column. League baseline: **0.559 points per drive, 38 % successful**.

| End type | n | pts/drive | Reading |
|---|---|---|---|
| `shotNearBasket` | 246 | 0.760 | got all the way. Also draws 19.9 % fouls |
| `pullup` | 96 | 0.781 | efficient, but generates no help rotation |
| `interiorPass` | 64 | 0.469 | drop-off or dump-off |
| `kickout` | 184 | 0.418 | the advantage was created and mostly not converted |
| `turnover` | 28 | 0.000 | |
| `pullout` | 13 | 0.000 | reset — the drive failed |

**`blowby` is the cleanest single indicator of individual advantage creation**:
it says the handler beat his man, independent of whether a teammate then made
the shot. Pair it with `help_closed_ft` to separate "beat his man" from "moved
the defence".

The kick-out row is the standing question in this dataset. Kick-outs pull 7.0 ft
of help — nearly as much as rim attacks — and return 0.42 points per drive
against 0.76. Advantage created, advantage not cashed.

---

## 4. Pick-and-roll

Two roles, two vocabularies, and they are not interchangeable.

**Ball-handler coverage** (`bhrDefType`) — what the on-ball defender does:
`over` (fights over the screen, 66 % of picks), `under` (goes beneath it, 18 %),
`switch` (14 %), `ice` and `blitz` (rare).

**Screener coverage** (`scrDefType`) — what the big does: `soft` (drop, 44 %),
`show` (steps up, 40 %), `switch` (14 %), `ice`, `blitz`.

Read them together. `over / soft` is a conventional drop coverage; `over / show`
is a hedge; `switch / switch` is a full switch. The coverage-pair matrix on the
Pick & Roll page is the right view.

### What to look at, by role

**Ball handler:** `ppp` overall, then split by coverage. The interesting player
is the one whose profile *changes* with coverage — someone who scores 0.55 per
pick against drop and 0.30 against switch has a specific, coachable problem. The
app plots each handler's coverage splits against the league line for exactly
this.

**Screener:** `handler_ppp_when_screening` is often more informative than the
screener's own scoring. A screener who never touches the ball but whose handler
produces 0.55 per pick is doing the job. `pct_switch_forced` measures whether
the screen is a genuine threat.

**Roll rate** (derived here) is 66 % league-wide, with only 2.8 % pop. A defence
facing an ACB pick-and-roll is defending a roll, which is a large part of why
`under` is playable in this league — there is little pop threat to punish it.

### `direct` and why it matters

`direct = true` means the handler or screener shot, assisted or was fouled
straight off the pick — 54 % here, ~52 % league-wide. It is the subset the
season aggregates count. Filter on it before comparing anything to
`handler_total_picks`, or you will see roughly double the volume.

It is also a useful signal in its own right: a coverage with a low direct rate
(`under`, 44.6 %) is removing the action from the play rather than defending its
outcome.

---

## 5. Spacing

Six numbers, and they say different things:

| Metric | Reads as |
|---|---|
| `hull_area` | how much floor the five occupy. Coarse but intuitive |
| `mean_nn_dist` | how far apart neighbours are — closest to the coaching notion of "spacing" |
| `width` / `depth` | horizontal versus vertical stretch. Width is the corner-occupancy story |
| `spacing_entropy` | how evenly the offence surrounds the rim, 0–1. Catches a team bunched on one side that still has a large hull |
| `n_in_corners` | corner occupancy — the single most actionable one, because corner spacing is a decision |
| `n_def_in_paint` | the defensive answer: is the space actually being conceded? |

League baselines over 1,910 chances: hull 579 ft², nearest-neighbour 14.9 ft,
width 36.9 ft, depth 29.1 ft, 0.49 players in a corner, 0.71 offensive and 1.37
defensive players in the paint.

**How to read spacing honestly.** Spacing improves the inputs to a good shot —
the widest quintile of chances gets 4.62 ft of defender separation and 45.7 shot
quality against 4.28 ft and 43.1 in the tightest. It does not add information
about whether a *given* shot goes in, because that information is already
carried by shot location and defender distance. And the widest quintile turns
the ball over more (14.5 % vs 9.8 %) — longer passes. See case study 7.

---

## 6. Defence

The honest framing: this platform measures **defensive activity and geometry**,
not defensive value.

### Closeouts — the one defensive action the data measures well

Three distance checkpoints and a duration make a closeout genuinely measurable.

| Number | Reading |
|---|---|
| `start_distance` → `end_distance` | 14.3 → 5.4 ft league-wide |
| `close_rate_fps` | how fast the gap shrinks, 7.0 ft/s league-wide |
| `arrived_tight` | end distance ≤ 4 ft |
| `bhr_action` | what the offence did: shot 47 %, drive 21 %, pass 19 %, other 12 % |
| `stop` | no points, no foul. 68 % league-wide |

**The response matters more than the speed.** A closeout that forces a pass
concedes 0.36 points and stops the possession 86 % of the time; one that
concedes a shot allows 0.91. And the pass-forcing closeouts are the *slowest*
group (6.5 ft/s), which points at balance and angle rather than speed.

Arriving tight cuts points from 0.77 to 0.56 and more than doubles the foul rate
(11.6 % vs 4.7 %). In the bonus that trade is roughly neutral; out of it, take it.

### Matchups

`matchups` gives assignment intervals, so you can say who guarded whom and for
how long, and what that attacker produced in the interval. Read it as
**workload and assignment**, not as a rating: points allowed on a matchup
includes everything the scheme did around the defender.

### Nearest-defender profiles

Being closest at release is not the same as being the assigned defender, and
being closest on a make is not evidence of bad defence. The app publishes this
with a minimum of 15 contests and an explicit warning; the standard error on
opponent FG% at that volume is about 13 points.

---

## 7. Help defence and the collapse curve

The most distinctive thing tracking adds. For any drive, sampling every 0.2 s:

* **nearest defender** — the on-ball matchup
* **second-nearest defender** — the proxy for help
* **defenders in the paint** — rim protection arriving

League behaviour: help closes **7.35 ft** (12.30 → 4.95) and the paint gains
**1.78 defenders**.

Read the curve, not the endpoint. A drive that pulls help *early* has created an
advantage the offence can still use; one that pulls it only at the rim has
created a contest. And note the ordering: the drives that draw the most help are
the drives that score the most, because help is the *consequence* of an
advantage, not its refutation.

---

## 8. Transition

Transition is a tempo decision, not an efficiency decision. Better shots (50.6
quality vs 41.9), far less contested (56 % vs 75 %), in half the time (5.9 s vs
12.8 s) — and the same 0.83 points per chance, because the turnover rate rises.

The scouting use is in the tails: which teams get a shot up fastest, which
concede the most in transition, and which convert the quality edge instead of
giving it back at the handoff.

---

## 9. Player profiles

Deliberately **multi-dimensional**. Four separate profiles, never combined:

1. **Offensive** — shot profile, drive profile, pick-and-roll profile (both
   roles), passing, turnovers.
2. **Spatial** — court occupancy, preferred locations, spacing contribution.
3. **Defensive** — closeout geometry, matchup workload, contest behaviour.
4. **Physical** — distance, distance per minute, mean and peak speed,
   acceleration and deceleration, distance by movement-intensity band.

There is no composite score, on purpose. Combining a 20-attempt shooting number
with a 12-closeout defensive number and a 25-minute physical number into one
figure would hide exactly the uncertainty that matters. The radar chart on the
comparison tab shows **percentile ranks within the tracked population**, and says
so.

### Similarity and clusters

Similarity is cosine distance in a 9-component PCA space built from season
offensive style. It answers "who plays like this" — not "who is as good as
this", and not "who could replace this player". The app shows the three features
that drive each match so the answer is auditable.

The clustering result is a finding in itself: silhouette peaks at 0.266 (k = 3)
and never exceeds 0.27. **ACB offensive roles are a continuum.** The three poles
that do emerge — interior finisher/screener, movement shooter, on-ball creator —
are real; anything finer is interpolation. Compare players by position on the
continuum, not by cluster label.

---

## 10. Reading the scouting page

Every automated note carries its metric, value, league percentile and sample
size. A note only fires when the sample reaches 20 events.

Read them in this order:

1. **Shot and action mix** — the most stable signals at this sample size,
   because they are decisions.
2. **Coverage tendencies** — how they guard the pick-and-roll, and how often
   they switch. Also decisions.
3. **Efficiency** — least reliable at one or two games. Treat as context.

And then the standing caveat, which the page prints itself: each team has one or
two tracked games here. This is a scouting *format* demonstrated on a small
sample, not a season report.
