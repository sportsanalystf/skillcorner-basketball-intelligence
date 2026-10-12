# Methodology

Every metric this project *computes* — as opposed to reads from the delivered
files — is defined here: formula, units, assumptions, data requirement, and the
limitation that matters most when reading it.

Two labels are used throughout the application and this document:

| Label | Meaning |
|---|---|
| **Delivered** | The number comes straight from SkillCorner (e.g. `shotQuality`, `contestLevel`, `bhrDefType`). Reported as-is. |
| **Derived here** | Computed by this project. Never presented as a SkillCorner metric. |

---

## 1. Court geometry and coordinate transforms

### 1.1 Court constants

FIBA dimensions, converted to feet (the delivered unit). Nothing here is fitted
from the data; the data is used only to *check* the constants.

| Constant | Value (ft) | Source |
|---|---|---|
| Court length × width | 91.86 × 49.21 | FIBA 28 × 15 m |
| Hoop centre from centre court | 40.76 | FIBA 1.575 m from the endline |
| Three-point arc radius | 22.15 | FIBA 6.75 m |
| Corner three line from centre line | 21.66 | FIBA 6.60 m, 0.90 m inside the sideline |
| Key half-width / length | 8.04 / 19.03 | FIBA 4.90 m wide, 5.80 m deep |
| Restricted area radius | 4.10 | FIBA 1.25 m |

**Validation.** With the hoop at `(-40.76, 0)`, the recomputed shooter-to-hoop
distance matches the delivered `shots.distance` to a **median absolute error of
0.010 ft** across 1,433 shots. That is a strong confirmation that the geometry
constants and the coordinate convention are both right.

### 1.2 The attacking-frame transform

```
attacking = broadcast                if possessions.leftHoop is true
attacking = (−x, −y) of broadcast    if possessions.leftHoop is false
```

Derived empirically, not assumed: joining every shot to the shooter's tracking
position at its `startFrame` gives a median |Δx| and |Δy| of **0.0000 ft** on
both branches (n = 728 / 730). The transform is a 180° rotation about court
centre, **not** a mirror in x — mirroring only x would put the attacking team's
left on the wrong side.

The transform is an involution, so the inverse is the same function. Nothing in
the pipeline rewrites coordinates in place: normalised values are always written
to new `*_att` columns and the delivered `x` / `y` stay untouched.

### 1.3 Shot zones

Zones use the vocabulary of the season aggregate files so that game-level and
season-level views are comparable:
`restricted_area`, `paint_non_ra`, `mid_range`, `corner_3_left`,
`corner_3_right`, `wing_3_left`, `wing_3_right`, `top_3`, `backcourt`, `far`.
The wing/top boundary is placed at 30° from the baseline through the hoop.

Our three-point test agrees with the delivered `three` flag on **98.4 %** of
attempts. The 1.6 % that disagree are within about a foot of the arc, which is
the same order as the tracking's own positional error, so neither label is
"wrong" — the arc is simply not resolvable at this precision.

---

## 2. Tracking engine

### 2.1 Segmentation

A player's frames are split into **segments**: maximal runs of consecutive
`frame_idx` for that player. Every derivative is computed inside a segment and
never across a gap. Without this, a player who disappears off-camera for 40
frames and reappears 30 ft away produces a 19 ft/s "sprint" that never happened.

Segments shorter than 5 frames yield NaN rather than a number.

### 2.2 Smoothing

Savitzky–Golay filter, **window 15 frames (0.60 s), polyorder 2**, applied to x
and y within each segment (`mode="interp"` at segment edges).

*Why Savitzky–Golay rather than a moving average of the same length:* a moving
average attenuates the peak of a real change of direction, which is exactly the
signal that matters for cuts, closeouts and drives. A local quadratic fit
preserves curvature while still suppressing per-frame jitter.

*Why 0.60 s:* long enough that per-frame noise (σ ≈ 0.9 ft on detected
positions) averages down, short enough that a hard cut — roughly 0.3–0.5 s —
survives.

### 2.3 Velocity, acceleration, distance

Central differences on the smoothed track, one-sided at segment edges:

```
v      = d(x_s, y_s) / dt,           dt = 1/25 s
speed  = |v|
a      = dv / dt
a_tan  = (v · a) / |v|      signed: positive = accelerating along the heading
step   = |Δ(x_s, y_s)|      per-frame distance; distance = Σ step
```

**Movement-intensity bands (derived here).** Defined from the empirical speed
distribution in this dataset, not from any SkillCorner classification:

| Band | Speed (ft/s) |
|---|---|
| standing | < 2 |
| walking | 2 – 5 |
| jogging | 5 – 10 |
| running | 10 – 15 |
| high speed | 15 – 20 |
| sprinting | ≥ 20 |

These names are conventional; the thresholds are ours. Any report that splits
distance by band is labelled accordingly in the UI.

### 2.4 What smoothing actually changes — and a finding

Measured over all ten games, comparing central-difference speed on the *raw*
delivered positions against the smoothed pipeline:

| | raw | smoothed |
|---|---|---|
| mean speed | 5.95 ft/s | 5.94 ft/s |
| 99th percentile | 18.5 ft/s | 18.5 ft/s |
| maximum | 98.1 ft/s | 48.0 ft/s |
| correlation with the delivered `speed` field | 0.999 | 0.999 |

**The delivered positions are already temporally smooth.** Differencing them
raw reproduces SkillCorner's own `speed` column at r = 0.999. Our filter
therefore does one useful job — clipping the tail, halving the implausible
maximum — and essentially nothing to the bulk of the distribution. This is
worth knowing before anyone spends compute on a heavier filter: there is no
hidden signal to recover.

Plausibility gates: speed > 36 ft/s and |a| > 40 ft/s² are flagged (not
removed). After smoothing, 0.00 % of rows exceed the speed gate and 0.03–0.09 %
exceed the acceleration gate.

---

## 3. Possession and chance model

`possessions` is one span of team control; `chances` is one scoring opportunity
inside it. An offensive rebound keeps the possession but starts a new chance, so
**chances are the analytic denominator** for points-per-possession style
numbers, exactly as the SkillCorner primer recommends.

Per chance we compute: duration (game clock and wall clock), shot-clock used,
frontcourt delay, counts of every event family inside it, the terminal shot with
its context, and the outcome.

**Action chains.** Named action markings inside a chance are sorted by frame and
joined into a token string, e.g. `pick:over > drive:kickout > closeout`. Pick
tokens carry the ball-handler coverage, drive tokens carry the end type.
Efficiency is then grouped on the *action prefix only* — grouping on the full
chain including the terminal outcome is circular (a chain ending "made 3" always
scores 3).

---

## 4. Spatial metrics (all derived here)

Computed in the attacking frame from the five tracked offensive players and five
tracked defenders at a given frame. All return NaN rather than a guess when
fewer than three players are tracked.

| Metric | Definition | Unit |
|---|---|---|
| `hull_area` | area of the convex hull of the five offensive players | ft² |
| `mean_pairwise_dist` | mean of the 10 pairwise distances | ft |
| `mean_nn_dist` | mean distance to each player's nearest teammate | ft |
| `width` / `depth` | y-range / x-range of the offensive set | ft |
| `spread_radius` | mean distance from the offensive centroid | ft |
| `spacing_entropy` | normalised Shannon entropy of the players' bearing from the hoop, 8 bins; 1 = perfectly even distribution around the basket | 0–1 |
| `n_in_paint`, `n_in_corners`, `n_beyond_arc` | occupancy counts | players |
| `nearest_def_dist`, `second_def_dist` | 1st and 2nd nearest defender to a focal player | ft |
| `help_gap` | `second_def_dist − nearest_def_dist`; how isolated the on-ball defender is | ft |
| `n_def_within_{4,6,10}ft` | defenders inside a radius of the focal player | count |
| `n_def_in_paint`, `nearest_def_to_rim` | rim protection | count / ft |

**Validation.** Our `nearest_def_dist` at shot release reproduces SkillCorner's
own `closestDefDist` to a **median absolute difference of 0.01 ft**, with 98.2 %
of shots within 1 ft. The shooter's position from tracking matches the delivered
`shots.location` to 0.005 ft. That is an end-to-end check of the frame join, the
rotation and the geometry in one number.

### 4.1 Court control (derived here — an analytical model)

A time-to-arrive surface over a 60 × 34 half-court grid.

```
t_p(cell) = reaction + |cell − (p + v_p · reaction)| / v_max
control(cell) = σ( β · ( min_d t_d(cell) − min_o t_o(cell) ) )
```

with `reaction = 0.20 s`, `v_max = 21 ft/s` (the 99.9th percentile of smoothed
player speed here) and `β = 1.2 s⁻¹`, so a half-second time advantage maps to
about 0.65 control.

**Assumptions, stated plainly.** Straight-line travel; no turning cost; no
acceleration limit; every player equally fast; no ball-flight time; no
possession of the ball. This is the standard time-to-intercept simplification
from football pitch-control work. The output is a **relative accessibility
surface**, not a probability of winning a loose ball, and it is labelled that
way in the app. Where the model would be too fragile to say anything, the app
shows the simpler nearest-defender geometry instead.

### 4.2 Court value and spatial advantage (derived here)

```
value(cell)     = points(cell) × P(make | distance to hoop)
                  P(make) = 0.10 + 0.55 · σ((13.5 − r) / 7.5)
                  points  = 3 beyond the arc, 2 inside, 0 in the backcourt
advantage(cell) = (control(cell) − 0.5) × 2 × value(cell)
```

The advantage index summed over the half court gives one number per frame, which
the app plots through a possession. A rising line means the offence is gaining
access to *valuable* space, not merely to space. **This index has no external
validation.** It is a descriptive lens, and the app says so next to every chart
that uses it.

---

## 5. Expected field goal (xFG) — derived here

The `shots` table already carries SkillCorner's own `shotQuality` (0–100). That
is reported unchanged everywhere in the app. Separately we fit a transparent
model so that (a) the drivers are inspectable and (b) spacing features can be
tested for marginal value.

**Specification.**

```
logit P(make) = s(shot distance) + s(closest defender distance)
              + β · [angle, shot clock, release time, dribbles before]
              + one-hot [shot type, contest level, creation type]
              + [three, catch-and-shoot, created from paint, transition, assist opportunity]
              + [hull area, nearest-neighbour spacing, defenders in paint,
                 players in corners, mean teammate distance, help gap]
```

`s(·)` are natural cubic splines (5 knots, degree 3) on standardised inputs; L2
regularisation, `C = 0.5`.

**Fitting population.** 1,322 box-score field goal attempts — the shot events
minus misses while fouled, following the FIBA convention the aggregates use.

**Validation.** 5-fold cross-validation **grouped by game**, so a game's own
shots never appear in both train and test.

| Model | Log loss | Brier | AUC |
|---|---|---|---|
| Base rate (45.2 %) | 0.6885 | 0.2477 | 0.500 |
| Logistic, no spatial features | 0.6239 | 0.2181 | 0.687 |
| **Logistic + spatial features** | **0.6228** | **0.2176** | **0.689** |
| Gradient boosting (depth 3) | 0.6479 | 0.2285 | 0.658 |
| SkillCorner `shotQuality` (their model) | — | — | **0.711** |

Three honest observations:

1. **SkillCorner's index beats ours** on this sample (0.711 vs 0.689). Ours is
   for interpretation, not for replacing theirs.
2. **Gradient boosting is worse than logistic regression.** At n = 1,322 the
   extra flexibility buys nothing and costs generalisation.
3. **Spatial features add ~0.002 AUC.** Once the model knows where the shot came
   from and how close the defender was, team spacing carries almost no extra
   information about whether *this* shot goes in. Reported as a negative result
   rather than dropped.

Calibration is good in the middle and slightly conservative at the extremes:
the largest bin-level gap between predicted and observed is 8.6 percentage
points, and the overall predicted mean matches the observed mean to under 1 pp.

**`shot_value_added` = points − (xFG × points if made)**, summed per player. At
10–30 attempts per player this is dominated by shooting variance, and the app
says so.

---

## 6. Drive metrics

Outcomes are decoded from the delivered `bhrOutcomes` token list using the
`touches` vocabulary. No outcome is inferred that the tokens do not state.

| Metric | Definition |
|---|---|
| `points_direct` | 3 if `FGM3`, else 2 if `FGM`, else 0 |
| `points_created` | 3 if `AST3`, else 2 if `AST2`, else 0 |
| `points_total` | `points_direct + points_created`. Excludes free throws, matching the aggregate file's `points` column so the two are comparable |
| `drew_foul` | any of `FOUL`, `FOU_S`, `FOU_N`, `FOU_B`, `FOU_O`, `FOU_T`, `FOU_F` |
| `successful` | scored, assisted, or drew a foul — SkillCorner's own aggregate definition |
| `drive_gain_ft` | distance to hoop at drive start minus at drive end |
| `reached_paint` | drive end location inside the key |

**Defensive collapse (derived here).** Every 5 frames (0.2 s) from drive start
to drive end (capped at 5 s), we record the nearest and second-nearest defender
distance to the ball handler and the number of defenders in the paint.

```
help_closed_ft        = second_def_dist at drive start − its minimum during the drive
paint_defenders_added = max defenders in the paint − defenders in the paint at start
```

8,351 samples across 660 drives.

---

## 7. Pick-and-roll metrics

Coverage vocabularies are taken **verbatim** from the data and never extended:

* `bhrDefType`: `over`, `under`, `switch`, `ice`, `blitz`, `none`
* `scrDefType`: `soft`, `show`, `switch`, `ice`, `blitz`, `none`

Handler and screener outcomes are decoded from `bhrOutcomes` / `scrOutcomes`
with the same token rules as drives.

**Roll / pop / stay (derived here).** The dataset has no roll-or-pop field. We
classify the screener's action from tracking: change in distance to the
offensive hoop over the 38 frames (1.5 s) after the screen.

| Label | Rule |
|---|---|
| roll | moved ≥ 4 ft closer to the rim |
| pop | moved ≥ 4 ft further from the rim |
| stay | net movement < 4 ft |

Distribution over 1,520 picks: 66.1 % roll, 31.1 % stay, 2.8 % pop. The 4 ft
threshold is a judgement call; it is stated in the UI wherever the label appears.

**Comparability with the season aggregates.** The aggregate file counts only
picks where `direct = true`. Our tracked rate is 54.1 %, consistent with the
documented league-wide ~52 %. Any comparison against `handler_total_picks`
filters on `direct` first.

---

## 8. Defensive metrics

### What is computed

| Metric | Definition |
|---|---|
| `distance_closed_ft` | `startDistance − endDistance` on a closeout |
| `close_rate_fps` | distance closed ÷ closeout duration |
| `closeout_speed_fps` | defender's own path length ÷ duration |
| `arrived_tight` | `endDistance ≤ 4 ft` |
| `stop` | closeout conceded no points and no foul |
| matchup workload | seconds and assignments per defender from the `matchups` intervals, plus the points the guarded player scored inside those intervals |
| nearest-defender profile | volume, mean distance and opponent FG% for shots where a player was closest at release |

### What is deliberately *not* computed

* **Any single defensive rating.** The season aggregates are offence-only, so a
  defensive number would rest entirely on ten games — 20 to 60 chances per
  defender. Points allowed on a matchup also confounds team scheme with the
  individual. The app publishes volume, geometry and outcomes, and refuses to
  collapse them into one number.
* **Contest quality independent of the shooter.** Whether a contest was "good"
  cannot be separated from the shooter's own decision at this sample size.
* **Season-level defensive rate stats.** The data does not exist.

---

## 9. Player similarity and role clustering

Built on the **season** aggregates (293 games), not the ten tracked games,
because style comparison needs volume.

1. **Feature vector** (22 features): shot mix (rim / mid-range / three / corner
   share, catch-and-shoot share, contested share, average distance, assisted
   share, attempts per game, free-throw rate), drive behaviour (drives per game,
   kick-out / pull-up / rim end rates, blowby rate, assist rate) and
   pick-and-roll role (handler and screener picks per game, handler shot / pass
   / assist rates, role ratio). Volume features are per game so that a starter
   and a rotation player with the same style land near each other.
2. **Eligibility:** ≥ 8 games and ≥ 40 attempts → 233 player-team seasons.
3. **Standardise** (z-score), median-impute the few gaps, **PCA** to the
   components covering 85 % of variance → 9 components, 87.6 % explained.
4. **Similarity:** cosine similarity in PCA space. For each pair the app also
   shows the three standardised features that are closest, so the user can see
   *why* two players look alike. PCA rather than UMAP precisely because the map
   is linear and therefore explainable.
5. **Clustering:** K-means over the same space; k chosen by silhouette across
   k = 3…9.

**The clustering result is itself a finding.** Silhouette peaks at k = 3 with a
score of only 0.266 and never exceeds 0.27 anywhere in the range. Offensive
style in this league is closer to a **continuum** than to well-separated
archetypes. The app reports the k = 3 partition (interior finisher / screener,
movement-and-spot-up shooter, on-ball creator) as the statistically supported
one, and offers a k = 6 partition alongside it explicitly flagged as
exploratory. Cluster names are never hard-coded — they are written after reading
the standardised feature means, and the raw table sits next to them so a reader
can disagree.

**Stated limits.** This describes offensive *style*, not quality, and does not
mean two players are interchangeable. Nothing about defence enters the model
because the source data does not contain it.

---

## 10. Statistical practice

* Bootstrap confidence intervals (4,000 resamples) for every difference of means
  that is reported as a finding.
* Welch's t-test where variances plainly differ; χ² for rate comparisons.
* Cross-validation **grouped by game**, never row-random, because shots within a
  game are not independent.
* Negative and null results are reported with the same prominence as positive
  ones — see findings 3, 6 and 8 in `CASE_STUDIES.md`.
* Where a claim depends on a sample of fewer than ~100 events, the sample size
  is printed next to it in the UI.

No statistical test appears in this project unless it answers a basketball
question that was asked first.
