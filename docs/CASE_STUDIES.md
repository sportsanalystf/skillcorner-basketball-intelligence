# Case Studies — findings from the data

Every number below was produced by `scripts/11_findings.py` and is stored in
`reports/findings.json`. Nothing here is illustrative or rounded from memory.
Each study follows the same shape: **finding → evidence → basketball reading →
limitation**. Several of these are null or negative results; they are reported
with the same prominence as the positive ones, because that is the point.

Sample scale, stated once: ten ACB 2025/26 games with full tracking (2,055
chances, 1,459 shots, 1,520 picks, 660 drives, 601 closeouts), plus season
offensive aggregates over 293 games for the player-comparison work.

---

## 1. Defender distance tells you nothing — until you know where the shot came from

### Finding

Compare the closest-defender distance on made shots against missed shots and
you get **nothing**: 4.43 ft on makes, 4.61 ft on misses, t = −1.02, **p = 0.31**
— and the sign points the wrong way. Control for shot location and the same
variable becomes one of the strongest predictors in the model: **+0.102 log-odds
per foot (odds ratio 1.107, p < 10⁻⁵)**.

### Evidence

Marginal comparison, 1,322 box-score attempts:

| | mean closest defender |
|---|---|
| Made | 4.43 ft |
| Missed | 4.61 ft |
| Welch t | −1.02 (p = 0.31) |

Why it cancels — the confound, by zone:

| Zone | n | FG% | mean defender distance |
|---|---|---|---|
| Rim | 340 | 62.9 % | 3.16 ft |
| Short mid-range / paint | 316 | 48.1 % | 3.17 ft |
| Long mid-range | 153 | 37.3 % | 4.27 ft |
| Above-the-break 3 | 403 | 34.7 % | 5.91 ft |
| Corner 3 | 96 | 34.4 % | 6.85 ft |

The highest-percentage shots are the most heavily contested, and the
lowest-percentage shots are the most open. Aggregate the two and the effect
disappears — a clean Simpson's paradox.

Within strata, separation reasserts itself:

| Zone | made | missed | Welch t | p |
|---|---|---|---|---|
| **Rim** | 3.57 ft | 2.47 ft | +5.35 | **1.7 × 10⁻⁷** |
| **Corner 3** | 8.20 ft | 6.15 ft | +2.75 | **0.0085** |
| Short mid-range / paint | 3.41 ft | 2.94 ft | +2.28 | 0.024 |
| Long mid-range | 4.61 ft | 4.07 ft | +1.51 | 0.13 |
| **Above-the-break 3** | 5.87 ft | 5.94 ft | −0.22 | **0.83** |

Logistic model, `outcome ~ closest_def_dist + shot_distance + three`:

| Term | Coefficient | p |
|---|---|---|
| closest defender distance | **+0.1020** | 1.0 × 10⁻⁵ |
| shot distance | −0.0940 | 4.1 × 10⁻¹³ |
| three | +0.5626 | 0.022 |

### Basketball reading

Two things, one useful and one uncomfortable.

The useful one: **contest matters most where the defender can physically alter
the shot** — at the rim, where an extra foot of separation is worth 1.1 ft of
daylight between a layup and a challenged finish, and in the corner, where the
shooter has no room to drift. Rim finishing is a separation problem, not just a
touch problem.

The uncomfortable one: **on above-the-break threes, being open did not predict
making it** (p = 0.83). Contest and outcome are effectively unrelated in this
sample for the highest-volume three-point zone in the league. That does not mean
contesting is pointless — it means at this sample size the shooter's own
variance swamps the defender's contribution, and any evaluation that credits a
defender for opponent three-point misses is measuring noise.

### Limitation

1,322 attempts across ten games. The above-the-break null is a *failure to
detect* an effect, not proof of its absence: with 403 attempts the 95 %
confidence interval on the separation gap comfortably contains effects that
would matter. `closestDefDist` is also measured at release only — it says
nothing about a contest that arrived late.

---

## 2. Transition creates much better shots and exactly the same number of points

### Finding

Transition chances produce dramatically better looks — **shot quality 50.6 vs
41.9** and a contested rate of **56 % vs 75 %** — and score **exactly the same**:
0.829 vs 0.838 points per chance (p = 0.89).

### Evidence

| | Transition (n = 357) | Half court (n = 1,698) |
|---|---|---|
| SkillCorner shot quality | **50.6** | 41.9 |
| Contested rate | **56.1 %** | 75.3 % |
| Points per chance | 0.829 | 0.838 |
| Turnover rate | **13.4 %** | 11.9 % |
| Chance duration | 5.9 s | 12.8 s |
| Time to first shot | 5.8 s | 12.5 s |
| Three-point rate | 34.8 % | 37.0 % |

Shot quality difference: Welch t = 4.57, **p = 7.8 × 10⁻⁶**.
Points per chance difference: Welch t = −0.14, **p = 0.89**.

### Basketball reading

Transition buys a cleaner shot and buys it in half the time — 5.9 seconds of
chance instead of 12.8. What it gives back is the ball: the turnover rate is 1.5
points higher, and at 0.84 points per chance every extra turnover costs roughly
what the improved shot quality earns.

The practical read is that **running is a tempo decision, not an efficiency
decision**. It converts possessions faster at the same rate of return, which
raises the variance and the pace of a game without moving its expected margin.
A team choosing to run should do it because it wants more possessions or wants
to exploit a specific opponent's transition defence, not because the shots look
better — they do look better, and it does not show up in the points.

### Limitation

`transition` is SkillCorner's flag, used as delivered. 357 transition chances
gives a standard error of about 0.06 points per chance, so a true difference
smaller than ~0.12 ppc would be invisible here. Turnover rates on 357 chances
are similarly noisy.

---

## 3. Off-ball movement earns a foot of separation and converts it into nothing

### Finding

Sorting shots by how far the four *off-ball* offensive players travelled in the
3 seconds before release: the busiest quartile gains a full extra foot of
separation (95 % CI **[0.59, 1.42] ft**), is open 38.2 % of the time versus
23.6 %, and carries a **5-point-higher expected FG%** — and converts at
**exactly the same 42.6 %** (gap +0.04 pp, 95 % CI [−7.5, +7.2] pp).

### Evidence

| Quartile | Off-ball distance | n | FG% | model xFG | shot quality | open rate | separation at shot |
|---|---|---|---|---|---|---|---|
| Q1 (least) | 35.0 ft | 359 | 0.426 | 0.449 | 43.2 | 23.6 % | 3.90 ft |
| Q2 | 49.7 ft | 359 | 0.426 | 0.430 | 41.2 | 23.2 % | 4.03 ft |
| Q3 | 64.0 ft | 359 | 0.379 | 0.450 | 41.2 | 25.0 % | 4.28 ft |
| Q4 (most) | 127.9 ft | 359 | 0.426 | **0.501** | **47.7** | **38.2 %** | **4.91 ft** |

Bootstrap (4,000 resamples), Q4 − Q1:
separation **+1.005 ft, CI [0.588, 1.420]** — clearly real.
FG% **+0.0004, CI [−0.075, +0.072]** — indistinguishable from zero.

### Basketball reading

The mechanism works. Movement genuinely displaces defenders and produces
measurably cleaner catches — that is not in doubt, the separation and openness
effects are large and precise. What the ten games cannot show is the conversion.

The gap between "expected 50.1 %" and "actual 42.6 %" in the busiest quartile is
where the interesting question lives. Two explanations fit the data and this
sample cannot separate them: either heavy off-ball movement raises the physical
cost of the shot in ways an xFG model built on location and defender distance
does not see (a shooter arriving off two screens at speed is not the same
shooter as one standing still and equally open), or 359 shots is simply too few
and the true conversion follows the quality.

This is precisely the question a development staff should want answered, and it
argues for tracking shooter velocity and body orientation at release, not just
separation.

### Limitation

359 shots per quartile. The FG% confidence interval is ±7 pp — wide enough to
contain a meaningful effect in either direction. The 3-second window is a
modelling choice; shorter and longer windows were not swept.

---

## 4. The help defender closes 7.3 feet, and closes hardest on the drives that score

### Finding

Sampling defensive geometry every 0.2 s through 660 drives (8,351 samples), the
second-nearest defender closes an average of **7.35 ft** between drive start
and its closest approach (12.30 → 4.95 ft, paired t = 42.5, p ≈ 0), and the
defence adds **1.78 bodies to the paint**.

### Evidence

| Drive end type | n | pts/drive | help closed | paint defenders added |
|---|---|---|---|---|
| **Shot near the basket** | 246 | 0.760 | **8.42 ft** | **2.04** |
| Turnover | 28 | 0.000 | 7.57 ft | 1.61 |
| Interior pass | 64 | 0.469 | 7.04 ft | 1.39 |
| Kick-out | 184 | 0.418 | 7.02 ft | 1.66 |
| Stoppage | 29 | 0.000 | 6.15 ft | 1.69 |
| Pull-up | 96 | 0.781 | 6.01 ft | 1.63 |
| Pull-out | 13 | 0.000 | 5.32 ft | 2.00 |

Successful minus unsuccessful drives: +0.52 ft more help closed, 95 % CI
**[−0.19, +1.21]** — directionally consistent, not statistically separable.

### Basketball reading

The ordering is the interesting part, and it is the opposite of the naive
expectation. **The drives that draw the most help are the drives that score the
most.** Shots near the basket pull 8.4 ft of help and 2.0 extra paint defenders
and still return 0.76 points per drive; pull-ups pull the least help (6.0 ft)
and return 0.78.

Read forward rather than backward: a driver who gets deep enough to *force*
rotation has already won the possession, and the help arriving is a consequence
of the advantage, not a defence of it. The pull-up sits at the other end — it
never asks the defence to rotate, so it never generates help, and it produces
its points entirely from shot-making.

The kick-out row is the one a coach should stare at. Kick-outs draw nearly as
much help as rim attacks (7.02 vs 8.42 ft) but return 0.418 points per drive
against 0.760. The advantage is being created and then not converted — either
the kick-out is going to the wrong player, or the catch is arriving too late to
punish the rotation.

### Limitation

"Help defender" is operationalised as the second-nearest defender, which is a
proxy: on a switch or a stunt the second-nearest player may not be the one
rotating. 660 drives across ten games; the success-versus-failure comparison in
particular is underpowered.

---

## 5. Going under the screen concedes least — but the sample cannot prove it

### Finding

Ball-handler coverage ranked by points conceded per pick: **under 0.371 < switch
0.422 < over 0.468**. Under also cuts the direct-pick rate to 44.6 % against
56.5 % for over — it removes the action from the play rather than defending it.
But the over-minus-under gap is **+0.097 points per pick with a 95 % CI of
[−0.021, +0.214]**, which crosses zero.

### Evidence

| Handler coverage | picks | pts/pick | handler shot rate | assist rate | TO rate | foul rate | direct rate |
|---|---|---|---|---|---|---|---|
| over | 1,004 | 0.468 | 22.2 % | 8.6 % | 2.8 % | 5.6 % | 56.5 % |
| switch | 218 | 0.422 | 22.0 % | 9.2 % | **4.6 %** | **8.3 %** | 54.6 % |
| under | 280 | **0.371** | 20.0 % | 5.7 % | 2.5 % | 6.1 % | **44.6 %** |
| ice | 14 | 0.714 | 14.3 % | 21.4 % | 0 % | 0 % | 71.4 % |
| blitz | 4 | 0.000 | 25.0 % | 0 % | 0 % | 0 % | 50.0 % |

Screener coverage: soft 0.448, show 0.446, switch 0.422, ice 0.714 (n = 14),
blitz 0.000 (n = 4).

Screener response (our tracking-derived label): **66.1 % roll, 31.1 % stay,
2.8 % pop.**

### Basketball reading

Two patterns are solid even though the headline gap is not.

**Switching is not free.** It concedes about the same points per pick as going
over, but it does so with a 64 % higher turnover rate (4.6 % vs 2.8 %) and a
48 % higher foul rate (8.3 % vs 5.6 %). Switching trades a cleaner structural
outcome for more live-ball chaos in both directions — attractive against a team
that struggles to attack a mismatch, expensive against one that hunts it.

**Under is a suppression tactic, not a stop tactic.** Its edge shows up in the
direct rate (44.6 %) far more than in the assist or shot rates. The handler
neither shoots much more nor passes much more; the action simply resolves into
something else. Against ACB, where 66 % of screeners roll and only 2.8 % pop,
going under is a defensible read — there is no pop threat to punish it.

**Ice and blitz cannot be evaluated here.** 14 and 4 events. They are shown in
the app with their counts and no interpretation.

### Limitation

The over-under gap's confidence interval crosses zero, so this is a consistent
direction, not a proven effect. Coverage labels are SkillCorner classifications
with their own error rate, and 1,520 picks in one league over ten games is not a
basis for scheme advice.

---

## 6. Closing out tight cuts points by a fifth — and doubles the fouls

### Finding

Closeouts that arrive within 4 ft concede **0.558 points against 0.769** for
looser ones — a gap of 0.211, 95 % CI **[0.013, 0.401]** — and foul on **11.6 %
of possessions versus 4.7 %** (χ², p = 0.0035). Stop rate is identical: 67.8 %
versus 67.9 %.

### Evidence

Across 601 closeouts, the defender starts on average 14.30 ft away, arrives at
5.39 ft, and closes at 7.04 ft/s.

| Ball-handler response | n | start dist | end dist | closing rate | pts allowed | stop rate |
|---|---|---|---|---|---|---|
| **Pass** | 117 | 13.73 ft | 5.18 ft | 6.48 ft/s | **0.359** | **86.3 %** |
| Drive | 127 | 13.23 ft | 3.94 ft | 7.40 ft/s | 0.606 | 59.1 % |
| Other | 74 | 14.11 ft | 5.00 ft | 6.80 ft/s | 0.608 | 54.1 % |
| **Shot** | 282 | 15.06 ft | 6.23 ft | 7.17 ft/s | **0.908** | 67.7 % |

Tight (≤ 4 ft, n = 199) versus loose (n = 402): points 0.558 vs 0.769; fouls
11.6 % vs 4.7 %; stops 67.8 % vs 67.9 %.

### Basketball reading

The tight-closeout trade is real and it is a trade, not a free win. You take
0.21 points per closeout off the board and hand back roughly one extra foul per
fifteen closeouts. In the bonus that is close to a wash; out of the bonus it is
clearly worth doing.

The bigger lever is in the response column. **A closeout that forces a pass is
worth 0.55 points more than one that concedes a shot** (0.359 vs 0.908) and
produces a stop 86 % of the time. Note the geometry: the pass-forcing closeouts
are not the fastest ones (6.48 ft/s, the slowest of the four groups) and they
end furthest from the ball among the non-shot outcomes. Speed is not the
variable that matters — **arrival angle and balance are**, and this dataset can
measure the outcome but not the angle.

Drives conceded off a closeout are the group where defenders arrive tightest
(3.94 ft) and fastest — a signature of over-closing.

### Limitation

601 closeouts, and the points-allowed confidence interval very nearly touches
zero at the lower bound. `bhrAction` is a SkillCorner classification. The
mechanism claim about arrival angle is an inference from the geometry, not a
measurement — the dataset does not carry defender body orientation.

---

## 7. Spacing helps the shot you get, not the points you score

### Finding

Chances in the widest spacing quintile score **0.929 points against 0.813** for
the tightest — but the correlation between hull area and points is **r = 0.019
(p = 0.42)**, and the Q5 − Q1 gap of 0.117 has a 95 % CI of **[−0.050, +0.282]**.
Meanwhile, adding six spacing features to the shot model moves out-of-fold AUC
from **0.687 to 0.689**.

### Evidence

| Hull-area quintile | mean hull | n | points/chance | turnover rate |
|---|---|---|---|---|
| Q1 (tightest) | 356 ft² | 379 | 0.813 | 9.8 % |
| Q2 | 494 ft² | 379 | 0.858 | 9.8 % |
| Q3 | 580 ft² | 379 | 0.805 | 14.0 % |
| Q4 | 656 ft² | 379 | 0.924 | 11.6 % |
| Q5 (widest) | 812 ft² | 379 | 0.929 | 14.5 % |

Shot-level model comparison, 5-fold grouped by game, n = 1,322:

| Model | Log loss | Brier | AUC |
|---|---|---|---|
| Without spatial features | 0.6239 | 0.2181 | 0.687 |
| With spatial features | 0.6228 | 0.2176 | **0.689** |

Spacing does move the *inputs* the model cares about — the widest-spacing
quintile of shots has 4.62 ft of defender separation against 4.28 ft in the
tightest and a shot-quality index of 45.7 against 43.1.

### Basketball reading

Spacing works the way the coaching cliché says it does, and then stops. It
produces more separation and better shot quality — that shows up clearly. It
does not add information about whether a *given* shot goes in, because by the
time you know the shot's location and the defender's distance, the spacing has
already done its work through those two variables. Spacing is upstream, not a
separate effect.

The turnover column complicates the picture: the widest quintile turns it over
14.5 % of the time against 9.8 % in the tightest. Wider spacing means longer
passes. Whatever spacing gains in shot quality, part of it is paid back in
possessions lost, which is the most plausible reason the points-per-chance
gradient is so weak.

### Limitation

1,910 chances with spacing samples. The Q5 − Q1 confidence interval spans zero,
so the apparent gradient is not established. Hull area over five *tracked*
players is sensitive to broadcast dropouts, and the sample is one team-half of
each of ten games.

---

## 8. Shots with no tracked action attached are the best shots in the dataset

### Finding

Grouping every shot by the action that created it, the largest single group —
**383 shots with no tracked action marking in the preceding 4 seconds** —
converts at **48.6 %** with a shot-quality index of **50.2**, the best of any
creation type. Isolation is the worst at **28.1 %** on 32 shots.

### Evidence

| Creation type | n | FG% | model xFG | shot quality | pts/shot | 3PA rate | open rate | def. distance |
|---|---|---|---|---|---|---|---|---|
| **Unstructured** | 383 | **0.486** | 0.534 | **50.2** | 1.034 | 20.9 % | 34.1 % | 4.34 ft |
| Post | 37 | 0.432 | 0.448 | 44.1 | 0.946 | 16.2 % | 25.0 % | 2.88 ft |
| Drive | 424 | 0.410 | 0.472 | 41.1 | 0.868 | 11.1 % | 21.9 % | 3.51 ft |
| Pick | 319 | 0.404 | 0.429 | 42.2 | 1.006 | 61.4 % | 28.4 % | 4.84 ft |
| Closeout | 134 | 0.358 | 0.377 | 39.6 | 0.992 | 78.4 % | 32.8 % | 6.28 ft |
| Handoff | 54 | 0.352 | 0.369 | 37.7 | 0.926 | 68.5 % | 28.3 % | 4.79 ft |
| Off-ball screen | 76 | 0.329 | 0.375 | 38.5 | 0.829 | 57.9 % | 21.3 % | 4.68 ft |
| **Isolation** | 32 | **0.281** | 0.294 | 33.2 | 0.750 | 62.5 % | 6.7 % | 3.94 ft |

Most efficient action chains, chances with n ≥ 10:

| Action chain | chances | pts/chance | shot rate | turnover rate |
|---|---|---|---|---|
| `drive:shotNearBasket` | 15 | 1.533 | 73.3 % | 0 % |
| `pick:over > pick:over > closeout` | 10 | 1.100 | 100 % | 0 % |
| `pick:over > drive:shotNearBasket` | 17 | 1.059 | 76.5 % | 0 % |
| `pick:switch` | 17 | 1.000 | 64.7 % | 17.6 % |
| `pick:over` | 94 | 0.947 | 48.9 % | 12.8 % |
| `closeout` | 54 | 0.926 | 87.0 % | 7.4 % |

### Basketball reading

"Unstructured" is not a compliment to freelancing — it is a definition artefact
worth understanding. These are shots where SkillCorner's Game Intelligence
models detected no pick, drive, handoff, off-ball screen, isolation, post or
closeout in the previous 4 seconds. In practice that is largely **transition,
putbacks and immediate offensive-rebound finishes**: 20.9 % three-point rate
against 61.4 % for pick-created shots, the shortest average shot in the group.
The lesson is about the taxonomy, not about tactics: a shot-creation study built
only on named actions silently discards the most efficient third of a team's
offence.

The isolation row is the one that survives as a basketball conclusion. Iso
shots are the *least open in the dataset* — 6.7 % open rate against a 28 %
average — and convert 12 points worse than the field. Isolation in the ACB is
not a shot-generation tool; it is what happens when nothing else did.

The chain table adds a second-order point: **doubling the pick works.**
`pick:over > pick:over > closeout` produced a shot on 100 % of its ten chances
with zero turnovers, and 90 % of those were threes. A second screen against a
defence already playing over the first one is how the corner opens.

### Limitation

The 4-second lookback for the creation label is a parameter, not a fact; a
longer window would reclassify some "unstructured" shots. Chain groups have 10
to 94 chances — enough to notice a pattern, nowhere near enough to prescribe
one. Action detection is a model output with its own error rate.

---

## 9. Broadcast tracking is better than its reputation, and already smoothed

### Finding

Computed independently from the raw frames, this dataset reproduces every
quality figure SkillCorner publishes: **76.2–82.8 % of positions directly
detected**, mean prediction error **1.52–1.86 ft**, 95th percentile
**4.88–6.28 ft**, extrapolated positions **4.36–5.15 ft**. Separately, the
delivered positions turn out to be **already temporally smooth**: central
differencing them raw reproduces SkillCorner's own `speed` field at **r = 0.999**.

### Evidence

| | claimed (README) | measured here |
|---|---|---|
| Positions detected | 76–83 % | 76.2–82.8 % |
| Mean predError | 1.5–1.9 ft | 1.52–1.86 ft |
| 95th pct predError | 4.9–6.3 ft | 4.88–6.28 ft |
| Extrapolated predError | 4.4–5.2 ft | 4.36–5.15 ft |
| Detected predError | ~0.9 ft | 0.84–0.91 ft |

Raw versus smoothed kinematics, ten games:

| | raw | Savitzky–Golay (15 frames, order 2) |
|---|---|---|
| mean speed | 5.95 ft/s | 5.94 ft/s |
| 99th percentile | 18.5 ft/s | 18.5 ft/s |
| **maximum** | **98.1 ft/s** | **48.0 ft/s** |
| correlation with delivered `speed` | 0.999 | 0.999 |

Two further properties, neither documented in the source repository:

* **The video frame index is continuous within a period and jumps at every
  period boundary** — gaps of 2 to 2,075 frames across the ten games. Any
  pipeline that assumes a single continuous frame timeline will mis-align
  everything after the first quarter.
* **7 orphaned `touchId` references** out of 24,487 (0.029 %): 1 drive, 5
  passes, 1 dribble point at a touch that does not exist.

### Basketball reading

Not a basketball finding, an engineering one, and it is the finding with the
most practical value. Broadcast tracking at 1.5 ft mean error is entirely
adequate for the spatial questions in this project — spacing measured in tens of
feet, defender distance measured in single feet, help arrival measured in
seconds. It is *not* adequate for anything requiring sub-foot precision, and the
17–24 % extrapolated positions carry 3–5× the error of the detected ones, which
is why every spatial metric here degrades to NaN rather than guessing when
players are missing.

The smoothing result saves work: there is no hidden signal to recover from these
positions. A heavier filter would cost compute and change nothing except the
tail.

### Limitation

Ten games from one league and one vendor. Detection rate varies by broadcast —
the worst game here (76.2 %) has 8 % fewer detected positions than the best.

---

## 10. Offensive roles in the ACB are a continuum, not a set of archetypes

### Finding

Clustering 233 qualified player-team seasons in a 9-component PCA style space
(87.6 % of variance), the silhouette score **peaks at k = 3 with 0.266** and
never exceeds 0.27 anywhere between k = 3 and k = 9. There is no k at which
offensive roles separate cleanly.

### Evidence

| k | inertia | silhouette |
|---|---|---|
| **3** | 2,440 | **0.266** |
| 4 | 2,241 | 0.239 |
| 5 | 2,112 | 0.163 |
| 6 | 1,991 | 0.170 |
| 7 | 1,879 | 0.188 |
| 8 | 1,812 | 0.163 |
| 9 | 1,720 | 0.167 |

The k = 3 partition, described from standardised feature means (labels written
after the fact, not imposed):

| Cluster | n | Highest features | Lowest features |
|---|---|---|---|
| 0 — interior finisher / screener | 65 | screener picks per game (+1.30), rim rate (+1.26), FT rate (+0.87) | shot distance (−1.28), three rate (−1.25) |
| 1 — movement and spot-up shooter | 74 | corner-three share (+1.03), three rate (+0.76), assisted share (+0.72) | drives per game (−0.60), FT rate (−0.59) |
| 2 — on-ball creator | 94 | drives per game (+1.00), handler role ratio (+0.99), shot volume (+0.69) | catch-and-shoot share (−0.98), assisted share (−0.98) |

Representative players nearest each centroid: Juan Fernandez, Youssoupha Fall,
Yankuba Sima (cluster 0); Josep Puerto, Alberto Abalde, Gytis Radzevicius
(cluster 1); Kassius Robertson, Darrun Hilliard, Chris Duarte (cluster 2).

### Basketball reading

The three clusters that do emerge are the oldest distinction in the sport — who
finishes inside, who spaces, who creates — and they are recovered from nothing
but shot mix, drive behaviour and pick role, which is a decent validation that
the feature space is measuring something real.

What the silhouette curve says is that everything *finer* than that is
interpolation. There is no natural break separating a "movement shooter" from a
"secondary creator" in this league; players sit on a smooth surface between the
three poles. Any system that assigns ACB players to six or eight crisp
archetypes is imposing a structure the data does not contain — which is exactly
why the app ships the k = 3 partition as the supported result and labels the
k = 6 view as exploratory.

The practical consequence for scouting: **compare players by position on the
continuum, not by cluster membership**. Two players in the same cluster can be
far apart, and the app's similarity view — cosine distance with the three
features that drive each match — is the more honest tool.

### Limitation

Offence only, because the season aggregates contain nothing else. 233
player-team seasons with ≥ 8 games and ≥ 40 attempts. Style, not quality:
nothing in this model says who is good.
