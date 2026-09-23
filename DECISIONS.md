# Decisions

Why revdrive is the way it is. Each entry is something that was measured, not
argued. The model is Qwen3.8-27B (AWQ INT4), served by sglang on two L40S and
read through [rev](https://github.com/54yyyu/rev)'s remote engine; one-token
answers, their log-probabilities read exactly. Offline numbers fit on frames of
courses 0-2 and are checked on frames of courses 10-12. Closed-loop numbers are
one run per course unless stated; the benchmark in `results/` is the record.

## The benchmark, in one paragraph

rev on Qwen3.8-27B, strict, the windshield camera, a cone every 2.5 m, a 1.5 m/s ceiling, one run per course (`results/`, 2026-09-23): 21.1% of the course on average before the first cone touched, none finished; best 77.8% (course 0), and seven of eleven runs ended within the first 14%. Its bearing was 5.9 deg off on average over the decisions it made. The same harness given the true bearing and charged rev's 1 s latency (oracle-point) finished all eleven with no cone touched; rev shown grey pictures (blind) got 0.8%. On course 100, whose first 90 deg corner comes right after the start, rev's 6.8% is blind's 6.4%.

## The model says where the course is; a controller drives there

The harness is judged by an oracle through the same interface - the true
bearing of the course, the same controller, the same latency - before the model
is judged at all. If the oracle fails, the harness is the constraint.

- Holding a wheel angle between answers fails even for the oracle: pure pursuit
  on the true line leaves the course at 0.35 s of latency, because each answer
  is stale by the time it lands.
- Answering with a point on the ground instead works. The point is fixed in the
  world at the pose the picture was taken from, so a late answer is still about
  the right place; it sits further out by latency x speed so it is still ahead
  when it lands; successive answers are averaged (25% of the old target kept).
  The oracle then finishes courses 0-5 at 0.3, 0.5 and 0.8 s of latency, and
  still does with 4 deg of random error on every answer (8 deg: 83% at 0.5 s).
- Strict, at the model's 1 s latency and 1.5 m/s, the oracle finishes every
  course with no cone touched. With 6 deg of noise it finishes the 8 m lane of
  course 100 but not the 6 m lanes of courses 0-9 (1.9 m of room either side of
  a 1.8 m car, against 2.9). So the generated courses ask for a bearing within
  about 5 deg; that, not the harness, is what is measured.

## What the model is asked

A one-token reader cannot give a number, and the form of the question decides
what can be read out. Same frames throughout (oracle drives, labelled with the
true centreline):

- "How should you turn the steering wheel?" with seven levels ran *against* the
  oracle (r = -0.2 to -0.85): the model steers away from the cones in front of
  it, which in a bend is the inside line. Any seven- or eleven-way pick (bend
  levels, lettered marks painted on the ground) is dominated by the first option
  whenever unsure: lettered marks were 14% exact.
- Telling it where the wheel already is leaks the answer: r = 0.87 with
  "Steering wheel: turned left" in the state, 0.46 without.
- Digits on a scale (which tenth of the picture's width, read as the
  distribution over "0"-"9") track the course but shrink toward the middle
  column: 0.06-0.19 of the true angle closed loop. Qwen's own pointing
  (`{"point_2d": [x, y]}`, generated) erred 6.2 deg; digits with a fitted gain 5.9.
- Two-way questions keep what the model sees. "Is the middle of the lane about
  d m ahead left or right of the middle of the picture?", asked of five views
  turned -30..+30 deg, each with its mirror image, and read through a linear
  head fitted on wobbly drives: 3.9 deg mean error (median 3.2, 84% within 6),
  against 7.3 for "straight ahead". This is `rev`. The mirror cancels any lean
  toward a side or a word (one reading alone leaned "left" by +0.49 of full
  scale); the five views and their mirrors are ten calls in parallel, about
  1.05 s.
- A yes/no per direction ("would driving straight at the middle of this view
  stay between the cones?") got "no" nearly everywhere (P(yes) 0.08-0.13): 4.9
  deg at best.
- Nonlinear heads read no better than the linear one: the answers stay near
  50/50 (mean |P(left) - P(right)| about 0.1).

## Inputs, aligned with DrivingBench

DrivingBench gives its agents a narrow and a wide camera at the top middle of
the windshield, speed, the measured wheel, and a long task description. For a
one-token reader, less is read better. Same 270 wobbly frames:

- From the driver's seat at 90 deg with a short prompt: 4.5 deg. Their narrow
  camera (52 deg): 5.0 with dense cones. Their mount with 90 deg views cut from
  their wide camera, dense cones: 3.9 - the default (`--camera windshield`). The
  narrow view was the loss, not the mount: a bend leaves it sooner.
- Their full task description: +0.3 deg. The narrow and wide images alongside
  the view asked about: +1.1 deg. The measured wheel: no help (the wheel alone
  predicts 6.3 deg). Extra pictures dilute what the model reads from the one the
  question is about. The information is there to be asked for; it is not all
  shown at once.
- The picture from 1 s earlier next to the current one: 5.8 deg against 4.5.
- Their cone density (they publish none; about 2.5 m on their map, against 9 m
  on straights and 4.5 m in bends before; `--cone-spacing 2.5`) helps a little:
  sharp-bend error 10.9 against 12.1 with their narrow camera. The median gap
  measures 2.7 m because cones are placed on a 0.5 m grid along the line.
- Their prompt's advice for sharp turns ("don't be afraid to do sharp turns",
  "start your turn earlier than you expect") in our question made answers twice
  as decisive and read better offline (4.2 deg against 4.5) but drove no better
  (strict 46/42/6/8% of courses 0/1/2/100 against 46/41/7/10) and 0.15 s slower.

## Speed and rules

- `--strict` (the default) is DrivingBench's rule as their operator applies it:
  touching any cone, or the car's centre crossing the cone line, ends the run.
  `--lenient` counts cones and allows 4 m off the centreline.
- The model picks its speed as a digit (0 = stop, 9 = the ceiling). Under a
  3.5 m/s ceiling it chose about 2.8 m/s almost everywhere and did not slow for
  bends; its speed answers are bimodal (at one decision, 0.33 on "0" and 0.61 on
  "9"). A 1.5 m/s ceiling - DrivingBench's cars drove 0.8-1.6 m/s - drove rev
  furthest: strict 78/45/7/6% of courses 0/1/2/100 against 30/41/7/8%. Slower
  gives more answers per metre, which corrects small misreadings before they
  reach a cone; it does not help where the model cannot see the bend at all.

## What did not help

- Memory of the model's own answers: the previous decision's bearing in the
  readout (read better offline, drove worse: 47/46/13/0% against 85/46/38/9%,
  lenient); a path fitted through the last 4 s of answered points (78/42/6/6%
  against 78/45/7/6%, strict); the same weighted by confidence (83/12/6/6%:
  confident answers are also wrong a quarter of the time, and memory locks them
  in). The code was removed.
- Aiming 4 m further: calmed the wheel with simulated noise (88% of runs
  finished against 62%) but drove rev worse (28/5/6/8%); a point further out
  sits at a bigger angle in a bend, where the model reads worst.
- Confidence as a gate: below 0.2 the answers are off by 10-16 deg on average
  and wrong by more than 6 deg over half the time, but a good answer is more
  confident than a bad one only 60% of the time.

Offline error on an oracle's frames is necessary, not sufficient: every change
here that read better offline was driven before it was kept.

## Why sharp corners fail

- Image input works. Through the same read path, trivial pictures are answered
  with certainty: a colour (1.000/0.000), an arrow's direction (1.000/0.000),
  counting 1-5 squares (all right, p 0.97-0.98).
- Left or right of the middle is coarse: a dot 56 px left of centre in a 512 px
  picture (about 10 deg in a 90 deg view) is called "right" (P(left) 0.27).
- At a failed 90 deg corner the model says "left" with P about 0.73 for every
  view and every mirror image alike, whatever the wording. Asked in chat about
  the straight-ahead view 56 deg into a left corner, it answers "the course
  proceeds straight ahead, following the path defined by the two parallel lines
  of orange cones". Hiding every cone further than 20 m changes nothing. Seen
  from inside a tight corner, its outer line runs across the view and reads as
  one side of a straight lane; one picture does not show that the lane turns.

## Known gaps

- A simulator: flat asphalt, low-poly cones, no curbs, painted stalls or walls.
  A real lot has more cues that a corner is a corner.
- Course 100 follows DrivingBench's published description (a left entry turn,
  a 47 m aisle with gentle bends, right turns, 27 m and 18 m aisles, 8 m wide,
  about 130 m); their turn radii are not published and the 8-9 m used here are
  guesses. Their map suggests one large U-turn instead.
- A 90 deg view turned 30 deg reaches 75 deg to the side, 15 deg past a 120 deg
  wide camera.
- One run per course. Run-to-run spread was not measured; the model's answers
  are deterministic, but the car's path depends on answer timing.
- The fields of view of the comma cameras are approximations (52 and 120 deg,
  rectilinear here; the wide one is a fisheye there).
