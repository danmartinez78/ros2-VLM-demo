# ODD observation system architecture

Status: **target design**  
Tracking: #75 and follow-on estimator/monitor work

## Purpose

The ODD observation system answers:

> What relevant operating-domain conditions are currently present, how confident are we, and are any configured ODD limits being approached or exceeded?

The design deliberately separates **estimating the world** from **deciding whether the configured ODD has been exited**.

The first intended use is observation, logging, evaluation, and architecture research. It is **not** a safety-certified minimum-risk-maneuver path and should not be described as one.

## Architectural principle

Use the best available estimator for each ODD axis or coupled set of axes, normalize their outputs, and feed a deterministic monitor.

```text
sensor/state inputs
      |
      +--> weather estimator ---------+
      +--> semantic-scene estimator --+
      +--> dynamic-actor estimator ---+----> normalized ODD estimates
      +--> terrain/road estimator -----+              |
      +--> pose/speed estimator -------+              |
                                                     v
                                          coupled constraint nodes
                                                     |
                                                     v
                                          deterministic exit monitor
                                                     |
                                                     v
                                      observation / logging / evaluation
```

The architecture is intentionally **not** "one giant VLM decides whether we are inside the ODD."

## Core concepts

### ODD axis

A measurable or inferable dimension relevant to the operational design domain.

Examples:

- precipitation intensity;
- visibility / fog;
- surface condition;
- terrain/road class;
- construction-zone presence;
- actor density or actor class constraints;
- vehicle speed;
- position relative to a geofence;
- slope or curvature;
- illumination or other environmental conditions.

An axis may be directly measured, derived from state, or inferred from perception.

### Estimator

A component that produces an estimate for one axis or a coherent group of axes.

A normalized estimator output should carry enough information for downstream reasoning and audit:

```text
axis / estimate
units or categorical vocabulary
confidence or quality
source timestamp
freshness / age
evidence or supporting observations
estimator identity/version
optional uncertainty bounds
```

The exact ROS message/interface can evolve, but the semantic contract should remain stable.

### Constraint node

A deterministic component that evaluates a limit involving **interdependent axes**.

Use a constraint node when the allowed value of one axis depends on another.

Example:

```text
rain intensity ----+
                   +--> speed-vs-rain constraint --> status
vehicle speed -----+
```

A constraint node can implement a table, curve, piecewise rule, envelope, or other explicit function.

It should not be used merely as a generic name for every comparison.

### Exit monitor

The deterministic aggregation layer that evaluates independent axis limits and constraint-node outputs and produces the overall ODD observation state.

The monitor should know configured limits and policies. Perception estimators should not each independently encode a duplicate copy of the whole ODD definition.

## Independent versus coupled axes

### Independent axis

If the ODD limit for an axis can be evaluated without another changing variable, the estimate can feed the exit monitor directly.

Example:

```text
pose
  |
  v
geofence estimator
  |
  v
inside/outside + distance-to-boundary
  |
  v
exit monitor
```

The geofence estimator is an estimator, not a "constraint node" simply because it compares position to geometry.

### Coupled axes

If validity depends on a combination of axes, route them through an explicit constraint.

Example:

```text
camera --> rain estimator --------+
                                  +--> speed/rain constraint --> exit monitor
pose ---> speed estimator --------+
```

This keeps interdependence visible and testable.

Other examples could include:

- speed versus road curvature;
- speed versus visibility;
- slope versus payload/configuration;
- wind versus vehicle/mission state.

## Method selection by axis

The architecture permits heterogeneous perception methods. Model uniformity is not a goal.

### Deterministic state / geometry

Prefer deterministic computation when the required state is directly available.

Examples:

- pose -> geofence relation;
- odometry -> speed;
- map geometry -> known road class or slope where map quality is sufficient.

Advantages:

- explicit semantics;
- predictable failure modes;
- straightforward unit testing;
- low compute cost.

### Detector / tracker

Prefer detector/tracker pipelines for well-defined dynamic entities when localization, counts, tracks, or motion estimates are required.

Examples:

- people;
- vehicles;
- tracked obstacle trajectories;
- actor density.

A tracker can provide structured temporal state that a VLM can optionally consume as context, but the tracker output remains independently usable.

### CNN / ViT-style discriminative estimator

A specialized visual model can be appropriate for a narrow, repeatable axis with sufficient training data.

Examples might include:

- rain-intensity classification;
- visibility category;
- surface-condition classification.

Advantages can include lower latency and more controlled output spaces. The principal risk is domain shift outside the training distribution, which must be measured rather than assumed away.

### VLM

A VLM is especially useful for semantic axes where the relevant visual concept set is broad, compositional, or difficult to enumerate exhaustively.

Candidate examples:

- construction-zone context;
- unusual roadway/environment context;
- semantic weather/environment observations;
- combinations of objects, signage, and scene state;
- temporal semantic changes over a short visual window.

VLM output should still be normalized into an estimator contract rather than passed downstream as unconstrained prose.

## Why parallel pipelines are intentional

The target architecture is:

```text
axis A -> method best suited to A --+
axis B -> method best suited to B --+
axis C -> coupled method/constraint -+--> deterministic monitor
axis D -> deterministic state -------+
```

not:

```text
all sensors -> one model -> final ODD decision
```

Benefits:

- each axis can be evaluated with task-specific metrics;
- models can be swapped without redesigning the monitor;
- deterministic state remains deterministic;
- coupled rules stay explicit;
- failures are easier to triage;
- onboarding and ownership boundaries are clearer;
- compute-heavy methods can be used only where justified.

## Estimator output contract

A useful logical shape is:

```json
{
  "axis": "precipitation",
  "estimate": "moderate_rain",
  "value": null,
  "units": null,
  "confidence": 0.86,
  "timestamp_s": 123.45,
  "age_ms": 80,
  "evidence": {
    "summary": "persistent visible rain streaks and wet roadway",
    "source": "front_camera_temporal_window"
  },
  "estimator": {
    "name": "weather_vlm",
    "version": "..."
  }
}
```

For continuous quantities, `estimate` may instead be numeric with units and uncertainty.

The monitor should not need to parse prose to determine whether a limit is exceeded.

## Confidence and evidence

Confidence is not a substitute for a calibrated probability unless the estimator is explicitly calibrated that way.

The architecture should preserve:

- the estimate itself;
- model-reported confidence if available;
- independent quality/freshness flags;
- evidence/provenance sufficient to inspect why the estimate was produced.

Downstream policies may later decide how to handle low-confidence estimates, stale data, conflicting estimators, or missing axes. Those policies should be explicit rather than hidden inside the estimator prompt.

## Temporal semantic estimators

Some axes are inherently temporal. For those, use the temporal VLM architecture in [`temporal-vlm-architecture.md`](temporal-vlm-architecture.md).

Examples:

- precipitation increasing/decreasing;
- visibility degrading;
- vehicle approaching/receding;
- road changing from clear to blocked;
- entering/leaving a construction zone.

A temporal estimator should receive a bounded recent window and produce a compact structured assessment such as:

```json
{
  "change_detected": true,
  "change": "visibility_degrading",
  "state_start": "clear",
  "state_end": "moderate_fog",
  "evidence_start_s": 0.5,
  "evidence_end_s": 1.0,
  "confidence": 0.84
}
```

The normalized ODD-axis estimator may then map this temporal assessment into the current axis state plus trend.

## Deterministic exit monitor

The exit monitor is deliberately simpler than the perception stack.

Conceptually:

```text
for each independent axis:
    read current estimate
    evaluate configured limit

for each coupled constraint:
    read constraint status

combine statuses deterministically
emit:
    in_odd / boundary / out_of_odd / unknown
    active violations
    near-boundary conditions
    stale/unknown inputs
    provenance references
```

The exact state vocabulary may evolve, but the monitor should remain deterministic for a fixed set of inputs and configuration.

### Boundary state

A distinct boundary/near-boundary state is useful for evaluation and logging even when no reactive action is attached.

It allows experiments to distinguish:

- comfortably inside the configured domain;
- near a limit;
- clearly outside a limit;
- unknown because required evidence is unavailable or stale.

## Missing, stale, and conflicting observations

A production-quality observation system cannot treat absence of evidence as evidence of being inside the ODD.

The monitor should distinguish at least:

```text
valid estimate
stale estimate
missing estimate
invalid estimate / estimator failure
conflicting evidence (if multiple estimators are fused)
```

How an `unknown` axis affects overall ODD status is a policy decision and should be configuration-driven.

For the current research/logging scope, retaining the unknown state is preferable to fabricating certainty.

## Logging-only scope

The initial system is intended to:

- flag potential ODD exits;
- record evidence and timing;
- support offline evaluation;
- compare estimator architectures;
- identify boundary cases and failure modes.

It does **not** currently:

- command braking or steering;
- trigger a minimum-risk maneuver;
- claim ASIL/SIL certification;
- replace an independently engineered safety arbiter;
- make a VLM directly responsible for safety actuation.

This reduced operational consequence allows broader experimentation, but it does not remove the need for rigorous measurement if results are later used to justify a safety-relevant design.

## Temporal memory experiments

Short native-video context is one temporal mechanism, not the only one.

The research branches under consideration are:

1. **Single-frame VLM baseline** — no temporal memory.
2. **Recurrent text world-state** — prior compact observations are fed back into the next request.
3. **Detector/tracker upstream** — structured object/track state augments visual reasoning.
4. **Learned sequence memory** — e.g. a Mamba-style memory layer between visual encoding and language reasoning.
5. **Knowledge-graph memory** — structured model output updates a graph; deterministic retrieval injects relevant state into later prompts.

These should be evaluated as explicit experimental variants, not silently combined.

A likely future architecture is layered:

```text
short-horizon native visual window
        +
structured longer-horizon state/memory
        ->
current semantic assessment
```

## Evaluation strategy

Measure estimators before measuring the aggregate monitor.

### Per-axis metrics

Examples:

- classification accuracy / F1;
- regression error;
- trend/direction accuracy;
- event-time accuracy;
- false-positive rate on known-clear/static cases;
- calibration/uncertainty quality;
- performance by domain slice and out-of-distribution slice;
- latency and resource use.

### Constraint metrics

For coupled axes:

- correctness around the constraint boundary;
- monotonicity where the policy requires it;
- interpolation behavior between configured points;
- missing/stale input handling.

### Monitor metrics

- overall exit-detection precision/recall;
- boundary-state correctness;
- time from real condition change to observed exit;
- unknown/stale-state handling;
- traceability from overall result back to source estimates.

## Implementation guidance

Start with separate nodes/components for major processing functions and collapse only when profiling demonstrates a reason.

For example:

```text
camera -> rain estimator --------+
                                 +-> speed/rain constraint -> exit monitor
pose -> speed estimator ---------+
pose -> geofence estimator -------------------------------> exit monitor
camera -> construction-zone VLM --------------------------> exit monitor
```

This separation keeps contracts testable and permits later composition into a monolithic deployment process without forcing monolithic code structure at the beginning.

## Non-goals

This document does not prescribe:

- a final list of ODD axes for any specific autonomous product;
- certification strategy;
- a final safety arbiter architecture;
- one universal perception model;
- a specific ROS graph for all deployments;
- automatic safety action from VLM output.

It defines the research/system decomposition used by this repository so experiments can be compared against a stable architectural intent.
