# Wuji Hand2 three-clip benchmark

`selection/selection.lock` freezes the three native frame windows and binds
source, MANO, object, robot, qpos-order, collision, and solver hashes. A
result cannot change selection or replace a failed frame.

Each clip records canonical input, warm-start, source-only interaction graph,
warm interaction evaluation, per-frame checkpoint chain, final artifact,
independent full-surface validation, metrics, export, and self-contained HTML.
The independent validator queries every formal robot collision sample with a
fresh strict winding backend; persisted solver SDF arrays are not sufficient
evidence.
