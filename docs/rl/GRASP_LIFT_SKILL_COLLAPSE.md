# Stage16 Grasp-Lift Skill Collapse

The frozen localization establishes a one-update U25-to-U26 loss of persistent
grasp and lift. Its follow-up policy-preservation shadow search repaired the
actor-only/critic-baseline harness and selected a 0.50x actor-LR candidate.
The candidate preserves 10/10 frame-0 grasp/lift and U25 GRASP-reset lift,
while its U25 CONTACT reset remains a documented limitation.

The result is a controlled shadow candidate only. The next action is continuous
C0 live verification; it is not authorization for a formal PPO default switch.
