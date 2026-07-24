# Eq. (9) Temporal-Scope Interpretations

The frozen benchmark compares two already-versioned profiles:

| Profile | Interpretation | Fidelity status |
| --- | --- | --- |
| `scipy_slsqp_active_set_contact_rich_v2` | `literal_full_state_temporal`: temporal regularization covers the full optimizer state, including base correction and finger q | paper-consistent; `author_exact` unresolved; historical retained |
| `scipy_slsqp_active_set_contact_rich_v3_fixed` | `decomposed_finger_temporal_plus_base_priors`: temporal regularization covers finger q while base translation/rotation use independent priors | paper-consistent; `author_exact` unresolved; validated quality-neutral engineering profile |

The paper writes optimizer coordinates in the first Eq. (9) term and does not publish author code;
both readings remain defensible. Independent base priors do not by themselves prove that base
temporal smoothness is absent. Q1–Q3 does not scan weights, tune a sequence, add a profile, project,
or run a new temporal ablation.

Only dynamic GRAB units determine empirical engineering preference. The rule requires at least two
of three dynamic clips to improve principal interaction metrics, no significant (>10%) clip
regression, no constraint failure, and no >20% continuity deterioration, using identical frozen
parameters. Otherwise both interpretations remain. No outcome can establish an author-exact reading.
