# Stage 1 — TopoRetarget paper audit

## Audit scope

The complete 16-page arXiv v2 PDF was extracted and read, including the abstract, Sections 1–7,
Equations 1–12, Figures 1–5, Tables 1–6, and Appendices A.1–A.5. The local source and hash are
recorded in `docs/PAPER_FIDELITY.yaml`.

## Coverage

All equations, tables, figures, required sections, datasets, baselines, fixed-parameter claims,
augmentation claims, and the virtual-contact limitation have manifest entries. Tables 1–6 are
transcribed row by row in `docs/reproduction/`. Paper-provided values are mapped to the paper
configuration files with section/page/table provenance.

## Unpublished details and blockers

Missing solver, frame, geometry, metric, simulator, low-level control, private dataset, and Wuji
asset details are registered with unique IDs in `docs/ASSUMPTIONS.md` and converted into author
questions. Nulls in config represent `not_provided`; they are not inferred defaults.

## Strict versus extended boundary

The strict boundary contains only what is stated in the paper. Any future MANO cleanup, alternate
solver, new dataset adapter, or SPIDER-related work must be labeled as an extension rather than
silently merged into the strict method.

## Definition of done

Stage 1 is complete when the PDF hash, required manifest IDs, assumptions, parameter contracts,
and unit tests pass. It is not method-complete or result-complete: numerical implementation and
the original data/hardware experiments remain future work.

