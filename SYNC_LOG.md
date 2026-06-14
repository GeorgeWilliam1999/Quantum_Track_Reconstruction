# SYNC_LOG — 2026-06-14 cross-chat reconciliation

Coordinator pass to merge the parallel chats' work into one coherent state without
losing anything. Canonical base (not re-litigated): **LVT @ `17db26f`** (OneBQF =
canonical 1-bit HHL; `segment_metrics.py`; `fast.py` exact cKDTree step build);
1BQF runs via the matrix-free engine; metrics are a recomputed VIEW at absolute τ=0.35.

---

## 1. Inventory (as found, re-verified)

**LVT (`LHCb_VeLo_Toy_Model`)** — clean, `main` @ `17db26f`, pushed = origin/main. No
other branches, worktrees, or stashes. The canonical cKDTree A-build is present
(`construct_hamiltonian` → `cKDTree(...).query_ball_point`). **Left untouched.**

**QTR (`Quantum_Track_Reconstruction`)** —
- Branches: `main`, `erf-study-task5` (now retired), `sync-merge-2026-06-14` (this pass).
- No extra worktrees; no stashes.
- During this session `main` **advanced** d…→`d6fb474d` and was **pushed to origin** by a
  parallel chat (the Epsilon_study_2 §7.12 *indefiniteness* notebook/figures/draft,
  commits `3243dfe2`/`0b5bb231`/`d6fb474d`). That work is therefore already in main and
  inherited by this branch — nothing to re-commit.
- No sibling `*Quantum*` checkouts. Other repos under `/data/bfys/gscriven` (Allen,
  Velo_toy, TrackExtrapolation, …) are unrelated and untouched.

## 2. `erf-study-task5` — RETIRED
`git merge-base --is-ancestor erf-study-task5 main` = YES (tip `0707e3f7` fully contained
in main; `git log main..erf-study-task5` empty). Deleted with `git branch -d`. **No work lost.**

## 3. What was merged — `sync-merge-2026-06-14` (5 commits on top of `main` d6fb474d)

| commit | group | contents |
|---|---|---|
| `86493be0` | infra | matrix-free 1BQF engine `obqf_matrixfree.py` (new) + `helpers.solve_quantum_statevector` engine dispatch (default `matrixfree`, env `QTRK_OBQF_ENGINE`, key-safe) + `build_submission` flat 16 GB tiers / chunks 50–200 + `SCALING_DEEP_DIVE.md` + `_profile_*`/`_probe_*`/`_validate_*` evidence scripts |
| `a72bf64f` | guides | `DATA_GENERATION_GUIDE.md` (new) + 5 per-study `DATA_GUIDE.md` + `DATA_INDEX.md` refresh |
| `f9b4a1b4` | Verify | `Verify_new_results/topup_1bqf_highT.py` (key-safe, idempotent 1BQF high-T topup mirroring qsvt coverage) |
| `19d538d4` | docs | `QSVT/README.md` infra note + `QSVT/AUDIT_2026-06-12.md` resolution block; **removed** the two superseded trackers (see §5) |
| `a3be3396` | docs | reconcile A-build speedup figure to Notion (~25×); 8.5× kept as the §5.2 micro-benchmark floor |

The §7.12 *indefiniteness* notebook + 7 figures + companion draft were **already committed
to main** by the parallel chat (`3243dfe2`), so the planned "indefiniteness" commit here was
correctly a no-op — the work is present and intact.

## 4. Deletions in the working tree — resolved

- **`Epsilon_study_2/outputs/notion_section7{,_10,_11,_12}_draft.md`** — appeared deleted but were
  **restored** (`git checkout HEAD --`). They were committed *today* (`7e8adc3e`), are referenced by
  the `project-epsilon-study2` memory ("Draft body kept at outputs/…"), and the new
  `notion_indefiniteness_writeup_draft.md` sits beside them — the deletion looked accidental, so per
  "lose nothing" they were kept.
- **`Toy_Characterisation/AUDIT.md`** — deletion **accepted**. PROJECT_STATUS.md explicitly records it
  as intentionally removed 2026-06-14 (superseded by PROJECT_STATUS + Notion).
- **`Epsilon_study_2/project_epsilon_study2.md`** — deletion **accepted** (flagged): it is a *stale
  duplicate* (21 lines) of the canonical memory note `.claude/memory/project_epsilon_study2.md`
  (a strict superset carrying §7–§7.12). Per CLAUDE.md, project notes live in memory. **Restorable**
  from history if you'd rather keep an in-repo copy.

## 5. Conflicts / inconsistencies reconciled

1. **A-build speedup: 8.5× vs 25×.** Both are *measured* but against different baselines:
   §A.1's full-pipeline original O(T³) build = **~100 s @T1000** (→ ~25× vs the ~4 s fast build),
   while §5.2's `_validate_fastbuild` reference build was an already-grouped **35.5 s** (→ 8.5×).
   Notion (Data Coverage, source of truth) + 4 in-repo docs say **~25×**. Per "Notion wins on numbers,"
   standardised the **headline to ~25×** (SCALING §7, DATA_INDEX, DATA_GENERATION_GUIDE, QSVT/README),
   keeping §5.2's **8.5×** as the explicitly-labelled conservative micro-benchmark floor. Exactness
   (`max|ΔA|=0`) is the guaranteed property regardless of multiplier. Memory updated to match.
2. **SCALING_DEEP_DIVE §7 "not committed / coordinate before committing LVT".** Stale — the cKDTree
   A-build consolidated into the canonical **`17db26f`** (pushed). §7 updated to "committed".
3. **"1BQF can't scale past T=700".** Already reconciled in **Notion** (Data Coverage explicitly:
   the wall was the Aer host-RAM OOM, not the statevector; matrix-free fixes it). Repo docs + memory
   now tell the same single story. No Notion edit was needed.
4. **`project-scaling-deep-dive` memory** updated: "ACTIONED but NOT committed" → committed; 25×
   headline; "P0 subprocess isolation" remains SUPERSEDED by the matrix-free engine.

## 6. Verification (all green, 2026-06-14)

- **LVT `test_pipeline.py`** → ALL PIPELINE TESTS PASSED.
- **`qtrk_pipeline/_selftest.py`** → ALL CHECKS PASSED, **`max|dA|=0.00e+00`**; quantum
  `P_anc=0.0224` (== the Aer value, confirming the matrix-free *production* engine).
- **A-build bit-identity** (`_validate_p3_inplace.py`): `max|dA|=0`, identical nnz at T=200/400/700/1000,
  step + ERF + noisy σ_res=0.05.
- **Matrix-free vs Aer** (`_validate_matrixfree.py`): cos=1.0000000000, Δsucc≤4e-19 at T=10/50.
- **QTR + LVT imports** clean.
- *Caveat:* a redundant noisy-cell production cross-check via the Aer engine **timed out** (Aer is too
  slow at T=50 σ_res=0.05 under a 10-min cap) — not a correctness gap; matrix-free bit-identity is
  established by the checks above.
- **`build_metrics.py` view rebuild: DEFERRED** — it rewrites `manifest/metrics.csv` (a store write,
  not concurrent-safe) and a parallel chat is active in the store (see §7). Run it once the store is quiet.

## 7. ⚠️ LIVE CONCURRENCY — left untouched, escalated to the user

A parallel chat is **actively editing this repo right now** (verified by mtimes during this pass;
`Maninm/` written at the current minute). I deliberately did **not** touch its in-flight work:

- `M PROJECT_STATUS.md` — being edited by the other chat (mtime 16:57). I did **not** apply the planned
  single-status update to avoid racing it. **PENDING USER DECISION: who owns the final PROJECT_STATUS edit.**
- `M Epsilon_study_2/theory.md` — other chat (mtime 17:10).
- New `Epsilon_study_2/gen_detector_physics.py` + `outputs/detector_physics.{json,report.md}` + 5
  `figures/epsilon_sensitivity/dp{1..5}_*.png` — a **detector-physics analysis**, written 17:06–17:12,
  possibly still in progress. **Left for its owning chat to commit** (avoids capturing half-written work).
- New `Maninm/` (a typo'd "Manim") — a Python `.venv` + quarto/manim tool install at the repo root.
  **RESOLVED (user-approved):** added `Maninm/` + generic `.venv/`/`venv/` rules to `.gitignore`
  (commit `fc32f9b5`); its contents were **not** committed.

By the end of the pass the concurrency had grown to **several** simultaneous chats editing this shared
working dir (verified live): a **high-T 1BQF refresh** (it checked out branch
`highT-1bqf-refresh-2026-06-14` over the shared checkout, + `HIGHT_1BQF_REFRESH_LOG.md`, re-running
Verify/Larger_Scatter store figures — using the new matrix-free engine), a **detector-physics** chat in
its own worktree `.git_dp_worktree/` (detached `639a8657`), and a **QSVT Segment_level_studies** chat
regenerating figures. All of their uncommitted/worktree work was **left untouched**.

**Final merge to `main`: DONE (user-approved).** My 7 reconciled commits were a clean linear
fast-forward over `origin/main` (`d6fb474d..fc32f9b5`). `main` was advanced as a **ref-only**
fast-forward (`git branch -f main fc32f9b5`, no checkout — so no active chat's working files moved) and
**pushed**: `d6fb474d..fc32f9b5  main -> main`. The redundant `sync-merge-2026-06-14` branch was then
retired (fully folded into main). The other chats' branch `highT-1bqf-refresh-2026-06-14` and the
`.git_dp_worktree` were **left in place** (theirs to finish/merge).

**Still left to the owning chats / pending:**
- `PROJECT_STATUS.md` + `theory.md` + the `detector_physics` analysis — **left to their owning chats**
  to commit; I did not edit PROJECT_STATUS to avoid racing. (User-confirmed.)
- `build_metrics.py` view rebuild — still **deferred** until the store is quiet (it rewrites
  `manifest/metrics.csv`, not concurrent-safe, and several chats are live in the store).

## 8. End state

- **LVT:** clean, one branch (`main` @ `17db26f`), pushed. ✅
- **QTR:** `main` @ `fc32f9b5` **pushed to origin** — contains all reconciled sync work (matrix-free
  engine, guides, Verify topup, QSVT docs, speedup reconciliation, deletions, gitignore, this log) on
  top of the parallel chat's already-pushed indefiniteness work. `sync-merge` retired. The working tree
  is **not** clean, but the entire remaining diff is **exclusively other chats' live in-flight work**
  (§7), deliberately untouched — not lost, theirs to commit. Branch `highT-1bqf-refresh-2026-06-14`
  (== main) + `.git_dp_worktree` belong to active chats.
- **Notion + memory + guides** tell one consistent story (matrix-free default; ~25× exact A-build with
  the 8.5× micro-benchmark floor noted; metrics-as-view at τ=0.35; per-solver efficiency-first points).
- **Nothing discarded.**
