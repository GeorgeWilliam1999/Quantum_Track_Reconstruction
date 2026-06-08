# Clarifications needed before finalising the presentation

<!-- STATUS: open questions for George -->

## A. Hardware figure (Hellinger / SSI panel)
1. The legend shows **solid = Emulator, dotted = Hardware**. Several IBM-Fez and IBM-Pittsburgh dotted points sit at SSI < 1. Was the **H2 (Quantinuum)** point ever run on real hardware, or only emulator? The legend doesn't show a dotted H2 line — confirm.
2. The five problem configurations on the x-axis are `2T 3L`, `2T 5L`, `4T 3L`, `4T 5L`, `8T 3L`. Confirm `T = tracks`, `L = layers`, and that this is the same data Xeno produced (i.e. not yours).
3. Should the slide attribute this work to **Xeno + Alain** (Marcel mentioned authorship needs discussion)?

## B. Realistic-MC benchmark (Xeno's segment-vs-SBT result)
4. The response text already quotes **94.2 % segment efficiency vs SBT 94.8 %** and **5.1–13.7 % raw quantum fake rate**. You said *"I do not yet have his results, they are not ready"* — should I:
   - (a) show this slide as a placeholder ("results pending — numbers above are draft from Xeno"), or
   - (b) drop the slide entirely and just mention it as an open item?
5. Was that simulation **statevector** (no quantum noise) or under a noise model? Needed to caption correctly.

## C. Your toy segment-level results
6. Headline numbers from `segment_level_analysis.ipynb` §16d table:
   - Segment-level efficiency = **100 %** at every $n \in \{10, 30, 100, 300, 500, 750, 1000\}$.
   - Track-level efficiency with **CC tracker**: 99 % → **44 % at n=1000**.
   - Track-level efficiency with **layered tracker** (§17): **≥ 99.96 % at every n up to 1000**.
   - False-positive segment rate at threshold 0.35: 1.2 % (n=300) → **19.8 % (n=1000)**.
   Confirm these are the headline numbers you want on the slide. Anything else?
7. Is the 1 % drop-rate study comparing **clean vs drop=1 %** the cleanest single panel to show? I'd pick `fig14_solver_segment_efficiency_overlay_drop1pct.pdf` (overlay, both conditions). OK?
8. Do you want the **§17 tracker A/B** result (CC vs layered) in this presentation? It's a strong "the Hamiltonian is fine, the tracker was the bottleneck" story but it's *not directly a reviewer response* — it's adjacent. Include or skip?
9. The §18 quantum (1BQF on the toy) sweeps go up to $n_\mathrm{trk}=6$ locally and were being pushed via condor for larger n. Any results from those condor runs ready to show, or is that also still in progress?

## D. Slide #1 logistics
10. Talk title? I'll default to *"TrackHHL 1-bit Quantum Filter — Reviewer Response Status"* unless you say otherwise.
11. Date / venue / your affiliation line for the title slide?
12. Co-presenter or sole presenter (Xeno on his slides, you on yours)?

## E. Open items slide — anyone owning what?
13. From the major-comment TODOs the response doc lists these as still open. Confirm owners (and which ones we will *not* deliver in the 2 weeks):
    - Major 1: classical limits / timescales prose, "outperform classical" sentence — owner?
    - Major 2: rest on Xeno's MC results.
    - Major 3: generality prose — owner?
    - Minor: hardware run on real device (Quantinuum tokens) — status?
    - Minor: Fig 2 axes / labels — yours?
    - Minor: error mitigation paragraph — owner?
    - Minor: Eq. 10 intermediate step (already drafted in the "test area" of the response doc — just needs merging) — owner?
    - L.221 `n_s` definition, L.240 reference, L.316-317 expansion, r1/r2/r3 consistency — owner?

## F. Tone of the closing slide
14. You said "we are not ready, can discuss open items". Should the closing slide explicitly list a **risk** ("response will be partial at the 2-week deadline") or frame it more neutrally ("schedule of remaining work")?
