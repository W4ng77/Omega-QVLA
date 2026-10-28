# Authoritative submission contract

Source: user-provided, page-verified description of the8-page submitted PDF,2026-10-28. The local arXiv2605.28803v3 PDF has an appendix and is NOT accepted as the submitted version. The following claims are grounded in the supplied submission specification, not arXiv appendix tables.

- Page4 §4.1 Eqs1–5, row-vector Y=XW with W[Cin,Cout]: Xprime=X P R, Wprime=R.T P.T W. In PyTorch W is transposed: permute input columns of weight, right-multiply each input block by R, then xrot @ wrot.T+bias. No output-side SVD/restore or high-precision low-rank branch.
- Block64, left singular vectors of each input-row block in paper orientation; R=U D H64/sqrt64. D is frozen random±1. Figure1 omits D ONLY for visualization. The D seed is unspecified.
- Weight-energy descending, zigzag across blocks. No unique permutation pseudocode; row/column norm terminology inconsistent. Snake is an explicitly chosen mapping, not the uniquely prescribed algorithm.
- Page4 §4.2 Eqs6–8: signed symmetric[-7,7]. Weights per output channel. LLM activation per token. DiT activation per step/layer/input channel. Weight group size is NOT GPTQ processing block size128.
- DiT robust_peak is unspecified. q99.9 comes from historical code and must be labeled as such. Neither frame count nor token cap is given.
- Correct DiT oracle: Y[m,n]=Sw[n] sum_k(Qx[m,k]*Sa[t,k]*Qw[n,k])+bias[n]. Sa[t,k] remains INSIDE the K reduction. The scalar epilogue shorthand in Eq9 is insufficient and must be corrected.
- Page5 §5.1: GR00T-N1.5 and Pi-0.5; language+full DiT attention/MLP target linears; block64 SVD-Hadamard+zigzag; LLM GPTQ-style processing block128,damping.01; DiT RTN;8 Euler calibration steps;10 unlabeled calibration trajectories; main hardwareH100.
- Calibration origins, frames/trajectory, tokens, initial states, seeds, checkpoint revisions, detailed LIBERO protocol and operation dtype are not fully specified. Recover what historical artifacts prove; mark unresolved details. A disclosed10x10-frame rebuild is not automatically the original experiment.
- Target scope does NOT mean all vision/auxiliary/nonlinear operations are INT4. Actual module names and shapes must be audited.
- Table2 GR00T HoloQ Goal91/Spatial86/Object92/Long82, average87.75→87.8; GR00T FP16 86/92/92/76, average86.5. Narrative87.0 conflicts with that table and must be corrected. Pi-0.5 HoloQ100/99/97/96, average98.0. These are reference claims, not tuning targets.
- Table4 no-per-step Object92/Long75/average85.75 versus per-step Object92/Long82/average87.75. Object alone cannot establish the per-step benefit; Long is required after implementation freeze.

step_weight and group_int are separate changed deployment variants. Neither proves a strictly equivalent original-method INT4 implementation. No extra per-step W banks are prescribed by the submission.
