"""
01 -- exact verification + measured resources of every block-encoding method on
small real store events and canonical clusters.

Output: outputs/01_verify_table.csv  (one row per matrix x method)
Run:    Q_env python 01_verify_and_measure.py
"""

import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/data/bfys/gscriven/Quantum_Track_Reconstruction/Toy_Characterisation/_shared")
sys.path.insert(0, "/data/bfys/gscriven/LHCb_VeLo_Toy_Model/src")

import be_lib as bl
import qtrk_pipeline as qp

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)

GAMMA, DELTA, EPS = 3.0, 1.0, 0.002
VERIFY_NMAX = 64          # exact block extraction up to this padded dim
TRANSPILE_NMAX = 64       # transpile FABLE-family circuits up to this padded dim


def matrices():
    out = {}
    for name, A in bl.canonical_clusters(GAMMA, DELTA).items():
        out[name] = sp.csr_matrix(A)
    for T in (2, 3, 5):
        ev, _ = qp.ensure_event(n_trk=T, rep=0, sigma_scatt=1e-4, sigma_res=0.0,
                                phi_max=0.2, hit_ineff=0.0)
        ham = qp.build_hamiltonian(ev, epsilon=EPS, kernel="step",
                                   gamma=GAMMA, delta=DELTA)
        out[f"T{T}_step"] = ham.A.tocsr()
        if T == 3:
            hame = qp.build_hamiltonian(ev, epsilon=EPS, kernel="erf",
                                        erf_sigma=EPS / 3.0,
                                        gamma=GAMMA, delta=DELTA)
            out[f"T{T}_erf"] = hame.A.tocsr()
    return out


def offdiag(A):
    C = sp.csr_matrix(A - sp.diags(A.diagonal()))
    C.eliminate_zeros()
    return C


def run_method(tag, ctor, A, target=None, rel_to=None, transpilable=False):
    """Build, verify (if small), count. target overrides the comparison matrix."""
    d = A.shape[0]
    n, N = bl.npad(d)
    row = dict(method=tag, d=d, n_pad=N)
    t0 = time.time()
    try:
        qc, alpha, info = ctor(A)
    except Exception as e:  # noqa
        row["error"] = f"BUILD FAIL: {e}"
        return row
    row.update(alpha=alpha, qubits=qc.num_qubits, n_sys=info["n_sys"],
               n_anc=info["n_anc"], prep_s=round(info.get("prep_s", time.time() - t0), 4))
    for k in ("n_rot", "s0", "s_pad", "chi_greedy", "scale", "hah_max"):
        if k in info:
            row[k] = info[k]
    if N <= VERIFY_NMAX:
        B = bl.extract_block(qc, info["n_sys"]).real
        if target is not None:
            tgt = target
            err = np.abs(B[: tgt.shape[0], : tgt.shape[1]] - tgt).max()
        else:
            tgt = A.toarray() if sp.issparse(A) else np.asarray(A)
            err = np.abs(B[:d, :d] * alpha - tgt).max()
        row["enc_err_max"] = float(err)
    if transpilable and N <= TRANSPILE_NMAX:
        try:
            row.update({f"g_{k}": v for k, v in bl.gate_counts(qc).items()
                        if k != "ops"})
        except Exception as e:  # noqa
            row["g_cx"] = f"transpile fail: {e}"
    return row


def main():
    rows = []
    mats = matrices()
    for mname, A in mats.items():
        d = A.shape[0]
        n, N = bl.npad(d)
        C = offdiag(A)
        Cabs = sp.csr_matrix(abs(C))
        print(f"--- {mname}: d={d} pad={N} nnz_off={C.nnz}", flush=True)

        entries = [
            ("fable", lambda M: bl.fable(M.toarray()), A, None, True),
            ("sfable", lambda M: bl.sfable(M.toarray()), A, None, True),
            ("lsfable", lambda M: bl.lsfable(M.toarray()), A, None, True),
            ("dict_A", bl.dictionary_hermitian, A, None, False),
            ("dict_C", lambda M: bl.dictionary_hermitian(M, include_diagonal=False), C, None, False),
            ("camps_A", bl.camps_sparse, A, None, False),
            ("camps_C", lambda M: bl.camps_sparse(M, include_diagonal=False), C, None, False),
        ]
        for tag, ctor, M, tgt, transp in entries:
            r = run_method(tag, ctor, M, target=tgt, transpilable=transp)
            r["matrix"] = mname
            rows.append(r)
            print("   ", tag, {k: r.get(k) for k in ("alpha", "enc_err_max", "g_cx", "g_oneq", "g_depth")},
                  flush=True)

        # szegedy on |C| (verification uses info['target'] = discriminant)
        if N <= VERIFY_NMAX:
            qc, alpha, info = bl.szegedy_discriminant(Cabs)
            B = bl.extract_block(qc, info["n_sys"]).real
            err = np.abs(B - info["target"]).max()
            rows.append(dict(matrix=mname, method="szegedy_C", d=d, n_pad=N,
                             alpha=1.0, qubits=qc.num_qubits, n_sys=info["n_sys"],
                             n_anc=info["n_anc"], enc_err_max=float(err),
                             prep_s=round(info["prep_s"], 4)))
            print("    szegedy_C", dict(alpha=1.0, err=float(err)), flush=True)

        # dilation baseline
        qc, alpha, info = bl.dilation_baseline(A)
        r = dict(matrix=mname, method="dilation_A", d=d, n_pad=N, alpha=alpha,
                 qubits=qc.num_qubits, n_sys=info["n_sys"], n_anc=1,
                 prep_s=round(info["prep_s"], 4))
        if N <= VERIFY_NMAX:
            B = bl.extract_block(qc, info["n_sys"]).real
            r["enc_err_max"] = float(np.abs(B - info["target"]).max())
        rows.append(r)
        print("    dilation_A", dict(alpha=round(alpha, 3), err=r.get("enc_err_max")), flush=True)

        # classical-vs-circuit cross-check of the FABLE compression emulator
        if N <= VERIFY_NMAX:
            for thr in (0.0, 1e-3):
                enc, kept = bl.fable_classical(A.toarray(), thr)
                qcf, af, inf = bl.fable(A.toarray(), compress=thr)
                Bf = bl.extract_block(qcf, inf["n_sys"]).real * af
                x = np.abs(Bf[:d, :d] - enc).max()
                rows.append(dict(matrix=mname, method=f"fable_classical_xcheck_thr{thr}",
                                 d=d, n_pad=N, enc_err_max=float(x), n_rot=kept))
                print(f"    fable classical-emulator xcheck thr={thr}: {x:.2e} kept={kept}",
                      flush=True)

    df = pd.DataFrame(rows)
    lead = ["matrix", "method", "d", "n_pad", "alpha", "enc_err_max",
            "qubits", "n_anc", "n_rot", "g_cx", "g_oneq", "g_depth", "prep_s"]
    cols = [c for c in lead if c in df.columns] + [c for c in df.columns if c not in lead]
    df = df[cols]
    path = os.path.join(OUT, "01_verify_table.csv")
    df.to_csv(path, index=False)
    print("wrote", path)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
