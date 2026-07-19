"""
be_lib -- block-encoding constructions + verification + cost models for the
segment Hamiltonian A = (gamma+delta) I - C  (qtrk store events).

Every constructor returns (qc, alpha, info) where

    qc     : qiskit QuantumCircuit with the SYSTEM register on qubits [0, n_sys)
             (little-endian) and all ancillas above, such that
                 <0_anc, i| U |0_anc, j>  =  A_ij / alpha   (+ encoding error)
    alpha  : the total subnormalization (all rescale factors folded in)
    info   : dict of metadata (ancilla count, prep time, method-specific numbers)

Verification is by explicit block extraction (`extract_block`), practical up to
~15 total qubits.  Analytic large-T cost models live at the bottom.
"""

import time
import math

import numpy as np
import scipy.sparse as sp


# ======================================================================
# helpers
# ======================================================================

def npad(dim):
    """Qubits needed for dim, and the padded dimension."""
    n = max(1, int(math.ceil(math.log2(dim))))
    return n, 1 << n


def pad_dense(A, N):
    A = np.asarray(A, float)
    out = np.zeros((N, N))
    out[: A.shape[0], : A.shape[1]] = A
    return out


def extract_block(qc, n_sys, dim=None):
    """Top-left block <0_anc, i|U|0_anc, j> for j < dim.

    Batched gate-by-gate evolution of all `dim` basis columns at once (one pass
    over the circuit), so kilo-gate FABLE circuits stay tractable.
    """
    from qiskit.quantum_info import Operator

    n_tot = qc.num_qubits
    N = 1 << n_sys
    dim = dim or N
    psi = np.zeros((1 << n_tot, dim), complex)
    for j in range(dim):
        psi[j, j] = 1.0                   # |0_anc>|j>, system = low qubits
    T = psi.reshape((2,) * n_tot + (dim,))
    qindex = {q: i for i, q in enumerate(qc.qubits)}
    for inst in qc.data:
        if inst.operation.name in ("barrier", "measure"):
            continue
        G = Operator(inst.operation).data
        qs = [qindex[q] for q in inst.qubits]
        k = len(qs)
        # leading k axes ordered MSB-first over the gate's own index
        axes = [n_tot - 1 - q for q in reversed(qs)]
        T = np.moveaxis(T, axes, range(k))
        shp = T.shape
        T = (G @ T.reshape(1 << k, -1)).reshape(shp)
        T = np.moveaxis(T, range(k), axes)
    out = T.reshape(1 << n_tot, dim)
    return out[:dim, :]


def gate_counts(qc, basis=("rz", "ry", "rx", "h", "x", "cx"), opt=1):
    """Transpile to a {1q, cx} basis and count."""
    from qiskit import transpile

    t = transpile(qc, basis_gates=list(basis), optimization_level=opt)
    ops = t.count_ops()
    cx = ops.get("cx", 0)
    tot = sum(ops.values())
    return dict(cx=int(cx), oneq=int(tot - cx), depth=int(t.depth()), ops=dict(ops))


# ----------------------------------------------------------------------
# uniformly controlled Ry (Mottonen / FABLE): angle transform + Gray walk
# ----------------------------------------------------------------------

def _sfwht(a):
    """Scaled fast Walsh-Hadamard transform (in place on a copy)."""
    a = np.asarray(a, float).copy()
    n = a.size
    k = 1
    while k < n:
        for i in range(0, n, 2 * k):
            x = a[i : i + k].copy()
            y = a[i + k : i + 2 * k].copy()
            a[i : i + k] = (x + y) / 2.0
            a[i + k : i + 2 * k] = (x - y) / 2.0
        k *= 2
    return a


def _gray_permutation(a):
    """Permute vector so entry g(k) -> position k (g = binary-reflected Gray)."""
    n = a.size
    out = np.empty_like(a)
    for k in range(n):
        out[k] = a[k ^ (k >> 1)]
    return out


def _ucry_angles(theta):
    """Transformed angles for the Gray-walk uniformly controlled Ry."""
    return _gray_permutation(_sfwht(np.asarray(theta, float)))


def _ctrl_index(k, m):
    """Control qubit index (0-based, LSB=0) flipped between Gray(k) and Gray(k+1 mod 2^m)."""
    if k == (1 << m) - 1:
        return m - 1  # wraparound: gray(K-1) and gray(0) differ in the MSB
    return int(math.log2(((k + 1) & -(k + 1))))


def append_ucry(qc, theta_hat, target, controls, skip_zero=0.0):
    """Gray-walk UCRy: rotations theta_hat[k] then CNOT(controls[c_k], target).

    `controls` little-endian (controls[0] = LSB of the address).
    skip_zero: drop |angle| <= skip_zero rotations and cancel merged CNOTs
    (FABLE compression: consecutive CNOTs on the same target XOR-merge).
    Returns (n_rot, n_cnot) actually emitted.
    """
    K = len(theta_hat)
    m = len(controls)
    assert K == 1 << m
    n_rot = 0
    pend = 0  # XOR-accumulated pending control mask
    for k in range(K):
        if abs(theta_hat[k]) > skip_zero:
            # flush pending CNOTs (one per set bit)
            b = pend
            while b:
                c = int(math.log2(b & -b))
                qc.cx(controls[c], target)
                b &= b - 1
            pend = 0
            qc.ry(theta_hat[k], target)
            n_rot += 1
        pend ^= 1 << _ctrl_index(k, m)
    b = pend
    n_cnot_tail = 0
    while b:
        c = int(math.log2(b & -b))
        qc.cx(controls[c], target)
        b &= b - 1
    # count CNOTs emitted by inspecting the data is overkill; recount:
    n_cnot = sum(1 for g in qc.data if g.operation.name == "cx")
    return n_rot, n_cnot


# ======================================================================
# 1. FABLE  (Camps & Van Beeumen; oracle form of Kuklinski Thm 1)
# ======================================================================

def fable(A, compress=0.0):
    """FABLE block encoding of A / (2^n * scale), scale = max(1, max|a_ij|).

    Registers: sys (n, low) | row (n) | flag (1, top).
    """
    from qiskit import QuantumCircuit

    t0 = time.time()
    d = A.shape[0]
    n, N = npad(d)
    scale = max(1.0, float(np.max(np.abs(A))))
    Ap = pad_dense(np.asarray(A, float) / scale, N)

    theta = 2.0 * np.arccos(np.clip(Ap, -1.0, 1.0))
    # vec ordering: address = (row bits high, col bits low)  -> theta[i*N+j]
    theta_hat = _ucry_angles(theta.reshape(-1))

    qc = QuantumCircuit(2 * n + 1)
    sysq = list(range(n))
    rowq = list(range(n, 2 * n))
    flag = 2 * n
    for q in rowq:
        qc.h(q)
    # UCRy on flag controlled by (col=sys low, row high): address LSB = sys[0]
    n_rot, _ = append_ucry(qc, theta_hat, flag, sysq + rowq, skip_zero=compress)
    for a, b in zip(sysq, rowq):
        qc.swap(a, b)
    for q in rowq:
        qc.h(q)
    alpha = N * scale
    info = dict(n_sys=n, n_anc=n + 1, n_rot=n_rot, scale=scale,
                prep_s=time.time() - t0, method="fable")
    return qc, alpha, info


# ======================================================================
# 2. S-FABLE and 3. LS-FABLE  (Kuklinski & Rempfer)
# ======================================================================

def _hadamard_conj(A, N):
    """H^{ox n} A H^{ox n} via fast Walsh-Hadamard on both sides (orthonormal H)."""

    def fwht(M, axis):
        M = M.copy()
        n = M.shape[axis]
        k = 1
        while k < n:
            for i in range(0, n, 2 * k):
                if axis == 0:
                    x = M[i : i + k, :].copy()
                    y = M[i + k : i + 2 * k, :].copy()
                    M[i : i + k, :] = x + y
                    M[i + k : i + 2 * k, :] = x - y
                else:
                    x = M[:, i : i + k].copy()
                    y = M[:, i + k : i + 2 * k].copy()
                    M[:, i : i + k] = x + y
                    M[:, i + k : i + 2 * k] = x - y
            k *= 2
        return M

    Ap = pad_dense(A, N)
    return fwht(fwht(Ap, 0), 1) / N


def sfable(A, compress=0.0):
    """S-FABLE: FABLE-encode HAH/||HAH||_max, conjugate with Hadamards.

    alpha = 2^n * ||HAH||_max (entrywise max, >=1-clipped).
    """
    from qiskit import QuantumCircuit

    t0 = time.time()
    d = A.shape[0]
    n, N = npad(d)
    HAH = _hadamard_conj(np.asarray(A, float), N)
    qc_in, a_in, info = fable(HAH, compress=compress)
    qc = QuantumCircuit(qc_in.num_qubits)
    for q in range(n):
        qc.h(q)
    qc.compose(qc_in, inplace=True)
    for q in range(n):
        qc.h(q)
    alpha = a_in  # N * max|HAH|
    info.update(method="sfable", prep_s=time.time() - t0,
                hah_max=float(np.max(np.abs(HAH))))
    return qc, alpha, info


def lsfable(A):
    """LS-FABLE: transformed angles set directly to (pi/2) E_00 - A/N.

    Encodes H sin(HAH) H ~ A / 2^n; error is fixed by the matrix.  Rotations
    emitted only at nonzeros (+ the corner).
    """
    from qiskit import QuantumCircuit

    t0 = time.time()
    d = A.shape[0]
    n, N = npad(d)
    Ap = pad_dense(np.asarray(A, float), N)
    # transformed-angle matrix (pi/2) E_00 - A/N, doubled because the walk
    # applies Ry(theta_hat) and the encoded entry is cos(theta/2)
    theta_hat = (-2.0 * Ap / N).reshape(-1)
    theta_hat[0] += np.pi
    # NOTE: theta_hat lives in the *transformed* domain but FABLE's walk applies
    # angles in Gray order over the SAME address ordering used in `fable`;
    # placing -a_ij/N at transformed slot (i,j) reproduces the paper's oracle.
    theta_hat = _gray_permutation(theta_hat)  # reorder linear index -> Gray walk

    qc = QuantumCircuit(2 * n + 1)
    sysq = list(range(n))
    rowq = list(range(n, 2 * n))
    flag = 2 * n
    for q in sysq:            # LS-FABLE keeps the S-FABLE Hadamard conjugation
        qc.h(q)
    for q in rowq:
        qc.h(q)
    n_rot, _ = append_ucry(qc, theta_hat, flag, sysq + rowq, skip_zero=0.0)
    for a, b in zip(sysq, rowq):
        qc.swap(a, b)
    for q in rowq:
        qc.h(q)
    for q in sysq:
        qc.h(q)
    alpha = float(N)
    info = dict(n_sys=n, n_anc=n + 1, n_rot=int(np.count_nonzero(theta_hat)),
                prep_s=time.time() - t0, method="lsfable")
    return qc, alpha, info


# ======================================================================
# edge coloring (matching decomposition) of the coupling graph C
# ======================================================================

def edge_coloring(C):
    """Greedy proper edge coloring of the symmetric 0-pattern of C.

    Returns list of matchings [ {j: partner} ... ] covering every off-diagonal
    nonzero pair once.  Greedy uses <= 2*Delta - 1 colors (report vs Delta).
    """
    C = sp.csr_matrix(C)
    n = C.shape[0]
    Cu = sp.triu(C, k=1).tocoo()
    edges = list(zip(Cu.row.tolist(), Cu.col.tolist()))
    # sort by degree-sum descending helps greedy a bit; keep deterministic
    deg = np.asarray((C != 0).sum(1)).ravel()
    edges.sort(key=lambda e: -(deg[e[0]] + deg[e[1]]))
    colors = []          # list of dict j->partner
    used = []            # list of set of vertices used per color
    for (i, j) in edges:
        for c in range(len(colors)):
            if i not in used[c] and j not in used[c]:
                colors[c][i] = j
                colors[c][j] = i
                used[c].update((i, j))
                break
        else:
            colors.append({i: j, j: i})
            used.append({i, j})
    return colors


# ======================================================================
# 4. Dictionary-based block encoding (Yang et al. 2405.18007, Hermitian form)
#    LCU over involutions: A = sum_l A_l P_l  with the del-flag sector trick.
# ======================================================================

def dictionary_hermitian(A, include_diagonal=True, values=None):
    """Hermitian dictionary block encoding via LCU over matching involutions.

    For the step kernel: classes = [diagonal (value gamma+delta)] +
    [one class per edge color (value -1 each)]  ->  alpha = (gamma+delta) + chi'.
    For general symmetric A, classes split further by value (each (color, value)
    pair its own class); alpha = sum_l |A_l| * 1.

    Implementation (verification-grade): SELECT = block-diagonal sparse unitary
    over classes; each class unitary acts on (del + sys):
        matched pair (i,j):  |0,i> <-> |0,j>
        unmatched j:         |0,j> <-> |1,j>       (del flag -> orthogonal)
    PREP = StatePreparation( sqrt(|A_l|/alpha) ) with the value's sign folded
    into the select unitary (values here are real).
    Registers: sys (n, low) | del (1) | idx (m, top).
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import StatePreparation, UnitaryGate

    t0 = time.time()
    A = sp.csr_matrix(A)
    d = A.shape[0]
    n, N = npad(d)
    diag = np.asarray(A.diagonal()).ravel()
    s_diag = float(diag[0]) if include_diagonal else 0.0
    if include_diagonal and not np.allclose(diag[: d], diag[0]):
        raise ValueError("dictionary_hermitian assumes constant diagonal")
    Coff = A - sp.diags(np.r_[diag, np.zeros(0)])

    # split every color class by (rounded) value so each class has ONE value
    colors = edge_coloring(Coff)
    classes = []  # (value, {j: partner})
    for match in colors:
        by_val = {}
        seen = set()
        for j, p in match.items():
            if j in seen:
                continue
            seen.update((j, p))
            v = float(Coff[j, p])
            by_val.setdefault(round(v, 12), {})
            by_val[round(v, 12)][j] = p
            by_val[round(v, 12)][p] = j
        for v, mm in by_val.items():
            classes.append((v, mm))
    if include_diagonal and abs(s_diag) > 0:
        classes = [(s_diag, None)] + classes  # None = identity on d dims

    vals = np.array([abs(v) for v, _ in classes])
    alpha = float(vals.sum())
    s0 = len(classes)
    m, M = npad(max(2, s0))

    # PREP amplitudes
    amps = np.zeros(M)
    amps[:s0] = np.sqrt(vals / alpha)

    # SELECT: block diagonal over idx of (1+n)-qubit involutions with signs
    dimS = 2 * N
    Usel = sp.lil_matrix((M * dimS, M * dimS))
    for l in range(M):
        base = l * dimS
        if l >= s0:
            for k in range(dimS):
                Usel[base + k, base + k] = 1.0
            continue
        v, match = classes[l]
        sign = 1.0 if v >= 0 else -1.0
        P = sp.lil_matrix((dimS, dimS))
        touched = np.zeros(dimS, bool)
        if match is None:  # diagonal class: identity on first d, flag the rest
            for j in range(d):
                P[j, j] = sign
                touched[j] = True
            for j in range(d, N):
                P[N + j, j] = 1.0
                P[j, N + j] = 1.0
                touched[j] = touched[N + j] = True
        else:
            for j, p in match.items():
                if touched[j]:
                    continue
                P[p, j] = sign
                P[j, p] = sign
                touched[j] = touched[p] = True
            for j in range(N):
                if not touched[j]:  # unmatched: del-flag swap (value 0 in block)
                    P[N + j, j] = 1.0
                    P[j, N + j] = 1.0
                    touched[j] = touched[N + j] = True
        for j in range(dimS):
            if not touched[j]:
                P[j, j] = 1.0
        Usel[base : base + dimS, base : base + dimS] = P
    Usel = Usel.tocsr()

    qc = QuantumCircuit(n + 1 + m)
    sysq = list(range(n))
    delq = n
    idxq = list(range(n + 1, n + 1 + m))
    prep = StatePreparation(amps / np.linalg.norm(amps))
    qc.append(prep, idxq)
    # SELECT ordering: idx = MOST significant -> matches block-diag over l
    try:
        gate = UnitaryGate(Usel.toarray(), check_input=False, label="SELECT")
    except TypeError:
        gate = UnitaryGate(Usel.toarray(), label="SELECT")
    qc.append(gate, sysq + [delq] + idxq)
    qc.append(prep.inverse(), idxq)

    info = dict(n_sys=n, n_anc=1 + m, s0=s0, chi_greedy=len(colors),
                prep_s=time.time() - t0, method="dictionary")
    return qc, alpha, info


# ======================================================================
# 5. Camps sparse-oracle block encoding (Thm 4.1) -- generic geometry
#    D_s . O_c . O_A . D_s  encoding A / (s_pad * scale)
# ======================================================================

def camps_sparse(A, include_diagonal=True):
    """Camps et al. Thm 4.1 with slot structure from the edge coloring.

    Slots: [diag] + one per color class (padded to s = 2^m).  c(j, l) = matching
    partner (or j).  O_A encodes the value at (c(j,l), j), including the 0 of
    unmatched (j, l) slots -- for generic geometry those angles are per-(j, l)
    data, which is the honest cost of this method on our matrices.
    Verification-grade: O_c and O_A assembled as explicit sparse unitaries.
    Registers: sys (n, low) | slot (m) | flag (1, top).
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate

    t0 = time.time()
    A = sp.csr_matrix(A)
    d = A.shape[0]
    n, N = npad(d)
    diag = np.asarray(A.diagonal()).ravel()
    Coff = A - sp.diags(diag)
    colors = edge_coloring(Coff)
    slots = ([("diag", None)] if include_diagonal else []) + [("match", mm) for mm in colors]
    s0 = len(slots)
    m, S = npad(max(2, s0))
    scale = max(1.0, float(abs(A).max()))

    # c(j, l) table and value table, both over padded (S, N)
    cjl = np.tile(np.arange(N), (S, 1))
    val = np.zeros((S, N))
    for l, (kind, mm) in enumerate(slots):
        if kind == "diag":
            val[l, :d] = diag / scale
        else:
            for j, p in mm.items():
                cjl[l, j] = p
                val[l, j] = Coff[p, j] / scale  # A_{c(j,l), j}

    # O_c: |l>|j> -> |l>|c(j,l)>   (each row of cjl is a permutation: matchings
    # are involutions, identity elsewhere)
    Oc = sp.lil_matrix((S * N, S * N))
    for l in range(S):
        for j in range(N):
            Oc[l * N + int(cjl[l, j]), l * N + j] = 1.0
    Oc = Oc.tocsr()

    # O_A: Ry(2 arccos(val[l, j])) on the flag, controlled on (l, j)
    # assembled directly as a sparse unitary on flag|slot|sys
    dimF = 2 * S * N
    OA = sp.lil_matrix((dimF, dimF))
    for l in range(S):
        for j in range(N):
            a = float(np.clip(val[l, j], -1, 1))
            c, s_ = a, math.sqrt(max(0.0, 1 - a * a))
            k0 = l * N + j            # flag 0
            k1 = S * N + l * N + j    # flag 1
            OA[k0, k0] = c
            OA[k1, k0] = s_
            OA[k0, k1] = -s_
            OA[k1, k1] = c
    OA = OA.tocsr()

    qc = QuantumCircuit(n + m + 1)
    sysq = list(range(n))
    slotq = list(range(n, n + m))
    flag = n + m
    for q in slotq:
        qc.h(q)
    try:
        qc.append(UnitaryGate(OA.toarray(), check_input=False, label="O_A"),
                  sysq + slotq + [flag])
        qc.append(UnitaryGate(Oc.toarray(), check_input=False, label="O_c"),
                  sysq + slotq)
    except TypeError:
        qc.append(UnitaryGate(OA.toarray(), label="O_A"), sysq + slotq + [flag])
        qc.append(UnitaryGate(Oc.toarray(), label="O_c"), sysq + slotq)
    for q in slotq:
        qc.h(q)
    alpha = float(S) * scale
    info = dict(n_sys=n, n_anc=m + 1, s0=s0, s_pad=S, scale=scale,
                chi_greedy=len(colors), prep_s=time.time() - t0, method="camps")
    return qc, alpha, info


# ======================================================================
# 6. Szegedy / discriminant direct encoding (Camps Thm 5.1): alpha = 1
#    encodes D^{-1/2} C D^{-1/2}  (degree-normalized coupling)
# ======================================================================

def szegedy_discriminant(C):
    """U_P = O_P^dag SWAP O_P with O_P|0^n>|j> = sum_k sqrt(P_jk) |k>|j>.

    P = D^{-1} C (row-stochastic; zero-degree rows get a self-loop), so the
    encoded block is the discriminant D^{-1/2} C D^{-1/2} with alpha = 1.
    O_P is completed to a unitary by QR; verification-grade only.
    Registers: sys (n, low) | walk (n, top).  Block sits on the sys register.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate

    t0 = time.time()
    C = sp.csr_matrix(C)
    d = C.shape[0]
    n, N = npad(d)
    Cp = sp.lil_matrix((N, N))
    Cp[:d, :d] = C
    deg = np.asarray(abs(Cp).sum(1)).ravel()
    P = sp.lil_matrix((N, N))
    for j in range(N):
        if deg[j] == 0:
            P[j, j] = 1.0
        else:
            row = Cp.rows[j]
            for k, col in enumerate(row):
                P[j, col] = abs(Cp.data[j][k]) / deg[j]
    P = P.tocsr()

    # O_P columns for inputs |0>|j>: state sum_k sqrt(P_jk)|k>|j>
    dim = N * N  # (walk register k high, sys j low)
    V = np.zeros((dim, N))
    for j in range(N):
        col = np.zeros(dim)
        pr = np.asarray(P[j].todense()).ravel()
        for k in np.nonzero(pr)[0]:
            col[k * N + j] = math.sqrt(pr[k])
        V[:, j] = col
    # complete to unitary: QR on [V | random]
    rng = np.random.default_rng(7)
    Mfull = np.concatenate([V, rng.standard_normal((dim, dim - N))], axis=1)
    Q, _ = np.linalg.qr(Mfull)
    # fix signs so first N columns == V
    for j in range(N):
        s_ = np.dot(Q[:, j], V[:, j])
        if s_ < 0:
            Q[:, j] *= -1
    OP = Q

    qc = QuantumCircuit(2 * n)
    sysq = list(range(n))
    wq = list(range(n, 2 * n))
    try:
        g = UnitaryGate(OP, check_input=False, label="O_P")
    except TypeError:
        g = UnitaryGate(OP, label="O_P")
    qc.append(g, sysq + wq)
    for a, b in zip(sysq, wq):
        qc.swap(a, b)
    qc.append(g.inverse(), sysq + wq)

    Dm = None
    with np.errstate(divide="ignore"):
        dsq = np.where(deg > 0, 1.0 / np.sqrt(np.maximum(deg, 1e-300)), 0.0)
    Dm = np.asarray((sp.diags(dsq) @ Cp.tocsr() @ sp.diags(dsq)).todense())
    Dm[deg == 0, :] = 0
    Dm[:, deg == 0] = 0
    for j in np.nonzero(deg == 0)[0]:
        Dm[j, j] = 1.0  # self-loop discriminant
    info = dict(n_sys=n, n_anc=n, target=Dm, prep_s=time.time() - t0,
                method="szegedy")
    return qc, 1.0, info


# ======================================================================
# 7. Baseline: the production exact dilation (QSVT class construction)
# ======================================================================

def dilation_baseline(A, spectral_bounds=None, pad_frac=0.02):
    """U = [[X, S],[S, -X]] with X = affine(A) covering [-1,1] exactly.

    alpha relative to A: applying p to X is equivalent to a *rescaled* comb, so
    the effective subnormalization is span(A)/2 (the best possible for any
    encoding that must fit spec(A) in [-1,1]).  Gate cost: one dense (n+1)-qubit
    UnitaryGate (simulation-only; O(4^n) elements).
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate

    t0 = time.time()
    A = sp.csr_matrix(A)
    d = A.shape[0]
    n, N = npad(d)
    if spectral_bounds is None:
        w = np.linalg.eigvalsh(A.toarray())
        lo, hi = float(w[0]), float(w[-1])
    else:
        lo, hi = spectral_bounds
    span = hi - lo
    lo -= pad_frac * span
    hi += pad_frac * span
    Xd = np.zeros((N, N))
    Xd[:d, :d] = (2 * A.toarray() - (lo + hi) * np.eye(d)) / (hi - lo)
    w, V = np.linalg.eigh(Xd)
    w = np.clip(w, -1, 1)
    S = (V * np.sqrt(1 - w ** 2)) @ V.T
    U = np.block([[Xd, S], [S, -Xd]])
    qc = QuantumCircuit(n + 1)
    try:
        qc.append(UnitaryGate(U, check_input=False, label="U_dil"), list(range(n + 1)))
    except TypeError:
        qc.append(UnitaryGate(U, label="U_dil"), list(range(n + 1)))
    # the block encodes X = (2A - (lo+hi))/ (hi-lo): affine, not A/alpha.
    info = dict(n_sys=n, n_anc=1, affine=(lo, hi), target=Xd,
                alpha_eff=(hi - lo) / 2.0, prep_s=time.time() - t0,
                method="dilation")
    return qc, (hi - lo) / 2.0, info


# ======================================================================
# classical emulation of FABLE-family encodings (for compression curves at
# sizes where circuit extraction is infeasible) -- verified against the
# circuits at small n in 01_verify_and_measure.py
# ======================================================================

def _wht_unnorm(x):
    """Unnormalized Walsh-Hadamard transform of a vector (in place on copy)."""
    x = np.asarray(x, float).copy()
    n = x.size
    k = 1
    while k < n:
        for i in range(0, n, 2 * k):
            a = x[i : i + k].copy()
            b = x[i + k : i + 2 * k].copy()
            x[i : i + k] = a + b
            x[i + k : i + 2 * k] = a - b
        k *= 2
    return x


def fable_classical(A, thr=0.0):
    """Encoded matrix (times alpha) and kept-rotation count of compressed FABLE."""
    d = A.shape[0]
    n, N = npad(d)
    scale = max(1.0, float(np.max(np.abs(A))))
    Ap = pad_dense(np.asarray(A, float) / scale, N)
    theta = 2.0 * np.arccos(np.clip(Ap, -1.0, 1.0)).reshape(-1)
    x = _sfwht(theta)
    kept = int(np.count_nonzero(np.abs(x) > thr))
    x[np.abs(x) <= thr] = 0.0
    theta_rec = _wht_unnorm(x)
    enc = np.cos(theta_rec / 2.0).reshape(N, N)  # = A/scale when thr=0
    return enc[:d, :d] * scale, kept


def sfable_classical(A, thr=0.0):
    """Encoded matrix (times alpha) and kept rotations of compressed S-FABLE."""
    d = A.shape[0]
    n, N = npad(d)
    HAH = _hadamard_conj(np.asarray(A, float), N)
    enc_t, kept = fable_classical(HAH, thr)  # fable's internal scale handles max|HAH|
    encN = pad_dense(enc_t, N)
    out = _hadamard_conj(encN, N)
    return out[:d, :d], kept


def lsfable_classical(A):
    """Encoded matrix (times alpha=N): N * H sin(HAH) H; rotations = nnz."""
    d = A.shape[0]
    n, N = npad(d)
    HAH = _hadamard_conj(np.asarray(A, float), N)
    out = _hadamard_conj(np.sin(HAH), N) * N
    return out[:d, :d], int(np.count_nonzero(A)) + 1


def canonical_clusters(gamma=3.0, delta=1.0):
    """P4 chain and K(1,4) hub as A = (gamma+delta) I - C."""
    s = gamma + delta
    P4 = np.zeros((4, 4))
    for i in range(3):
        P4[i, i + 1] = P4[i + 1, i] = 1.0
    hub = np.zeros((5, 5))
    for i in range(1, 5):
        hub[0, i] = hub[i, 0] = 1.0
    return {"P4": s * np.eye(4) - P4, "hub_K14": s * np.eye(5) - hub}


# ======================================================================
# analytic cost models (large T; counts in {1q, CX} unless stated)
# ======================================================================

def model_fable(N, n, n_rot=None):
    """Full FABLE: N^2 rotations + N^2 CNOTs (+2n H, n SWAP)."""
    r = N * N if n_rot is None else n_rot
    return dict(rot=r, cx=r + 3 * n, anc=n + 1, alpha_factor="2^n * max|a|",
                classical=f"O(N^2 log N) = {N*N*max(1,n):,}")


def model_lsfable(nnz, n, N):
    return dict(rot=nnz + 1, cx=nnz + 3 * n, anc=n + 1, alpha_factor="2^n",
                classical=f"O(nnz) = {nnz:,}")


def model_transposition_oracle(n_pairs, n):
    """Cost of one matching (involution) compiled as basis-state transpositions
    (Herbert-Sorci-Tang Thm 3(a) + standard Toffoli->{1q,CX})."""
    # per transposition on n qubits: 2n CNOT + (12n-36) Toffoli, Toffoli = 6 CX
    cx = n_pairs * (2 * n + 6 * max(0, 12 * n - 36))
    return dict(cx=cx, anc=2, note="per matching, unstructured")


def model_dictionary(n, s0, classes_pairs, use_sbm=False, nnz=None):
    """Dictionary: PREP (2 * 2^m rotations) + SELECT.

    SELECT via transposition-compiled involutions (low-ancilla route) or via
    SBM (depth O(log ns), ancilla O(n^2 s)).
    """
    m = max(1, math.ceil(math.log2(max(2, s0))))
    prep_rot = 2 * (1 << m)
    if use_sbm:
        return dict(rot=prep_rot, cx=None, depth=f"O(log(n*s)) = O({math.ceil(math.log2(max(2,(n*(nnz or 1))))):d})",
                    anc=f"O(n^2 s) = {n*n*(nnz or 0):,}", alpha_factor="sum|A_l|")
    cx = 0
    for np_ in classes_pairs:
        cx += model_transposition_oracle(np_, n)["cx"]
    # each class unitary is controlled on the m-qubit idx: multiply by ~2m
    cx *= max(1, 2 * m)
    return dict(rot=prep_rot, cx=cx, anc=1 + m + 2, alpha_factor="sum|A_l|")


def model_hit_oracle(T, n, w_pad, layers=5, qrom_style="linear"):
    """Our proposed geometry (hit-level) oracle -- see PLAN / writeup.

    Index oracle = register arithmetic (layer increment + hit-field copy) +
    predicted-position QROM over the ~layers*T hit coordinates; value oracle =
    comparator vs epsilon (step) or arithmetic erf rotation.  alpha = w_pad.
    """
    n_hits = layers * T
    if qrom_style == "linear":
        qrom_cx = 4 * n_hits            # ~4 CX per QROM entry (unary iterate)
        qrom_anc = math.ceil(math.log2(max(2, n_hits))) + 2
    else:  # select-swap
        qrom_cx = int(8 * math.sqrt(n_hits) * 16)
        qrom_anc = int(math.sqrt(n_hits)) * 16
    arith_cx = 30 * (16 ** 1)           # comparator + adders at 16-bit fixed point
    return dict(cx=2 * qrom_cx + arith_cx, anc=qrom_anc + 32,
                alpha_factor="w_pad (occupancy)", alpha=w_pad,
                classical=f"O(T log T) sort = {int(n_hits*math.log2(max(2,n_hits))):,}")


# ======================================================================
# structure metrics of a real matrix (for 02_structure_scaling)
# ======================================================================

def structure_metrics(A):
    A = sp.csr_matrix(A)
    d = A.shape[0]
    n, N = npad(d)
    diag = np.asarray(A.diagonal()).ravel()
    Coff = (A - sp.diags(diag)).tocsr()
    Coff.eliminate_zeros()
    deg = np.asarray((Coff != 0).sum(1)).ravel()
    vals = np.abs(Coff.data)
    colors = edge_coloring(Coff)
    # distinct XOR displacements (the structured-dictionary axis)
    coo = sp.triu(Coff, k=1).tocoo()
    xors = set(int(i) ^ int(j) for i, j in zip(coo.row, coo.col))
    out = dict(
        d=d, n=n, N=N, nnz_off=int(Coff.nnz), delta_max=int(deg.max(initial=0)),
        delta_mean=float(deg.mean()) if d else 0.0,
        chi_greedy=len(colors),
        distinct_vals=int(np.unique(np.round(Coff.data, 10)).size),
        sum_abs_vals=float(vals.sum()),
        max_abs=float(np.abs(A.data).max()) if A.nnz else 0.0,
        distinct_xor=len(xors),
        classes_pairs=[len(mm) // 2 for mm in colors],
    )
    try:
        if d <= 3000:
            w = np.linalg.eigvalsh(A.toarray())
            out["lam_min"], out["lam_max"] = float(w[0]), float(w[-1])
        else:
            from scipy.sparse.linalg import eigsh
            out["lam_max"] = float(eigsh(A, k=1, which="LA", return_eigenvectors=False)[0])
            out["lam_min"] = float(eigsh(A, k=1, which="SA", return_eigenvectors=False)[0])
    except Exception:
        out["lam_min"] = out["lam_max"] = np.nan
    return out
