# exafs_shell_fit.py
# ======================================================================
# Physically consistent EXAFS fitting using Larch + FEFF-generated paths
# CN-free design (default): SS-path amplitudes share a single global scale Nscale
# Optional: per-shell amplitudes Nscale_j for each SS shell (per_shell_nscale=True)
# MS paths DO NOT use Nscale; they inherit σ² and ΔR from nearest SS shell.
#
# Shell-by-Shell Fitting Update
# ----------------------------------------------------------------------
# Upgrades kept from V2:
#  1) Noise-adaptive weighting (damped high-k) + SNR-aware metrics
#  2) Kaiser window defaults and tunables
#  3) DBSCAN-like shell clustering (pure-Python, no sklearn)
#  4) Automatic MS path pruning by effective amplitude & reachability
#  5) Smarter jitter (Gaussian, bounded, log-space for σ²)
#  6) Levenberg–Marquardt polishing stage
#  7) FEFF amplitude-aware deweighting for high shells (optional)
#  8) ΔE0 pre-scan grid to seed E0
#  9) Shell confidence scores (post-fit diagnostics)
# 10) Bayesian priors (soft penalties) in model selection
# 11) Multi-kweight transforms (kw = (1,2,3) etc.)
# 12) Parallel FEFF exec & parallel feffpath parsing
# 13) Robust dr_shell computation with safety clamps
# 14) Optional pre-fit denoising hooks (SG)
# 15) Automatic SNR-based kmax reduction
#
# Correctness/selection kept:
#  A) Reduced-χ² fallback using DOF = N_idp - N_varys
#  B) Shell confidence uses complex residual magnitude
#  C) Model selection via AICc + small, normalized regularizers
#  D) MS pruning uses RMS amplitude over k-window (stable)
#  E) Auto-bounds expansion when parameters peg their limits (with caps)
#
# NEW in V3 (Shell-by-shell update):
#  S1) Per-shell refinement loop with local R windows (R_j ± r_halfwin)
#  S2) 'shell:j' stage — vary only σ²_j, ΔR_j (and optionally ΔE0 on the first shell)
#  S3) Shell-only dataset builder: only SS+MS of the active shell
#  S4) Flexible shell order: near→far, far→near, or explicit list
#  S5) Per-shell LM polish + same regularizers/auto-bounds as global
# ======================================================================

from __future__ import annotations

import os
import glob
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional, Sequence, Iterable, Union

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from concurrent.futures import ThreadPoolExecutor, as_completed

# Larch imports (must be installed and importable)
from larch import Group
from larch.io import read_ascii
from larch.xafs import (
    feff6l, feff8l, autobk, feffpath, feffit_transform, feffit_dataset,
    feffit, feffit_report
)
from larch.fitting import param, param_group

__all__ = [
    "FeffitAutoShellModel", "ShellParameterManager", "fit",
    "save_fit_params_json", "load_fit_params_json", "seed_from_json",
    "snapshot", "FeffFitFramework",
    "plot_energy_mutrans", "plot_energy_mutrans_from_folder",
]

try:
    from bokeh.io import output_notebook
    output_notebook()
except Exception:
    pass

# ============================================================
# Part 1: Core utilities, numeric helpers, configuration
# ============================================================

def _safe_savgol(y: np.ndarray, preferred_win: int = 11, polyorder: int = 3) -> np.ndarray:
    y = np.asarray(y, float)
    n = y.size
    if n < polyorder + 2:
        return y.copy()
    win = min(preferred_win, n)
    if win % 2 == 0:
        win -= 1
    if win <= polyorder:
        win = polyorder + 1 + int(polyorder % 2 == 0)
        if win > n:
            return y
    return savgol_filter(y, window_length=win, polyorder=polyorder, mode="interp")

def _compute_dr_shell(kmin: float, kmax: float, tighten: float = 0.9,
                      lo: float = 0.08, hi: float = 0.15) -> float:
    dK = max(1e-6, float(kmax) - float(kmin))
    dr = (math.pi / (2.0 * dK)) * float(tighten)
    return float(max(lo, min(hi, dr)))

def _estimate_snr_k(chi_k: np.ndarray, k: np.ndarray, win: int = 17) -> np.ndarray:
    chi_k = np.asarray(chi_k, float)
    k = np.asarray(k, float)
    if chi_k.size != k.size:
        raise ValueError("chi_k and k must have same length")
    smooth = _safe_savgol(chi_k, preferred_win=win, polyorder=3)
    resid = chi_k - smooth
    pad = win // 2
    rp = np.pad(resid, (pad, pad), mode="reflect")
    std = np.empty_like(resid)
    for i in range(resid.size):
        std[i] = np.std(rp[i:i+win])
    std = np.maximum(std, 1e-12)
    return np.abs(smooth) / std

def _auto_trim_kmax(k: np.ndarray, chi_k: np.ndarray, snr_drop: float = 2.0,
                    min_kmax: float = 7.5, guard: float = 0.5) -> float:
    snr = _estimate_snr_k(chi_k, k)
    idx = np.where(snr >= float(snr_drop))[0]
    if idx.size == 0:
        return max(min_kmax, float(k.min()) + 2.0)
    kmax_est = k[idx[-1]]
    kmax_est = max(min_kmax, float(kmax_est) - float(guard))
    return kmax_est

def _exp_k_damp(k: np.ndarray, alpha: float) -> np.ndarray:
    k = np.asarray(k, float)
    a = max(0.0, float(alpha))
    return np.exp(-a * (k ** 2))

def _gaussian_jitter(value: float, scale: float, lo: float, hi: float) -> float:
    v = value
    for _ in range(8):
        v = float(np.random.normal(loc=value, scale=scale))
        if lo <= v <= hi:
            return v
    return float(np.clip(v, lo, hi))

def _estimate_path_amplitude_at_k(path_obj, k_ref: float) -> float:
    try:
        fd = getattr(path_obj, "_feffdat", None)
        if fd is not None and hasattr(fd, "k") and hasattr(fd, "amp"):
            k = np.asarray(fd.k, float)
            amp = np.asarray(fd.amp, float)
            if k.size > 3:
                i = int(np.argmin(np.abs(k - float(k_ref))))
                return float(max(0.0, amp[i]))
    except Exception:
        pass
    try:
        degen = float(getattr(getattr(path_obj, "_feffdat", None), "degen",
                              getattr(path_obj, "degen", 1.0)))
        R = float(getattr(path_obj, "reff", 2.0))
        return float(max(1e-6, degen / (R * R)))
    except Exception:
        return 1e-6

def _cluster_1d_dbscan(values: Iterable[float], eps: float, min_samples: int = 1) -> List[List[int]]:
    vals = np.asarray(list(values), float)
    idx_sorted = np.argsort(vals)
    vals_sorted = vals[idx_sorted]
    clusters_sorted: List[List[int]] = []
    cur: List[int] = []
    for i, v in enumerate(vals_sorted):
        if not cur:
            cur = [int(idx_sorted[i])]
            continue
        v_prev = vals_sorted[i - 1]
        if abs(v - v_prev) <= float(eps):
            cur.append(int(idx_sorted[i]))
        else:
            if len(cur) >= int(min_samples):
                clusters_sorted.append(cur)
            cur = [int(idx_sorted[i])]
    if cur and len(cur) >= int(min_samples):
        clusters_sorted.append(cur)
    clusters_sorted.sort(key=lambda inds: float(np.mean(vals[list(inds)])))
    return clusters_sorted

# ============================================================
# Part 2: Parameter manager with priors and shell-stage
# ============================================================

@dataclass
class PriorSpec:
    """Gaussian prior: value ~ N(mu, sigma) -> penalty = ((value-mu)/sigma)^2"""
    name: str
    mu: float
    sigma: float
    enabled: bool = True

class ShellParameterManager:
    """
    Manage global and per-shell fit parameters.
    - CN-free by default: one global Nscale for all SS shells.
    - Optional per-shell amplitudes: Nscale_j for each SS shell.
    - Optional Gaussian priors; jitter helpers; 'shell:j' stage for local updates.
    """
    def __init__(self, n_shells: int,
                 s02_init: float = 0.9, vary_s02: bool = False,
                 s02_bounds: Tuple[float, float] = (0.5, 1.0),
                 per_shell_nscale: bool = False) -> None:

        self.n_shells = int(n_shells)
        self.per_shell_nscale = bool(per_shell_nscale)

        # Base group with globals
        self.pars = param_group(
            s02=param(s02_init, min=s02_bounds[0], max=s02_bounds[1], vary=bool(vary_s02)),
            del_e0=param(-5.0, min=-25.0, max=25.0, vary=True),
            Nscale=param(0.5, min=0.1, max=1.0, vary=(not self.per_shell_nscale)),
        )
        # If per-shell mode: keep global Nscale fixed at 1 and add Nscale_j
        if self.per_shell_nscale:
            self.pars.Nscale.value = 1.0
            self.pars.Nscale.vary = False
            for j in range(1, self.n_shells + 1):
                setattr(self.pars, f"Nscale_{j}", param(1.0, min=0.5, max=1.5, vary=True))

        # Per-shell structural/disorder params
        for j in range(1, self.n_shells + 1):
            setattr(self.pars, f"sig2_{j}", param(0.001, min=0.0010, max=0.030, vary=True))
            setattr(self.pars, f"delr_{j}", param(0.000, min=-0.100, max=0.090, vary=True))

        # Parameter order (add per-shell amplitudes if enabled)
        self.order: List[str] = ["s02", "del_e0", "Nscale"]
        if self.per_shell_nscale:
            for j in range(1, self.n_shells + 1):
                self.order.append(f"Nscale_{j}")
        for j in range(1, self.n_shells + 1):
            self.order.append(f"sig2_{j}")
        for j in range(1, self.n_shells + 1):
            self.order.append(f"delr_{j}")

        # Priors (extend for per-shell amplitudes if desired)
        self.priors: Dict[str, PriorSpec] = {
            "s02":    PriorSpec("s02",    0.90, 0.05, enabled=False),
            "Nscale": PriorSpec("Nscale", 1.00, 0.25, enabled=False),
        }
        if self.per_shell_nscale:
            for j in range(1, self.n_shells + 1):
                self.priors[f"Nscale_{j}"] = PriorSpec(f"Nscale_{j}", 1.00, 0.25, enabled=False)
        for j in range(1, self.n_shells + 1):
            self.priors[f"sig2_{j}"] = PriorSpec(f"sig2_{j}", 0.005, 0.003, enabled=False)

    def stage(self, name: str) -> None:
        """
        Stage definitions:
          Stage 1 : E0
          Stage 2 : E0, σ²
          Stage 3 : E0, σ², ΔR
          Stage 4 : N (amplitude) — global Nscale or all Nscale_j if per-shell mode
          full    : all vary
          'shell:j': varies only σ²_j and ΔR_j
        """
        def _v(parname: str) -> None:
            p = getattr(self.pars, parname, None)
            if p is not None and hasattr(p, "vary"):
                p.vary = True

        key = (name or "").strip().lower()
        # Reset all vary flags
        for p in self.pars.__dict__.items():
            par = p[1]
            if hasattr(par, "vary"):
                par.vary = False

        if key in ("1", "stage1", "e0"):
            _v("del_e0")

        elif key in ("2", "stage2", "e0_sig2", "sig2"):
            _v("del_e0")
            for j in range(1, self.n_shells + 1):
                _v(f"sig2_{j}")

        elif key in ("3", "stage3", "e0_sig2_delr", "delr"):
            _v("del_e0")
            for j in range(1, self.n_shells + 1):
                _v(f"sig2_{j}")
                _v(f"delr_{j}")

        elif key in ("4", "stage4", "n", "amp"):
            if self.per_shell_nscale:
                for j in range(1, self.n_shells + 1):
                    _v(f"Nscale_{j}")
            else:
                _v("Nscale")

        elif key in ("full",):
            for par in self.pars.__dict__.values():
                if hasattr(par, "vary"):
                    par.vary = True

        elif key.startswith("shell:"):
            try:
                j = int(key.split(":", 1)[1])
            except Exception:
                raise ValueError(f"Bad shell stage specifier: {name!r}. Use 'shell:j' (e.g., 'shell:2').")
            if j < 1 or j > self.n_shells:
                raise ValueError(f"Shell index out of range: {j} (1..{self.n_shells})")
            _v(f"sig2_{j}")
            _v(f"delr_{j}")
            # del_e0 variation is controlled by caller

        else:
            raise ValueError(f"Unknown stage '{name}'")

    def show(self) -> None:
        print("\nCurrent parameters:\n")
        for name, p in sorted(self.pars.__dict__.items()):
            if hasattr(p, "value"):
                bnd = f" [{getattr(p, 'min', '')}, {getattr(p, 'max', '')}]" if hasattr(p, "min") else ""
                print(f"{name:12s}: {p.value: .6f}  vary={getattr(p, 'vary', False)}{bnd}")

    def jitter(self,
               frac: Optional[Dict[str, float]] = None,
               bounds: Optional[Dict[str, Tuple[float, float]]] = None,
               gaussian_scale_frac: float = 0.25) -> None:
        P = self.pars
        frac = frac or {"Nscale": 0.10, "del_e0": 0.30}
        bounds = bounds or {"Nscale": (0.7, 1.0), "del_e0": (-25.0, 25.0)}

        # Global Nscale + del_e0 jitter
        for name in ("Nscale", "del_e0"):
            par = getattr(P, name, None)
            if par is None or not hasattr(par, "value"):
                continue
            v0 = float(par.value)
            sc = abs(v0) * frac.get(name, 0.10)
            lo, hi = bounds.get(name, (getattr(par, "min", -np.inf), getattr(par, "max", np.inf)))
            par.value = _gaussian_jitter(v0, max(1e-6, sc * gaussian_scale_frac), lo, hi)
            if hasattr(par, "init_value"):
                par.init_value = par.value

        # Per-shell amplitudes jitter (if enabled)
        if self.per_shell_nscale:
            for j in range(1, self.n_shells + 1):
                par = getattr(P, f"Nscale_{j}", None)
                if par is not None and hasattr(par, "value"):
                    v0 = float(par.value)
                    sc = abs(v0) * frac.get("Nscale", 0.10)
                    lo = getattr(par, "min", 0.5); hi = getattr(par, "max", 1.5)
                    par.value = _gaussian_jitter(v0, max(1e-6, sc * gaussian_scale_frac), lo, hi)
                    if hasattr(par, "init_value"):
                        par.init_value = par.value

        # Per-shell sig2, delr jitter
        for j in range(1, self.n_shells + 1):
            ps = getattr(P, f"sig2_{j}")
            s0 = float(ps.value)
            s0 = max(1e-5, s0)
            logv = math.log(s0)
            logv = _gaussian_jitter(logv, 0.25, math.log(1e-5), math.log(0.05))
            ps.value = float(np.clip(math.exp(logv), getattr(ps, "min", 1e-6), getattr(ps, "max", 0.05)))
            if hasattr(ps, "init_value"):
                ps.init_value = ps.value

            pr = getattr(P, f"delr_{j}")
            r0 = float(pr.value)
            sc = max(1e-4, 0.15 * max(1e-3, abs(r0)))
            lo = getattr(pr, "min", -0.08); hi = getattr(pr, "max", 0.08)
            pr.value = _gaussian_jitter(r0, sc, lo, hi)
            if hasattr(pr, "init_value"):
                pr.init_value = pr.value

    def prior_penalty(self) -> float:
        pen = 0.0
        P = self.pars
        for name, pr in self.priors.items():
            if not pr.enabled:
                continue
            par = getattr(P, name, None)
            if par is None or not hasattr(par, "value"):
                continue
            pen += ((float(par.value) - float(pr.mu)) / max(1e-9, float(pr.sigma))) ** 2
        return float(pen)

# ============================================================
# Part 3: Main model with per-shell dataset builder
# ============================================================

class FeffitAutoShellModel:
    """
    EXAFS model:
      - SS shells: amplitude ∝ s02 * (Nscale or Nscale_j) * FEFF_degeneracy
      - MS paths: NO Nscale; inherit σ², ΔR from nearest SS shell

    NEW (): shell-by-shell refinement mode using local R windows and only the
    paths that belong to the current shell.
    """

    def __init__(self, datafile: str, feff_dir: str = ".",
                 # windows / grids
                 kmin: Optional[float] = None, kmax: Optional[float] = None,
                 kweights: Optional[Sequence[int]] = None,
                 kweight: Optional[int] = None,
                 rmin: Optional[float] = None, rmax: Optional[float] = None,
                 dk: Optional[float] = 1.0, dr: Optional[float] = None,
                 window: str = "kaiser",
                 # model options
                 max_shells: Optional[int] = None,
                 rbkg: Optional[float] = None,
                 include_ms: bool = True,
                 ms_prune: bool = True,
                 ms_amp_threshold: float = 0.02,
                 ms_r_margin: float = 0.4,
                 # noise & weighting
                 alpha_k_damp: float = 0.0,
                 snr_autocut: bool = True,
                 snr_threshold: float = 2.0,
                 snr_guard: float = 0.5,
                 denoise_preview: bool = True,
                 # runtime
                 auto_cache: bool | str = True,
                 cache_dir: Optional[str] = None,
                 max_nfev: Optional[int] = None,
                 # NEW
                 per_shell_nscale: bool = False) -> None:

        self.datafile = datafile
        self.feff_dir = feff_dir

        self.kmin = 2.5 if kmin is None else float(kmin)
        self.kmax = 10.0 if kmax is None else float(kmax)
        if kweights is not None:
            self.kweights = tuple(int(w) for w in kweights)
        else:
            self.kweights = (int(kweight) if kweight is not None else 2,)
        self.window = str(window).lower()
        self.dk = None if dk is None else float(dk)
        self.dr = None if dr is None else float(dr)

        self.rmin = 1.0 if rmin is None else float(rmin)
        self.rmax = 3.5 if rmax is None else float(rmax)

        self.auto_cache = auto_cache
        self.cache_dir = cache_dir

        self.max_shells = None if max_shells is None else int(max_shells)
        self.rbkg = 1.0 if rbkg is None else float(rbkg)
        self.include_ms = bool(include_ms)
        self.ms_prune = bool(ms_prune)
        self.ms_amp_threshold = float(ms_amp_threshold)
        self.ms_r_margin = float(ms_r_margin)

        self.alpha_k_damp = float(alpha_k_damp)
        self.snr_autocut = bool(snr_autocut)
        self.snr_threshold = float(snr_threshold)
        self.snr_guard = float(snr_guard)
        self.denoise_preview = bool(denoise_preview)

        self.max_nfev = None if max_nfev is None else int(max_nfev)

        # NEW: amplitude mode flag
        self.per_shell_nscale = bool(per_shell_nscale)

        # internals
        self.data: Optional[Group] = None
        self.ss_paths: List[Dict[str, Any]] = []
        self.ms_paths: List[Dict[str, Any]] = []
        self.shells:   List[Dict[str, Any]] = []
        self.ms_to_shell: Dict[str, int] = {}

        self.pars_mgr: Optional[ShellParameterManager] = None
        self.paths = []
        self.trans = None
        self.dset  = None
        self.out   = None

        self._bound_expansion_counts: Dict[str, int] = {}

        # load data
        self._load_data()

        # SNR preview / kmax auto-trim
        if self.snr_autocut:
            self._preview_and_autocut_kmax()

        # dr for clustering
        if self.dr is None:
            self.dr_shell = _compute_dr_shell(self.kmin, self.kmax, tighten=0.9, lo=0.08, hi=0.15)
        else:
            self.dr_shell = float(self.dr)

        # scan paths & build model
        self._scan_paths()
        self._cluster_ss_paths_to_shells_dbscan()
        self._assign_ms_to_shells()
        if self.ms_prune:
            self._prune_ms_paths()
        self._build_parameters()
        self._build_paths()
        self._build_dataset()

    # ---------------- data loading + autobk ----------------
    def _load_data(self):
        try:
            self.data = read_ascii(self.datafile)
        except Exception:
            self.data = None

        if self.data is not None and hasattr(self.data, "mu"):
            pass
        elif self.data is not None and hasattr(self.data, "mutrans"):
            self.data.mu = self.data.mutrans
        else:
            import numpy as _np
            try:
                arr = _np.loadtxt(self.datafile)
            except Exception as e:
                raise RuntimeError(
                    f"Could not load ASCII file '{self.datafile}' via read_ascii or loadtxt: {e}"
                )
            if arr.ndim == 1 or arr.shape[1] < 2:
                raise RuntimeError(
                    f"Headerless file must have at least 2 columns (energy, mutrans). Found shape {arr.shape}"
                )
            g = Group()
            g.energy  = _np.asarray(arr[:, 0], float)
            g.mutrans = _np.asarray(arr[:, 1], float)
            g.mu      = g.mutrans.copy()
            g.table   = arr
            self.data = g

        if self.kmin < 2.0:
            print("[autobk warning] kmin < 2.0 Å⁻¹ includes pre-EXAFS region; fits may distort.")
        autobk(
            self.data,
            kmin=float(self.kmin),
            kmax=float(self.kmax),
            rbkg=float(self.rbkg),
            kweight=int(self.kweights[0]),
        )

    def _preview_and_autocut_kmax(self) -> None:
        d = self.data
        if d is None or not hasattr(d, "k") or not hasattr(d, "chi"):
            return
        k = np.asarray(d.k, float)
        chi = np.asarray(d.chi, float)
        kw = int(self.kweights[0])
        chi_k = (k ** kw) * chi
        if self.denoise_preview:
            chi_k = _safe_savgol(chi_k, preferred_win=17, polyorder=3)
        kmax_new = _auto_trim_kmax(k, chi_k, snr_drop=self.snr_threshold,
                                   min_kmax=max(self.kmin + 4.0, 7.5),
                                   guard=self.snr_guard)
        if kmax_new < self.kmax - 0.25:
            print(f"[SNR autocut] kmax {self.kmax:.2f} → {kmax_new:.2f} (SNR threshold={self.snr_threshold})")
            self.kmax = float(kmax_new)

    # ---------------- FEFF paths ----------------
    def _scan_paths(self) -> None:
        flist = sorted(glob.glob(os.path.join(self.feff_dir, "feff*.dat")))
        if not flist:
            raise RuntimeError(f"No feff*.dat files found in '{self.feff_dir}'")
        self.ss_paths, self.ms_paths = [], []

        def _probe(fpath: str):
            p = feffpath(fpath)
            rec = dict(file=fpath, reff=float(p.reff), nleg=int(p.nleg))
            try:
                rec["degen"] = float(getattr(getattr(p, "_feffdat", None), "degen", getattr(p, "degen", 1.0)))
            except Exception:
                rec["degen"] = 1.0
            return rec

        with ThreadPoolExecutor(max_workers=min(16, max(2, os.cpu_count() or 2))) as ex:
            fut = {ex.submit(_probe, f): f for f in flist}
            for f in as_completed(fut):
                rec = f.result()
                (self.ss_paths if rec["nleg"] == 2 else self.ms_paths).append(rec)

        self.ss_paths.sort(key=lambda r: r["reff"])
        self.ms_paths.sort(key=lambda r: r["reff"])

    def _cluster_ss_paths_to_shells_dbscan(self) -> None:
        ss = [rec for rec in self.ss_paths if (self.rmin <= rec["reff"] <= self.rmax)]
        if not ss:
            raise RuntimeError("No SS paths within selected R range.")
        Reff = np.array([r["reff"] for r in ss], float)
        clusters = _cluster_1d_dbscan(Reff, eps=self.dr_shell, min_samples=1)

        shells: List[Dict[str, Any]] = []
        for idx, cluster in enumerate(clusters, start=1):
            files = [ss[i]["file"] for i in cluster]
            reffs = [ss[i]["reff"] for i in cluster]
            degen_sum = 0.0
            for i in cluster:
                p = feffpath(ss[i]["file"])
                try:
                    degen_sum += float(getattr(getattr(p, "_feffdat", None), "degen", getattr(p, "degen", 1.0)))
                except Exception:
                    degen_sum += 1.0
            shells.append(dict(
                index=idx, files=files, reff=float(np.mean(reffs)),
                feff_cn=float(degen_sum)
            ))

        if self.max_shells is not None:
            shells = shells[:int(self.max_shells)]

        self.shells = shells
        print("Identified SS shells:",
              [(s["index"], round(s["reff"], 3), len(s["files"]), int(round(s["feff_cn"]))) for s in self.shells])

    def _assign_ms_to_shells(self) -> None:
        self.ms_to_shell.clear()
        if not self.include_ms or not self.shells:
            return
        for rec in self.ms_paths:
            j_best, d_best = None, 1e9
            for sh in self.shells:
                d = abs(rec["reff"] - sh["reff"])
                if d < d_best:
                    d_best, j_best = d, sh["index"]
            self.ms_to_shell[rec["file"]] = j_best

    def _prune_ms_paths(self) -> None:
        if not self.include_ms or not self.ms_paths or not self.shells:
            return
        keep: List[Dict[str, Any]] = []
        dropped = 0

        amps = []
        for rec in self.ms_paths:
            p = feffpath(rec["file"])
            fd = getattr(p, "_feffdat", None)
            if fd is not None and hasattr(fd, "k") and hasattr(fd, "amp"):
                k_arr = np.asarray(fd.k, float)
                a_arr = np.asarray(fd.amp, float)
                if k_arr.size >= 4:
                    mask = (k_arr >= self.kmin) & (k_arr <= self.kmax)
                    use = a_arr[mask] if np.any(mask) else a_arr
                    amps.append(float(np.sqrt(np.mean(use**2))))
                    continue
            amps.append(_estimate_path_amplitude_at_k(p, 0.5*(self.kmin + self.kmax)))

        amps = np.asarray(amps, float)
        scale = np.percentile(amps, 95) if amps.size else 1.0
        scale = max(scale, 1e-9)

        for rec, amp in zip(self.ms_paths, amps):
            if rec["reff"] > (self.rmax + self.ms_r_margin):
                dropped += 1
                continue
            rel = (amp / scale)
            if rel < float(self.ms_amp_threshold):
                dropped += 1
                continue
            keep.append(rec)

        print(f"[MS prune] kept {len(keep)} / {len(self.ms_paths)} (dropped {dropped})")
        self.ms_paths = keep

    # ---------------- parameters, paths, dataset ----------------
    def _build_parameters(self) -> None:
        self.pars_mgr = ShellParameterManager(
            n_shells=len(self.shells),
            per_shell_nscale=self.per_shell_nscale
        )
        P = self.pars_mgr.pars
        P.s02.value, P.s02.min, P.s02.max, P.s02.vary = 0.9, 0.5, 1.0, False
        # Global Nscale bounds (kept even if per-shell mode; it is fixed to 1.0 there)
        P.Nscale.min, P.Nscale.max = 0.1, 1.0

    def _build_paths(self) -> None:
        self.paths = []
        for sh in self.shells:
            j = sh["index"]
            for f in sh["files"]:
                p0 = feffpath(f)
                path_degen = float(getattr(getattr(p0, "_feffdat", None), "degen", getattr(p0, "degen", 1.0)))
                amp_expr = f"s02*Nscale_{j}" if self.per_shell_nscale else "s02*Nscale"
                path = feffpath(
                    f,
                    s02=amp_expr,
                    e0="del_e0",
                    sigma2=f"sig2_{j}",
                    deltar=f"delr_{j}",
                    degen=path_degen,
                )
                path.shell_index = j
                self.paths.append(path)

        ms_appended = 0
        if self.include_ms:
            for rec in self.ms_paths:
                j = self.ms_to_shell.get(rec["file"])
                if j is None:
                    continue
                # MS stays WITHOUT Nscale by design
                path = feffpath(
                    rec["file"],
                    s02="s02",
                    e0="del_e0",
                    sigma2=f"sig2_{j}",
                    deltar=f"delr_{j}",
                )
                path.shell_index = j
                self.paths.append(path)
                ms_appended += 1

        ss_count = sum(len(s["files"]) for s in self.shells)
        print(f"Prepared {len(self.paths)} FEFF paths ({ss_count} SS + {ms_appended} MS)")

    def _build_dataset(self) -> None:
        tkwargs = dict(
            kmin=float(self.kmin), kmax=float(self.kmax),
            kw=self.kweights if len(self.kweights) > 1 else int(self.kweights[0]),
            rmin=float(self.rmin), rmax=float(self.rmax),
            window=self.window
        )
        if self.dk is not None:
            tkwargs["dk"] = float(self.dk)
        if self.dr is not None:
            tkwargs["dr"] = float(self.dr)

        self.trans = feffit_transform(**tkwargs)
        self.dset = feffit_dataset(self.data, self.paths, self.trans)

    # NEW (): shell-scoped dataset builder
    def _build_dataset_for_shell(self, j: int, r_halfwin: float = 0.12,
                                 include_ms: bool = True):
        if j < 1 or j > len(self.shells):
            raise ValueError(f"Shell index out of range: {j} (1..{len(self.shells)})")
        sh = self.shells[j-1]
        # Use fitted R if available, else Reff
        P = self.pars_mgr.pars
        delrj = getattr(P, f"delr_{j}", None)
        R_center = float(sh["reff"]) + (float(delrj.value) if (delrj is not None and hasattr(delrj, "value")) else 0.0)
        rmin_local = max(0.0, R_center - float(r_halfwin))
        rmax_local = R_center + float(r_halfwin)

        shell_paths = [p for p in self.paths if getattr(p, "shell_index", None) == j and (p.nleg == 2)]
        if include_ms:
            shell_paths += [p for p in self.paths if getattr(p, "shell_index", None) == j and (p.nleg > 2)]
        if not shell_paths:
            raise RuntimeError(f"No FEFF paths found for shell {j} for shell-local dataset.")

        tkwargs = dict(
            kmin=float(self.kmin), kmax=float(self.kmax),
            kw=self.kweights if len(self.kweights) > 1 else int(self.kweights[0]),
            rmin=float(rmin_local), rmax=float(rmax_local),
            window=self.window
        )
        if self.dk is not None:
            tkwargs["dk"] = float(self.dk)
        # Use global dr for transform (Fourier smoothing)
        if self.dr is not None:
            tkwargs["dr"] = float(self.dr)

        trans_local = feffit_transform(**tkwargs)
        dset_local = feffit_dataset(self.data, shell_paths, trans_local)
        return trans_local, dset_local, dict(r_center=R_center, rmin=rmin_local, rmax=rmax_local)

    def rebuild_after_ms_toggle(self, verbose: bool = True) -> None:
        self._assign_ms_to_shells()
        if self.ms_prune:
            self._prune_ms_paths()
        self._build_parameters()
        self._build_paths()
        self._build_dataset()
        if verbose:
            print(f"[rebuild] include_ms={self.include_ms} → dataset rebuilt.")

# ============================================================
# Part 4: Metrics, E0 pre-scan, info criteria
# ============================================================

    @staticmethod
    def _get_metrics(out):
        if out is None:
            return None, None
        rfac = getattr(out, "rfactor", None) or getattr(out, "r_factor", None)
        redc = (getattr(out, "redchi", None)
                or getattr(out, "chi2_reduced", None)
                or getattr(out, "chisqr_reduced", None))
        if redc is None:
            try:
                chisqr = getattr(out, "chisqr", None)
                nvarys = getattr(out, "nvarys", None)
                n_idp = (getattr(out, "n_idp", None)
                         or getattr(out, "nind", None)
                         or getattr(out, "ndata", None)
                         or getattr(out, "npts", None))
                if (chisqr is not None) and (nvarys is not None) and (n_idp is not None):
                    dof = float(n_idp) - float(nvarys)
                    if dof > 0:
                        redc = float(chisqr) / float(dof)
            except Exception:
                redc = None
        return (float(rfac) if rfac is not None else None,
                float(redc) if redc is not None else None)

    def _prime_init_values(self) -> None:
        if self.pars_mgr is None:
            return
        for par in self.pars_mgr.pars.__dict__.values():
            if hasattr(par, "value"):
                v = float(par.value)
                par.value = v
                if hasattr(par, "init_value"):
                    par.init_value = v

    def _count_varying_params(self) -> int:
        if self.pars_mgr is None:
            return 0
        n = 0
        for p in self.pars_mgr.pars.__dict__.values():
            if hasattr(p, "vary") and bool(getattr(p, "vary")):
                n += 1
        return n

    def _suggest_max_nfev(self, stage: str, nvar: int) -> int:
        base = 600 * max(1, (nvar + 1))
        mult = {"amp": 0.7, "e0": 0.5, "delr": 1.0, "sig2": 1.0, "N": 0.9, "full": 2.0}.get(stage, 1.0)
        est = int(base * mult)
        return max(500, min(est, 120_000))

    def _feffit_with_method(self, max_nfev: int, method: Optional[str], *, dset=None) -> Any:
        params = self.pars_mgr.pars
        dataset = dset if dset is not None else self.dset
        try:
            if method is not None:
                return feffit(params, dataset, max_nfev=int(max_nfev), method=method)
            return feffit(params, dataset, max_nfev=int(max_nfev))
        except TypeError:
            return feffit(params, dataset, max_nfev=int(max_nfev))

    def pre_scan_e0(self, grid: Tuple[float, float, float] = (-15.0, 15.0, 0.5),
                    verbose: bool = True) -> float:
        if self.pars_mgr is None:
            return 0.0
        lo, hi, step = [float(x) for x in grid]
        vals = np.arange(lo, hi + 0.25 * step, step, dtype=float)

        P = self.pars_mgr.pars
        saved = {k: (v.value, getattr(v, "vary", False)) for k, v in P.__dict__.items() if hasattr(v, "value")}
        try:
            for p in P.__dict__.values():
                if hasattr(p, "vary"):
                    p.vary = False
            if hasattr(P, "del_e0"):
                P.del_e0.vary = True

            best_val, best_metric = None, np.inf
            for v in vals:
                P.del_e0.value = float(v)
                self._prime_init_values()
                out = self._feffit_with_method(max_nfev=150, method="nelder")
                rfac, redc = self._get_metrics(out)
                metric = (redc if (redc is not None and np.isfinite(redc)) else (rfac if rfac is not None else np.inf))
                if metric < best_metric:
                    best_metric = metric
                    best_val = float(v)
            if best_val is not None:
                P.del_e0.value = best_val
                if verbose:
                    print(f"[E0 pre-scan] Seed del_e0 = {best_val:.3f}")
            return float(P.del_e0.value)
        finally:
            for k, (val, vary) in saved.items():
                par = getattr(P, k, None)
                if par is not None and hasattr(par, "value"):
                    par.value = float(val)
                    if hasattr(par, "init_value"):
                        par.init_value = float(val)
                    if hasattr(par, "vary"):
                        par.vary = bool(vary)

    def _info_criteria(self, out) -> Tuple[Optional[float], Optional[float]]:
        try:
            n = (getattr(out, "n_idp", None)
                 or getattr(out, "ndata", None)
                 or getattr(out, "npts", None))
            k = getattr(out, "nvarys", None)
            chisqr = getattr(out, "chisqr", None)
            if (n is None) or (k is None) or (chisqr is None):
                return None, None
            n = float(n); k = float(k); chisqr = float(chisqr)
            mse = max(chisqr / max(n, 1.0), 1e-300)
            aic  = 2.0 * k + n * math.log(mse)
            aicc = aic + (2.0 * k * (k + 1.0)) / max(n - k - 1.0, 1.0)
            bic  = (k * math.log(max(n, 1.0))) + n * math.log(mse)
            return aicc, bic
        except Exception:
            return None, None

# ============================================================
# Part 5: Fit controller — global + per-shell
# ============================================================

    def _post_metric(self, out, *, priors_enabled: bool) -> Tuple[float, float, float, float]:
        prior_pen = self.pars_mgr.prior_penalty() if priors_enabled else 0.0
        try:
            d = self.dset.data
            m = self.dset.model
            k = np.asarray(d.k, float)
            kw = int(self.kweights[0])
            resid = (k ** kw) * (np.asarray(d.chi, float) - np.asarray(m.chi, float))
            if self.alpha_k_damp > 0:
                resid = _exp_k_damp(k, self.alpha_k_damp) * resid
            data_kw = (k ** kw) * np.asarray(d.chi, float)
            denom = float(np.mean(data_kw**2)) if np.isfinite(np.mean(data_kw**2)) and np.mean(data_kw**2) > 0 else 1.0
            damp_norm = float(np.mean(resid**2)) / denom
        except Exception:
            damp_norm = 1.0

        aicc, _bic = self._info_criteria(out)
        _, redc = self._get_metrics(out)
        base_score = (aicc if (aicc is not None and np.isfinite(aicc)) else (redc if redc is not None else 1.0))

        LAMBDA_PRIOR = 1e-2
        LAMBDA_DAMP  = 1e-1
        combined = float(base_score + LAMBDA_PRIOR * prior_pen + LAMBDA_DAMP * damp_norm)
        return (prior_pen, damp_norm, (aicc if aicc is not None else np.nan), combined)

    def _sync_live_pars_from_out(self) -> None:
        if self.out is None or not hasattr(self.out, "params"):
            return
        Pout = self.out.params
        Plive = self.pars_mgr.pars
        for name in self.pars_mgr.order:
            pl = getattr(Plive, name, None)
            po = Pout.get(name) if hasattr(Pout, "get") else getattr(Pout, name, None)
            if pl is None or po is None or not hasattr(pl, "value"):
                continue
            val = float(po.value)
            pl.value = val
            if hasattr(pl, "init_value"):
                pl.init_value = val
            if hasattr(pl, "vary"):
                pl.vary = bool(getattr(po, "vary", getattr(pl, "vary", False)))

    def _param_near_bound(self, par, tol: float = 0.02) -> Optional[str]:
        if par is None or not hasattr(par, "value"):
            return None
        v = float(par.value)
        lo = getattr(par, "min", None)
        hi = getattr(par, "max", None)
        if lo is None or hi is None or not np.isfinite(lo) or not np.isfinite(hi):
            return None
        span = float(hi - lo)
        if span <= 0:
            return None
        if (v - lo) <= tol * span:
            return "min"
        if (hi - v) <= tol * span:
            return "max"
        return None

    def _expand_bounds_for_param(self, name: str, par, hit: str, bound_max_expansions: int) -> bool:
        if par is None:
            return False
        cnt = self._bound_expansion_counts.get(name, 0)
        if cnt >= int(bound_max_expansions):
            return False
        lo = getattr(par, "min", None)
        hi = getattr(par, "max", None)
        if (lo is None) or (hi is None) or not np.isfinite(lo) or not np.isfinite(hi):
            return False
        changed = False

        def _bump_sym(lo0, hi0, factor, cap_abs=None):
            span = max(abs(lo0), abs(hi0))
            new_span = span * float(factor)
            if cap_abs is not None:
                new_span = min(new_span, float(cap_abs))
            return -abs(new_span), +abs(new_span)

        nm = str(name).lower()
        if nm.startswith("delr_"):
            new_lo, new_hi = _bump_sym(lo, hi, factor=1.5, cap_abs=0.25)
            if new_lo < lo or new_hi > hi:
                par.min, par.max = float(new_lo), float(new_hi)
                changed = True
        elif nm.startswith("sig2_"):
            new_lo = max(1e-5, float(lo))
            new_hi = float(hi)
            if hit == "min":
                new_lo = max(1e-5, float(lo) * 0.5)
            elif hit == "max":
                new_hi = min(0.08, float(hi) * 1.5)
            if (new_lo < lo) or (new_hi > hi):
                par.min, par.max = float(new_lo), float(new_hi)
                changed = True
        # elif nm == "nscale":
        #     new_lo = max(0.2, float(lo) * (0.8 if hit == "min" else 1.0))
        #     new_hi = min(2.5, float(hi) * (1.2 if hit == "max" else 1.0))
        #     if (new_lo < lo) or (new_hi > hi):
        #         par.min, par.max = float(new_lo), float(new_hi)
        #         changed = False
        # elif nm == "s02":
        #     new_lo = max(0.4, float(lo) * (0.9 if hit == "min" else 1.0))
        #     new_hi = min(1.2, float(hi) * (1.1 if hit == "max" else 1.0))
        #     if (new_lo < lo) or (new_hi > hi):
        #         par.min, par.max = float(new_lo), float(new_hi)
        #         changed = False
        elif nm == "del_e0":
            new_lo = min(-80.0, float(lo)) if hit == "min" else float(lo)
            new_hi = max( 80.0, float(hi)) if hit == "max" else float(hi)
            if (new_lo != lo) or (new_hi != hi):
                par.min, par.max = float(new_lo), float(new_hi)
                changed = True

        if changed:
            self._bound_expansion_counts[name] = cnt + 1
        return changed

    def _auto_expand_on_bound_hits(self, tol: float, bound_max_expansions: int,
                                   *, verbose_local: bool = True) -> bool:
        if self.pars_mgr is None:
            return False
        P = self.pars_mgr.pars
        widened = False
        msgs = []
        for name in self.pars_mgr.order:
            par = getattr(P, name, None)
            if par is None or not getattr(par, "vary", False):
                continue
            hit = self._param_near_bound(par, tol=tol)
            if hit is None:
                continue
            before = (getattr(par, "min", None), getattr(par, "max", None))
            if self._expand_bounds_for_param(name, par, hit, bound_max_expansions):
                after = (getattr(par, "min", None), getattr(par, "max", None))
                widened = True
                msgs.append(f"  - {name}: hit={hit}  bounds {before} → {after} (count={self._bound_expansion_counts.get(name)})")
        if widened and verbose_local and msgs:
            print("[auto-bounds] Expanded bounds for:")
            for m in msgs:
                print(m)
        return widened

    # --------------- Per-shell loop (NEW) -----------------
    def fit_each_shell(self,
        *,
        r_halfwin: float = 0.12,
        include_ms_per_shell: bool = True,
        order: Union[str, Sequence[int]] = "near-to-far",
        vary_e0_on_first_shell: bool = True,
        lm_polish_each: bool = True,
        priors_enabled: bool = True,
        method: Optional[str] = None,
        methods_try: Optional[Sequence[Optional[str]]] = None,
        verbose: bool = True,
        bound_hit_tol: float = 0.02,
        bound_max_expansions: int = 3,
    ) -> None:
        """
        Shell-by-shell refinement:
          - local dataset: R in [R_j - r_halfwin, R_j + r_halfwin]
          - vary only σ²_j and ΔR_j (and optionally ΔE0 for first shell)
          - keep all other params fixed
          - write back, then move to next shell
        """
        if self.pars_mgr is None or not self.shells:
            raise RuntimeError("Parameters or shells not initialized.")

        # Determine order
        shells_idx = [sh["index"] for sh in self.shells]
        if isinstance(order, str):
            if order.lower().startswith("near"):
                ord_seq = shells_idx
            elif order.lower().startswith("far"):
                ord_seq = list(reversed(shells_idx))
            else:
                raise ValueError("order must be 'near-to-far', 'far-to-near', or a sequence of shell indices.")
        else:
            ord_seq = list(order)
            for j in ord_seq:
                if j not in shells_idx:
                    raise ValueError(f"Shell index {j} not present in model shells {shells_idx}.")

        # Default method cascade
        if methods_try is None:
            if (method or "").lower() == "nelder":
                methods_try = ("nelder", None)
            elif (method or "").lower() == "brute":
                methods_try = ("brute", None)
            else:
                methods_try = (method, "leastsq", None)

        P = self.pars_mgr.pars
        saved = {k: (v.value, getattr(v, "vary", False), getattr(v, "expr", None))
                 for k, v in P.__dict__.items() if hasattr(v, "value")}

        try:
            for j in ord_seq:
                if verbose:
                    print(f"\n===== Per-shell refinement: shell {j} =====")
                self.pars_mgr.stage(f"shell:{j}")
                if vary_e0_on_first_shell and (j == ord_seq[0]):
                    P.del_e0.vary = True
                else:
                    P.del_e0.vary = False
                # Never adjust amplitude scale during per-shell pass
                P.Nscale.vary = False
                # Rebuild shell-local dataset
                trans_local, dset_local, meta = self._build_dataset_for_shell(
                    j, r_halfwin=float(r_halfwin), include_ms=bool(include_ms_per_shell)
                )

                nvar = self._count_varying_params()
                stage_limit = self._suggest_max_nfev("shell", nvar)

                out_local = None
                for meth in methods_try:
                    out_local = self._feffit_with_method(stage_limit, meth, dset=dset_local)
                    self.out = out_local
                    self._sync_live_pars_from_out()
                    widened = self._auto_expand_on_bound_hits(
                        tol=bound_hit_tol,
                        bound_max_expansions=int(bound_max_expansions),
                        verbose_local=verbose)
                    if widened:
                        out_local = self._feffit_with_method(stage_limit, methods_try[0], dset=dset_local)
                        self.out = out_local
                        self._sync_live_pars_from_out()
                    break

                if lm_polish_each:
                    try:
                        self.pars_mgr.stage(f"shell:{j}")
                        if vary_e0_on_first_shell and (j == ord_seq[0]):
                            P.del_e0.vary = True
                        else:
                            P.del_e0.vary = False
                        out_local = self._feffit_with_method(max_nfev=4000, method="leastsq", dset=dset_local)
                        self.out = out_local
                        self._sync_live_pars_from_out()
                    except Exception as e:
                        if verbose:
                            print(f"[LM polish shell {j}] skipped: {e}")

                rfac, redc = self._get_metrics(out_local)
                if verbose:
                    print(f" [shell {j}] r-factor={rfac}, red-chi={redc}, window=({meta['rmin']:.3f},{meta['rmax']:.3f})")

        finally:
            for k, (_, vary, expr) in saved.items():
                par = getattr(P, k, None)
                if par is not None:
                    if hasattr(par, "vary"):
                        par.vary = bool(vary)
                    if hasattr(par, "expr"):
                        par.expr = expr

    # --------------- Global fit controller -----------------
    def fit(self, *,
            # global vs per-shell
            per_shell: bool = False,
            per_shell_r_halfwin: float = 0.25,
            per_shell_include_ms: bool = True,
            per_shell_order: Union[str, Sequence[int]] = "near-to-far",
            per_shell_vary_e0_on_first: bool = False,
            per_shell_lm_polish_each: bool = True,
            # convergence cycling (global)
            max_cycles: int = 100,
            staged: Tuple[str, ...] = ("amp", "e0", "delr", "sig2", "N", "full"),
            use_stages: bool = True,
            # convergence checks
            rfactor_tol: Optional[float] = 1e-6,
            redchi_tol: Optional[float] = 1e-15,
            metric_atol: float = 1e-15,
            required_hits: int = 3,
            # restarts
            restarts: int = 1,
            jitter_first: bool = True,
            jitter_frac: Optional[Dict[str, float]] = None,
            jitter_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
            # optimizer choice
            method: Optional[str] = None,
            methods_try: Optional[Sequence[Optional[str]]] = None,
            # extras
            do_e0_prescan: bool = False,
            lm_polish: bool = True,
            priors_enabled: bool = True,
            verbose: bool = True,
            stage_out: bool = True,
            # auto-bounds expansion controls
            auto_expand_bounds: bool = True,
            bound_hit_tol: float = 0.02,
            bound_max_expansions: int = 3,
            rerun_stage_on_expand: bool = True,
            ) -> Any:

        # Enable priors if requested
        if priors_enabled and self.pars_mgr is not None:
            for pr in self.pars_mgr.priors.values():
                pr.enabled = True

        # Optional global pre-scan for E0
        if do_e0_prescan:
            try:
                self.pre_scan_e0(verbose=verbose)
            except Exception as e:
                if verbose:
                    print(f"[E0 pre-scan] skipped: {e}")

        # Per-shell pass (NEW)
        if per_shell:
            self.fit_each_shell(
                r_halfwin=float(per_shell_r_halfwin),
                include_ms_per_shell=bool(per_shell_include_ms),
                order=per_shell_order,
                vary_e0_on_first_shell=bool(per_shell_vary_e0_on_first),
                lm_polish_each=bool(per_shell_lm_polish_each),
                priors_enabled=bool(priors_enabled),
                method=method,
                methods_try=methods_try,
                verbose=verbose,
                bound_hit_tol=float(bound_hit_tol),
                bound_max_expansions=int(bound_max_expansions),
            )
            # After per-shell refinement, rebuild global dataset (paths unchanged; params updated)
            self._build_dataset()
            if verbose:
                print("\n[per-shell] Completed shell-by-shell refinement. Proceeding to global fit polish...")

        # Global staged fitting
        if methods_try is None:
            if (method or "").lower() == "nelder":
                methods_try = ("nelder", None)
                restarts = max(restarts, 2)
                jitter_first = True
                if jitter_frac is None:
                    jitter_frac = {"Nscale": 0.06, "del_e0": 0.12}
            elif (method or "").lower() == "brute":
                methods_try = ("brute", None)
                restarts = max(restarts, 2)
                jitter_first = True
                if jitter_frac is None:
                    jitter_frac = {"Nscale": 0.06, "del_e0": 0.12}
            else:
                methods_try = (method, "leastsq", None)
                restarts = max(restarts, 2)
                jitter_first = True
                if jitter_frac is None:
                    jitter_frac = {"Nscale": 0.06, "del_e0": 0.12}

        P = self.pars_mgr.pars

        def _snapshot_params():
            return {k: float(p.value) for k, p in P.__dict__.items() if hasattr(p, "value")}
        def _restore_params(snap: Dict[str, float]):
            for k, v in snap.items():
                par = getattr(P, k, None)
                if par is not None and hasattr(par, "value"):
                    par.value = float(v)
                    if hasattr(par, "init_value"):
                        par.init_value = float(v)

        original_seeds = _snapshot_params()

        best = {"out": None, "rfac": np.inf, "redc": np.inf, "params": None,
                "prior_pen": np.inf, "damped_metric": np.inf, "aicc": np.inf}

        for irun in range(max(1, int(restarts))):
            if verbose:
                print(f"\n==== Global Restart {irun+1}/{restarts} ====")

            _restore_params(original_seeds)
            self._prime_init_values()

            if (irun > 0) or jitter_first:
                self.pars_mgr.jitter(frac=jitter_frac or {}, bounds=jitter_bounds or {})
                self._prime_init_values()

            out = None
            prev_full_rfac = None
            prev_full_redc = None
            metric_stable_hits = 0
            REQUIRED_PARAM_HITS = int(required_hits)
            METRIC_ATOL = float(metric_atol)
            user_max = int(self.max_nfev) if self.max_nfev is not None else None

            for ic in range(1, int(max_cycles) + 1):
                if verbose:
                    print(f"\n=== Cycle {ic} ===")
                stop_fit = False
                stages = staged if use_stages else ("full",)

                for stage in stages:
                    self.pars_mgr.stage(stage)
                    if stage_out:
                        print(f"\n--- Stage: {stage} ---")

                    self._prime_init_values()
                    nvar = self._count_varying_params()
                    stage_limit = user_max if user_max is not None else self._suggest_max_nfev(stage, nvar)

                    for meth in methods_try:
                        attempt = 0
                        while True:
                            attempt += 1
                            if verbose:
                                print(f"[fit] stage={stage} nvar={nvar} max_nfev={stage_limit} attempt={attempt} method={meth}")
                            out = self._feffit_with_method(stage_limit, meth)
                            self.out = out
                            self._sync_live_pars_from_out()

                            hit_limit = False
                            message = (getattr(out, "message", "") or "").lower()
                            nfev = getattr(out, "nfev", None)
                            if "maxfev" in message or "max iterations" in message:
                                hit_limit = True
                            if (nfev is not None) and (int(nfev) >= int(stage_limit)):
                                hit_limit = True
                            if not hit_limit or attempt >= 3:
                                break
                            stage_limit = int(stage_limit * 1.8)
                        break

                    rfac, redc = self._get_metrics(out)
                    if stage_out:
                        print(f" r-factor           = {rfac if rfac is not None else 'None'}")
                        print(f" reduced chi-square = {redc if redc is not None else 'None'}")

                    if auto_expand_bounds:
                        widened = self._auto_expand_on_bound_hits(
                            tol=float(bound_hit_tol),
                            bound_max_expansions=int(bound_max_expansions),
                            verbose_local=stage_out
                        )
                        if widened and rerun_stage_on_expand:
                            try:
                                self._prime_init_values()
                                out = self._feffit_with_method(stage_limit, methods_try[0])
                                self.out = out
                                self._sync_live_pars_from_out()
                                rfac, redc = self._get_metrics(out)
                                if stage_out:
                                    print(" [auto-bounds] Re-ran stage after widening.")
                                    print(f"  r-factor           = {rfac if rfac is not None else 'None'}")
                                    print(f"  reduced chi-square = {redc if redc is not None else 'None'}")
                            except Exception as e:
                                if stage_out:
                                    print(f" [auto-bounds] Re-run skipped: {e}")

                    if stage == "full":
                        if (rfactor_tol is not None and redchi_tol is not None
                                and rfac is not None and redc is not None):
                            if (rfac <= rfactor_tol) and (redc <= redchi_tol):
                                if verbose:
                                    print(f"Converged by tolerance on 'full' (r-factor={rfac:.6g}, red-chi={redc:.6g}).")
                                stop_fit = True
                        if not stop_fit and REQUIRED_PARAM_HITS > 0:
                            if (prev_full_rfac is not None and prev_full_redc is not None
                                    and rfac is not None and redc is not None):
                                same_r = abs(rfac - prev_full_rfac) <= METRIC_ATOL
                                same_c = abs(redc - prev_full_redc) <= METRIC_ATOL
                                if same_r and same_c:
                                    metric_stable_hits += 1
                                    if verbose:
                                        print(f"Metric stability hit {metric_stable_hits}/{REQUIRED_PARAM_HITS} "
                                              f"(Δr={abs(rfac - prev_full_rfac):.2e}, Δredχ²={abs(redc - prev_full_redc):.2e}).")
                                    if metric_stable_hits >= REQUIRED_PARAM_HITS:
                                        if verbose:
                                            print("Converged by 'full' metric stability.")
                                        stop_fit = True
                                else:
                                    if metric_stable_hits > 0 and verbose:
                                        print("Metric stability broken; resetting counter.")
                                    metric_stable_hits = 0
                        prev_full_rfac = rfac
                        prev_full_redc = redc
                        if stop_fit:
                            break

                    self._prime_init_values()

                if stop_fit:
                    break

            if lm_polish:
                try:
                    self.pars_mgr.stage("full")
                    self._prime_init_values()
                    out = self._feffit_with_method(max_nfev=4000, method="leastsq")
                    self.out = out
                    self._sync_live_pars_from_out()
                except Exception as e:
                    if verbose:
                        print(f"[LM polish] skipped: {e}")

            prior_pen, damp_norm, aicc_val, combined = self._post_metric(out, priors_enabled=priors_enabled)
            rfac, redc = self._get_metrics(out)
            if combined < best["damped_metric"]:
                best.update(dict(out=out, rfac=(rfac or np.inf), redc=(redc or np.inf),
                                 params=_snapshot_params(), prior_pen=prior_pen,
                                 damped_metric=combined, aicc=aicc_val))

        if best["params"] is not None:
            for k, v in best["params"].items():
                par = getattr(self.pars_mgr.pars, k, None)
                if par is not None and hasattr(par, "value"):
                    par.value = float(v)
                    if hasattr(par, "init_value"):
                        par.init_value = float(v)
        self.out = best["out"]

        if verbose:
            rfac, redc = self._get_metrics(self.out)
            print(f"[best] r-factor={rfac}, red-chi={redc}, AICc={best.get('aicc', np.nan)}, "
                  f"prior-pen={best['prior_pen']:.4g}, score={best['damped_metric']:.4g}")

        if self.auto_cache:
            try:
                if isinstance(self.auto_cache, str):
                    self.save_plot_cache(self.auto_cache if self.auto_cache.lower().endswith(".npz")
                                         else f"{self.auto_cache}.npz")
                else:
                    self.save_plot_cache(self._default_cache_path())
            except Exception as e:
                if verbose:
                    print(f"[auto-cache] Skipped saving cache: {e}")
        return self.out

    def _default_cache_path(self) -> str:
        stem = os.path.splitext(os.path.basename(self.datafile))[0]
        kw_tag = "x".join(str(int(w)) for w in self.kweights)
        tag = f"{stem}_k{int(self.kmin)}-{int(self.kmax)}_R{self.rmin:.2f}-{self.rmax:.2f}_kw{kw_tag}"
        fname = f"{tag}.npz"
        folder = self.cache_dir if self.cache_dir else os.path.dirname(os.path.abspath(self.datafile))
        return os.path.join(folder, fname)

# ============================================================
# Part 6: Reports, shell confidence, plots, caching
# ============================================================

    def report(self, filename: Optional[str] = None) -> str:
        if self.out is None:
            raise RuntimeError("Run fit() first.")
        rep = feffit_report(self.out)
        if filename:
            with open(filename, "w") as f:
                f.write(rep)
        return rep

    def print_shell_summary(self) -> None:
        if self.out is None:
            raise RuntimeError("Run fit() first.")
        P = self.out.params
        def fmt(par, nd=5):
            if par is None:
                return " --- "
            if getattr(par, "stderr", None) is not None:
                return f"{par.value:.{nd}f} ± {par.stderr:.{nd}f}"
            return f"{par.value:.{nd}f} (±unknown)"

        print("\nShell parameter summary (CN derived from amplitude × FEFF_CN):")
        print("Idx   R_exp(Å)      ΔR(Å)             R_fit(Å)            σ²(Å²)         CN_derived")
        print("--------------------------------------------------------------------------------------")
        for sh in self.shells:
            j = sh["index"]
            R_exp = float(sh["reff"])
            delr = P.get(f"delr_{j}")
            sig2 = P.get(f"sig2_{j}")
            R_fit_val = R_exp + (delr.value if delr is not None else 0.0)
            if delr is not None and getattr(delr, "stderr", None) is not None:
                R_fit_str = f"{R_fit_val:.5f} ± {delr.stderr:.5f}"
            else:
                R_fit_str = f"{R_fit_val:.5f} (±unknown)"

            if getattr(self.pars_mgr, "per_shell_nscale", False):
                amp_par = P.get(f"Nscale_{j}")
                nscale_here = float(amp_par.value) if amp_par is not None else 1.0
            else:
                amp_par = P.get("Nscale")
                nscale_here = float(amp_par.value) if amp_par is not None else 1.0

            cn_derived = sh["feff_cn"] * nscale_here
            print(f"{j:3d}  {R_exp:8.5f}  {fmt(delr):>12s}   {R_fit_str:>14s}  {fmt(sig2):>16s}    {cn_derived:8.3f}")

        # Per-shell CN diagnostic lines
        for sh in self.shells:
            j = sh["index"]
            feff_cn = float(sh["feff_cn"])
            if getattr(self.pars_mgr, "per_shell_nscale", False):
                nscale_here = float(P.get(f"Nscale_{j}").value)
            else:
                nscale_here = float(P.get("Nscale").value)
            cn_derived = feff_cn * nscale_here
            ratio = cn_derived / max(feff_cn, 1e-12)
            print(f"Shell {j}: FEFF_CN={feff_cn}, CN_derived={cn_derived:.3f}, ratio={ratio:.3f}")

    def shell_confidence(self, r_halfwin: float = 0.12) -> List[Dict[str, Any]]:
        if self.dset is None or self.out is None:
            raise RuntimeError("Run fit() first.")
        d = self.dset.data; m = self.dset.model
        r = np.asarray(getattr(d, "r", []), float)
        if r.size == 0:
            raise RuntimeError("R grid unavailable for shell confidence.")
        out: List[Dict[str, Any]] = []
        for sh in self.shells:
            j = sh["index"]
            P = self.out.params
            R_fit = float(sh["reff"]) + float(getattr(P.get(f"delr_{j}"), "value", 0.0))
            mask = (r >= (R_fit - r_halfwin)) & (r <= (R_fit + r_halfwin))
            if not np.any(mask):
                mask = (r >= (R_fit - 2*r_halfwin)) & (r <= (R_fit + 2*r_halfwin))
            d_re = np.asarray(getattr(d, "chir_re", 0.0))[mask]
            d_im = np.asarray(getattr(d, "chir_im", 0.0))[mask]
            m_re = np.asarray(getattr(m, "chir_re", 0.0))[mask]
            m_im = np.asarray(getattr(m, "chir_im", 0.0))[mask]

            data_mag  = float(np.mean(np.abs(d_re + 1j*d_im)))
            model_mag = float(np.mean(np.abs(m_re + 1j*m_im)))
            resid_mag = float(np.mean(np.abs((d_re - m_re) + 1j*(d_im - m_im))))
            score = model_mag / (resid_mag + 1e-12)
            out.append(dict(index=j, R_fit=R_fit, data_mag=data_mag,
                            model_mag=model_mag, resid_mag=resid_mag,
                            conf_score=score))
        out.sort(key=lambda dct: dct["index"])
        return out

    def _build_plot_cache_dict(self) -> dict:
        if self.dset is None:
            raise RuntimeError("Dataset not built. Run fit() first.")
        d = self.dset.data
        m = self.dset.model
        k = np.asarray(d.k, dtype=float)
        kw_primary = int(self.kweights[0])
        chi_k_data_kw = (k ** kw_primary) * np.asarray(d.chi, dtype=float)
        chi_k_fit_kw  = (k ** kw_primary) * np.asarray(m.chi, dtype=float)
        r = np.asarray(getattr(d, "r", []), dtype=float)

        def _get_arr(obj, name):
            arr = getattr(obj, name, None)
            return np.asarray(arr, dtype=float) if arr is not None else np.array([], dtype=float)

        cache = {
            "k": k,
            "chi_k_data_kw": chi_k_data_kw,
            "chi_k_fit_kw":  chi_k_fit_kw,
            "r": r,
            "chir_mag_data": _get_arr(d, "chir_mag"),
            "chir_mag_fit":  _get_arr(m, "chir_mag"),
            "chir_re_data":  _get_arr(d, "chir_re"),
            "chir_im_data":  _get_arr(d, "chir_im"),
            "chir_re_fit":   _get_arr(m, "chir_re"),
            "chir_im_fit":   _get_arr(m, "chir_im"),
            "meta_json": json.dumps({
                "datafile": os.path.abspath(self.datafile) if hasattr(self, "datafile") else None,
                "feff_dir": os.path.abspath(self.feff_dir) if hasattr(self, "feff_dir") else None,
                "kmin": float(self.kmin), "kmax": float(self.kmax),
                "kweights": list(self.kweights),
                "window": self.window,
                "rmin": float(self.rmin), "rmax": float(self.rmax),
                "timestamp": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
            }),
        }
        return cache

    def save_plot_cache(self, filename: str) -> str:
        if self.dset is None:
            raise RuntimeError("Nothing to save: dataset not built. Run fit() first.")
        cache = self._build_plot_cache_dict()
        if not filename.lower().endswith(".npz"):
            filename = f"{filename}.npz"
        abspath = os.path.abspath(filename)
        np.savez_compressed(abspath, **cache)
        print(f"[save_plot_cache] Saved plot bundle → {abspath}")
        return abspath

    def load_plot_cache(self, filename: str) -> dict:
        abspath = os.path.abspath(filename)
        if not os.path.isfile(abspath):
            raise FileNotFoundError(f"[load_plot_cache] File not found: {abspath}")
        z = np.load(abspath, allow_pickle=False)
        cache = {k: z[k] for k in z.files}
        print(f"[load_plot_cache] Loaded plot bundle ← {abspath}")
        return cache

    def plot_k(self, kmax: Optional[float] = None, smooth: bool = True,
               smooth_win: int = 11, smooth_poly: int = 3,
               *, cache: Optional[dict] = None, cache_file: Optional[str] = None) -> None:
        k, y_data, y_fit, meta = self._get_k_plot_arrays(cache=cache, cache_file=cache_file)
        if smooth:
            y_data = _safe_savgol(y_data, preferred_win=smooth_win, polyorder=smooth_poly)
        kw = int(meta.get("kweights", [self.kweights[0]])[0])
        _kmax_default = float(meta.get("kmax", getattr(self, "kmax", np.nan)))

        if self.alpha_k_damp > 0:
            damp = _exp_k_damp(k, self.alpha_k_damp)
            y_data = damp * y_data
            y_fit  = damp * y_fit

        plt.figure(figsize=(7, 4))
        plt.plot(k, y_data, label="data")
        plt.plot(k, y_fit,  label="fit")
        plt.xlabel(r"$k\ (\mathrm{\AA}^{-1})$")
        plt.ylabel(rf"$k^{kw}\chi(k)$")
        plt.legend()
        plt.xlim(0, float(kmax) if kmax is not None else _kmax_default)
        plt.tight_layout()
        plt.show()

    def _get_k_plot_arrays(self, cache: Optional[dict] = None, cache_file: Optional[str] = None):
        if cache is not None or cache_file is not None:
            if cache is None:
                cache = self.load_plot_cache(cache_file)
            k = np.asarray(cache["k"], dtype=float)
            data_kw = np.asarray(cache["chi_k_data_kw"], dtype=float)
            fit_kw  = np.asarray(cache["chi_k_fit_kw"],  dtype=float)
            meta = json.loads(str(cache.get("meta_json", "{}")))
            return k, data_kw, fit_kw, meta

        if self.dset is None:
            raise RuntimeError("Dataset not built. Run fit() first or provide cache.")
        d = self.dset.data; m = self.dset.model
        k = np.asarray(d.k, dtype=float)
        kw = int(self.kweights[0])
        data_kw = (k ** kw) * np.asarray(d.chi, dtype=float)
        fit_kw  = (k ** kw) * np.asarray(m.chi, dtype=float)
        meta = {"kmin": float(self.kmin), "kmax": float(self.kmax), "kweights": list(self.kweights)}
        return k, data_kw, fit_kw, meta

    def plot_r(self, rmax: Optional[float] = None,
               *, cache: Optional[dict] = None, cache_file: Optional[str] = None) -> None:
        r, y_data, y_fit, meta = self._get_r_plot_arrays(cache=cache, cache_file=cache_file)
        _rmax_default = float(meta.get("rmax", getattr(self, "rmax", np.nan)))

        plt.figure(figsize=(7, 4))
        plt.plot(r, y_data, label="data")
        plt.plot(r, y_fit,  label="fit")
        plt.xlabel(r"$R$ ($\mathrm{\AA}$)")
        plt.ylabel(r"$|\chi(R)|$")
        plt.legend()
        plt.xlim(0, float(rmax) if rmax is not None else _rmax_default)
        plt.tight_layout()
        plt.show()

    def _get_r_plot_arrays(self, cache: Optional[dict] = None, cache_file: Optional[str] = None):
        if cache is not None or cache_file is not None:
            if cache is None:
                cache = self.load_plot_cache(cache_file)
            r = np.asarray(cache["r"], dtype=float)
            if r.size == 0:
                raise RuntimeError("R-space arrays not available in cache.")
            data = np.asarray(cache["chir_mag_data"], dtype=float)
            fit  = np.asarray(cache["chir_mag_fit"],  dtype=float)
            meta = json.loads(str(cache.get("meta_json", "{}")))
            return r, data, fit, meta

        if self.dset is None:
            raise RuntimeError("Dataset not built. Run fit() first or provide cache.")
        d = self.dset.data; m = self.dset.model
        r = np.asarray(getattr(d, "r", None), dtype=float)
        if r.size == 0:
            raise RuntimeError("R grid unavailable.")
        data = np.asarray(getattr(d, "chir_mag", None), dtype=float)
        fit  = np.asarray(getattr(m, "chir_mag", None), dtype=float)
        if data is None or fit is None:
            raise RuntimeError("Dataset missing chir_mag arrays.")
        meta = {"rmin": float(self.rmin), "rmax": float(self.rmax)}
        return r, data, fit, meta

# ============================================================
# Part 7: JSON checkpoint helpers & FEFF framework (parallel)
# ============================================================

def save_fit_params_json(model: FeffitAutoShellModel, filename: str, save_vary: bool = False):
    model._sync_live_pars_from_out()
    P = model.pars_mgr.pars
    payload = {}
    for name, par in P.__dict__.items():
        if hasattr(par, "value"):
            entry = {"value": float(par.value)}
            if save_vary and hasattr(par, "vary"):
                entry["vary"] = bool(getattr(par, "vary", False))
            payload[name] = entry
    with open(filename, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved fitted values → {filename} (save_vary={save_vary})")

def load_fit_params_json(
    model: FeffitAutoShellModel,
    filename: str,
    *,
    strict: bool = False,
    apply_vary: str = "from_json",   # "from_json" | "keep_current" | "stage"
    stage_name: Optional[str] = None
) -> None:
    abspath = os.path.abspath(filename)
    if not os.path.isfile(abspath):
        raise FileNotFoundError(f"[load_fit_params_json] File not found: {abspath}")
    with open(abspath, "r") as f:
        payload = json.load(f)

    P = model.pars_mgr.pars
    updated = 0; missing = []; changes = []
    for name, info in payload.items():
        par = getattr(P, name, None)
        if par is None:
            msg = f"[load_fit_params_json] Warning: parameter '{name}' not in model."
            if strict:
                raise AttributeError(msg)
            else:
                print(msg); missing.append(name); continue
        old_val = float(getattr(par, "value", np.nan))
        new_val = float(info.get("value", old_val))
        par.value = new_val
        if hasattr(par, "init_value"): par.init_value = new_val
        vary_json = info.get("vary", getattr(par, "vary", False))
        if hasattr(par, "vary"):
            if apply_vary == "from_json": par.vary = bool(vary_json)
            elif apply_vary == "keep_current": pass
            elif apply_vary == "stage": pass
        updated += 1
        changes.append((name, old_val, new_val, getattr(par, "vary", None)))

    if apply_vary == "stage":
        if not stage_name:
            raise ValueError("apply_vary='stage' requires stage_name (e.g., 'full', 'e0', 'sig2', 'delr', 'N', 'amp').")
        model.pars_mgr.stage(stage_name)

    print(f"[load_fit_params_json] Updated {updated} parameters from '{abspath}'.")
    if missing:
        preview = ", ".join(missing[:8]); tail = " ..." if len(missing) > 8 else ""
        print(f"[load_fit_params_json] Skipped {len(missing)} name(s): {preview}{tail}")

    key_names = ["s02", "Nscale", "del_e0"]
    key_names += [f"Nscale_{j}" for j in range(1, len(model.shells)+1)]  # NEW: include per-shell amps
    key_names += [f"sig2_{j}" for j in range(1, len(model.shells)+1)]
    key_names += [f"delr_{j}" for j in range(1, len(model.shells)+1)]
    print("[load_fit_params_json] Key seeds diff:")
    for nm, ov, nv, vr in changes:
        if nm in key_names and (ov != nv):
            try:
                print(f"  {nm:10s}: {ov:.6g} -> {nv:.6g}  (vary={vr})")
            except Exception:
                print(f"  {nm:10s}: {ov} -> {nv}  (vary={vr})")

def seed_from_json(
    model: FeffitAutoShellModel,
    filename: str,
    *,
    strict: bool = False,
    show_after: bool = True,
    apply_vary: str = "from_json",
    stage_name: Optional[str] = None
) -> None:
    load_fit_params_json(model, filename, strict=strict, apply_vary=apply_vary, stage_name=stage_name)
    model._prime_init_values()
    if show_after:
        print("\n[seed_from_json] Live parameters after loading:")
        model.pars_mgr.show()

def snapshot(model: FeffitAutoShellModel, title: str = "snapshot") -> None:
    print(f"\n[{title}]")
    P = model.pars_mgr.pars
    keys = ["s02", "Nscale", "del_e0"]
    keys += [f"Nscale_{j}" for j in range(1, len(model.shells)+1)]  # NEW: include per-shell amps
    keys += [f"sig2_{j}" for j in range(1, len(model.shells)+1)]
    keys += [f"delr_{j}" for j in range(1, len(model.shells)+1)]
    for k in keys:
        par = getattr(P, k, None)
        if par is not None and hasattr(par, "value"):
            print(f"  {k:10s} = {par.value}  (init={getattr(par,'init_value', None)}, vary={getattr(par,'vary', None)})")

# ---- Parallel FEFF runner (unchanged in spirit) ----
class FeffFitFramework:
    def __init__(self, feff_folder: str, feff_input: str = "feff.inp", use_feff8: bool = True):
        self.feff_folder = feff_folder
        self.feff_input = feff_input
        self.use_feff8 = use_feff8
        if not os.path.isdir(self.feff_folder):
            raise FileNotFoundError(f"FEFF folder not found: {self.feff_folder}")

    def run_feff(self, verbose: bool = True) -> None:
        runner = feff8l if self.use_feff8 else feff6l
        runner(folder=self.feff_folder, feffinp=self.feff_input, verbose=verbose)

    @staticmethod
    def run_many(folders: Sequence[str], feff_input: str = "feff.inp",
                 use_feff8: bool = True, max_workers: Optional[int] = None,
                 verbose_each: bool = True) -> None:
        runner = feff8l if use_feff8 else feff6l
        max_workers = max_workers or min(16, max(2, os.cpu_count() or 2))

        def _task(folder: str):
            if not os.path.isdir(folder):
                return (folder, "missing")
            try:
                runner(folder=folder, feffinp=feff_input, verbose=verbose_each)
                return (folder, "ok")
            except Exception as e:
                return (folder, f"error: {e}")

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(_task, f): f for f in folders}
            results = []
            for f in as_completed(futs):
                results.append(f.result())
        print("[feff-run-many]", results)

# ============================================================
# Part 8: One-line API (fit) + plotting utilities
# ============================================================

def fit(
    datafile: str,
    feff_dir: str,
    *,
    preset: str = "oxide",
    # windows
    kmin=None, kmax=None, kweights: Optional[Sequence[int]] = None, kweight: Optional[int] = None,
    dk=None, dr=None, rmin=None, rmax=None, window: str = "kaiser",
    # model + constraints
    max_shells: Optional[int] = None,
    rbkg: Optional[float] = None,
    include_ms: Optional[bool] = True,
    plots: bool = True,
    verbose: bool = True,
    max_nfev: Optional[int] = None,
    auto_cache: bool | str = True,
    cache_dir: Optional[str] = None,
    # shell constraints (explicit control)
    high_shell_policy: str = "tie-to-2",  # "none" | "fix" | "tie-to-1" | "tie-to-2" | "loose-bounds"
    constraints_start_shell: Optional[int] = None,
    # per-shell options (NEW)
    per_shell: bool = True,
    per_shell_r_halfwin: float = 0.25,
    per_shell_include_ms: bool = True,
    per_shell_order: Union[str, Sequence[int]] = "near-to-far",
    per_shell_vary_e0_on_first: bool = False,
    per_shell_lm_polish_each: bool = True,
    # global restarts:
    restarts: int = 1,
    jitter_first: bool = True,
    jitter_frac: Optional[Dict[str, float]] = None,
    jitter_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    # optimizer choice:
    method: Optional[str] = None,
    methods_try: Optional[Sequence[Optional[str]]] = None,
    # extras:
    alpha_k_damp: float = 0.0,
    snr_autocut: bool = True,
    snr_threshold: float = 2.0,
    snr_guard: float = 0.5,
    ms_prune: bool = True,
    ms_amp_threshold: float = 0.02,
    ms_r_margin: float = 0.4,
    do_e0_prescan: bool = False,
    lm_polish: bool = True,
    priors_enabled: bool = True,
    # NEW:
    per_shell_nscale: bool = False,
):
    """
    One-call EXAFS fit () with per-shell refinement followed by global polish.
    Set per_shell_nscale=True to use per-shell amplitudes Nscale_j (otherwise global Nscale).
    """
    model = FeffitAutoShellModel(
        datafile=datafile,
        feff_dir=feff_dir,
        kmin=kmin, kmax=kmax,
        kweights=kweights, kweight=kweight,
        dk=dk, dr=dr, rmin=rmin, rmax=rmax,
        window=window,
        max_shells=max_shells, rbkg=rbkg,
        include_ms=(include_ms if include_ms is not None else True),
        ms_prune=ms_prune, ms_amp_threshold=ms_amp_threshold, ms_r_margin=ms_r_margin,
        alpha_k_damp=alpha_k_damp,
        snr_autocut=snr_autocut, snr_threshold=snr_threshold, snr_guard=snr_guard,
        auto_cache=auto_cache, cache_dir=cache_dir,
        max_nfev=max_nfev,
        per_shell_nscale=bool(per_shell_nscale),  # NEW
    )

    # presets
    pr = (preset or "quick").lower().strip()
    if pr == "oxide":
        defaults = dict(kmin=2.0, kmax=12.0, kweights=model.kweights, rmin=1.0, rmax=4.0, rbkg=1.2, include_ms=True)
    elif pr == "metal":
        defaults = dict(kmin=3.0, kmax=12.0, kweights=model.kweights, rmin=1.8, rmax=4.0, rbkg=1.2, include_ms=True)
    elif pr == "quick":
        defaults = dict(kmin=model.kmin, kmax=model.kmax, kweights=model.kweights,
                        rmin=model.rmin, rmax=model.rmax, rbkg=model.rbkg, include_ms=model.include_ms)
    else:
        raise ValueError("Unknown preset. Use: 'oxide', 'metal', or 'quick'.")

    for key in ("kmin", "kmax", "rmin", "rmax"):
        if getattr(model, key) is None:
            setattr(model, key, defaults[key])

    if getattr(model, "rbkg", None) is None:
        model.rbkg = defaults["rbkg"]
    if include_ms is None:
        model.include_ms = bool(defaults.get("include_ms", True))

    # autobk refresh
    autobk(model.data,
           kmin=float(model.kmin),
           kmax=float(model.kmax),
           rbkg=float(model.rbkg),
           kweight=int(model.kweights[0]))
    model.dr_shell = _compute_dr_shell(model.kmin, model.kmax, tighten=0.9, lo=0.08, hi=0.15)
    model._scan_paths()
    model._cluster_ss_paths_to_shells_dbscan()
    model._assign_ms_to_shells()
    if model.ms_prune:
        model._prune_ms_paths()
    model._build_parameters()
    model._build_paths()
    model._build_dataset()

    # Optional MS override
    if include_ms is not None and include_ms != model.include_ms:
        model.include_ms = bool(include_ms)
        model.rebuild_after_ms_toggle()

    # constraints_start_shell: compute from Nyquist if None
    if constraints_start_shell is None:
        dK = float(model.kmax) - float(model.kmin)
        dR = float(model.rmax) - float(model.rmin)
        n_indep = int(np.floor((2.0 * dK * dR) / np.pi))
        max_shells_free = max(1, (n_indep - 2) // 2)  # globals ~ (ΔE0, Nscale)
        constraints_start_shell = max_shells_free + 1

    _apply_shell_constraints(
        model, high_shell_policy=high_shell_policy,
        start_shell=int(constraints_start_shell), verbose=verbose
    )

    # Per-shell + global fit
    out = model.fit(
        per_shell=bool(per_shell),
        per_shell_r_halfwin=float(per_shell_r_halfwin),
        per_shell_include_ms=bool(per_shell_include_ms),
        per_shell_order=per_shell_order,
        per_shell_vary_e0_on_first=bool(per_shell_vary_e0_on_first),
        per_shell_lm_polish_each=bool(per_shell_lm_polish_each),

        staged=("amp", "e0", "delr", "sig2", "N", "full"),
        rfactor_tol=1e-6, redchi_tol=1e-15, metric_atol=1e-15, required_hits=3,
        verbose=verbose, stage_out=verbose,
        restarts=restarts, jitter_first=jitter_first,
        jitter_frac=jitter_frac, jitter_bounds=jitter_bounds,
        method=method, methods_try=methods_try,
        do_e0_prescan=do_e0_prescan,
        lm_polish=lm_polish,
        priors_enabled=priors_enabled,
        auto_expand_bounds=True,
        bound_hit_tol=0.02,
        bound_max_expansions=3,
        rerun_stage_on_expand=True,
    )

    if plots:
        try:
            model.plot_k()
            model.plot_r()
        except Exception as e:
            if verbose:
                print(f"[plot] Skipped plots: {e}")
    return model

# ---- SS-shell constraint policy
def _apply_shell_constraints(
    model: FeffitAutoShellModel,
    high_shell_policy: str = "none",
    start_shell: int = 3,
    verbose: bool = True,
) -> None:
    """
    Apply constraints to higher SS shells (index >= start_shell).
    Policies:
      'none'        : do nothing
      'fix'         : freeze σ² & ΔR on higher shells
      'tie-to-1'    : tie to shell 1
      'tie-to-2'    : tie to shell 2 (fallback to 1)
      'loose-bounds': relax bounds on higher shells
    """
    if model.pars_mgr is None or not model.shells:
        if verbose:
            print("[constraints] Skipped: parameters or shells not initialized.")
        return
    P = model.pars_mgr.pars
    nshell = len(model.shells)
    policy = (high_shell_policy or "none").lower().strip()
    start_shell = max(2, int(start_shell))
    start_shell = min(start_shell, nshell)

    if policy in ("", "none", "no", "off"):
        if verbose:
            print("[constraints] policy='none' → no SS-shell constraints applied.")
        return
    if nshell < 2 and policy.startswith("tie"):
        if verbose:
            print("[constraints] Only one shell; skipping tie constraints.")
        return

    def _ensure_bounds(par):
        if getattr(par, "min", None) is None: par.min = -np.inf
        if getattr(par, "max", None) is None: par.max =  np.inf

    def _tie_param(dst_par, ref_name: str):
        if dst_par is None: return
        if getattr(dst_par, "name", None) == ref_name: return
        _ensure_bounds(dst_par)
        dst_par.expr = ref_name
        dst_par.vary = False

    if policy in ("tie-to-1", "tie_to_1"):
        ref = 1
        eff_start = max(start_shell, ref + 1)
        for j in range(eff_start, nshell + 1):
            _tie_param(getattr(P, f"sig2_{j}", None), f"sig2_{ref}")
            _tie_param(getattr(P, f"delr_{j}", None), f"delr_{ref}")
    elif policy in ("tie-to-2", "tie_to_2"):
        ref = 2 if nshell >= 2 else 1
        eff_start = max(start_shell, ref + 1)
        for j in range(eff_start, nshell + 1):
            _tie_param(getattr(P, f"sig2_{j}", None), f"sig2_{ref}")
            _tie_param(getattr(P, f"delr_{j}", None), f"delr_{ref}")
    elif policy == "fix":
        for j in range(start_shell, nshell + 1):
            for nm in (f"sig2_{j}", f"delr_{j}"):
                par = getattr(P, nm, None)
                if par is not None:
                    _ensure_bounds(par)
                    par.expr = None
                    par.vary = False
    elif policy == "loose-bounds":
        for j in range(start_shell, nshell + 1):
            sig = getattr(P, f"sig2_{j}", None)
            dr = getattr(P, f"delr_{j}", None)
            if sig is not None:
                sig.expr = None
                sig.min, sig.max = 0.002, 0.020
                sig.vary = True
            if dr is not None:
                dr.expr = None
                dr.min, dr.max = -0.020, 0.020
                dr.vary = True
    else:
        raise ValueError("Unknown high_shell_policy. Use: 'none', 'fix', 'tie-to-1', 'tie-to-2', or 'loose-bounds'.")

# ----------------------------------------------------------------------
# Simple energy–μtrans loader & interactive plotter (unchanged)
# ----------------------------------------------------------------------

def plot_energy_mutrans(
    files,
    *,
    stacked: bool = False,
    offset_step: float | None = None,
    title: str | None = None,
    xlabel: str = "Energy (eV)",
    ylabel: str = "μ (transmission)",
    legend: bool = True,
    figsize=(9, 6),
    savepath: str | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    nan_policy: str = "drop",
    interactive: bool = True,
):
    def _safe_name(p: str) -> str:
        return os.path.splitext(os.path.basename(p))[0]
    def _auto_color(i: int) -> str:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        return colors[i % len(colors)]
    def _read_matrix(path: str) -> np.ndarray:
        try:
            arr = np.genfromtxt(path, comments="#", invalid_raise=False)
        except Exception:
            arr = np.loadtxt(path, comments="#", ndmin=2)
        if arr.ndim == 1:
            arr = arr[:, None]
        return arr
    def _iter_pairs(arr: np.ndarray):
        ncol = arr.shape[1]
        last_even = (ncol // 2) * 2
        for k in range(0, last_even, 2):
            x = arr[:, k]; y = arr[:, k + 1]
            yield x, y, (k // 2) + 1

    if not files:
        raise ValueError("No files provided.")

    all_traces = []
    y_spans = []
    for path in files:
        if not os.path.isfile(path):
            print(f"[plot_energy_mutrans] Warning: file not found, skipping → {path}")
            continue
        arr = _read_matrix(path)
        if arr.shape[1] < 2:
            print(f"[plot_energy_mutrans] Warning: '{path}' has < 2 columns; skipping.")
            continue
        base = _safe_name(path)
        for x, y, pair_no in _iter_pairs(arr):
            if nan_policy == "drop":
                mask = np.isfinite(x) & np.isfinite(y)
                x = x[mask]; y = y[mask]
            if x.size == 0 or y.size == 0:
                continue
            label = base if pair_no == 1 else f"{base} [pair {pair_no}]"
            all_traces.append({"label": label, "x": x, "y": y})
            y_spans.append(float(np.nanmax(y) - np.nanmin(y)))

    if len(all_traces) == 0:
        raise RuntimeError("No plottable traces found (check files/columns).")

    if offset_step is None:
        offset_step = 0.08 * max(y_spans) if (stacked and len(y_spans) > 0) else 0.0

    fig, ax = plt.subplots(figsize=figsize)
    plotted_lines = []
    for i, tr in enumerate(all_traces):
        x, y, label = tr["x"], tr["y"], tr["label"]
        off = (i * offset_step) if stacked else 0.0
        ln, = ax.plot(x, y + off, lw=1.8, color=_auto_color(i), label=label)
        plotted_lines.append(ln)

    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if xlim: ax.set_xlim(*xlim)
    if ylim: ax.set_ylim(*ylim)
    ax.grid(alpha=0.2)

    leg = None
    if legend:
        leg = ax.legend(fontsize=9, ncol=1)

    plt.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=200)
        print(f"[plot_energy_mutrans] Saved figure → {os.path.abspath(savepath)}")

    if interactive:
        for ln in plotted_lines:
            ln.set_picker(5)

        label_to_line = {ln.get_label(): ln for ln in plotted_lines}
        legend_entry_to_label = {}
        if leg is not None:
            for legline, legtext in zip(leg.get_lines(), leg.get_texts()):
                legline.set_picker(True); legtext.set_picker(True)
                legend_entry_to_label[legline] = legtext.get_text()
                legend_entry_to_label[legtext] = legtext.get_text()

        history: List[Dict[int, bool]] = []

        def _snapshot():
            return {id(ln): ln.get_visible() for ln in plotted_lines}
        def _push_history():
            history.append(_snapshot())
            if len(history) > 100:
                history.pop(0)
        def _sync_legend_for_line(line):
            if leg is None:
                return
            lab = line.get_label()
            for legline, legtext in zip(leg.get_lines(), leg.get_texts()):
                if legtext.get_text() == lab:
                    alpha = 1.0 if line.get_visible() else 0.25
                    legline.set_alpha(alpha); legtext.set_alpha(alpha)
        def _sync_all_legends():
            if leg is None:
                return
            for ln in plotted_lines:
                _sync_legend_for_line(ln)
        def _toggle_line(line):
            _push_history()
            line.set_visible(not line.get_visible())
            _sync_legend_for_line(line)
            fig.canvas.draw_idle()
        def _on_pick(event):
            artist = event.artist
            if artist in plotted_lines:
                _toggle_line(artist); return
            if artist in legend_entry_to_label:
                lab = legend_entry_to_label[artist]
                line = label_to_line.get(lab)
                if line is not None:
                    _toggle_line(line)
        def _on_key(event):
            if event.key in ("ctrl+z", "cmd+z"):
                if history:
                    state = history.pop()
                    for ln in plotted_lines:
                        if id(ln) in state:
                            ln.set_visible(state[id(ln)])
                    _sync_all_legends()
                    fig.canvas.draw_idle()

        _sync_all_legends()
        fig.canvas.mpl_connect("pick_event", _on_pick)
        fig.canvas.mpl_connect("key_press_event", _on_key)
        print("[interactive] Click a line or legend entry to toggle visibility. Press Ctrl+Z to undo.")

    plt.show()
    return fig, ax

def plot_energy_mutrans_from_folder(folder: str, pattern: str = "*.dat", **plot_kwargs):
    import glob as _glob
    files = sorted(_glob.glob(os.path.join(folder, pattern)))
    if not files:
        raise FileNotFoundError(f"No files found in {folder!r} matching {pattern!r}")
    return plot_energy_mutrans(files, **plot_kwargs)
