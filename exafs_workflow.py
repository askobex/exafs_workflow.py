# ---------------------------------------------------------
# exafs_workflow.py — cleaned core (compute once, store, write-only export)
# ---------------------------------------------------------
from __future__ import annotations

# Standard library
import glob
import math
import os
import re
import fnmatch
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

# Third-party
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from larch import Group
from larch.xafs import autobk, cauchy_wavelet, pre_edge, xftf, xftr
from scipy.signal import savgol_filter
from scipy.interpolate import PchipInterpolator as PchipInterpolator # shape-preserving
# =========================================================
# EXAFS All-in-One (clean core)
# =========================================================

class EXAFSAll:
    """
    Unified EXAFS workflow (clean core):
      • Read text data and headers (raw & cleaned).
      • Evaluate fluorescence channels, select "good", sum per side.
      • Build μ(E) (from chosen side), pre-edge, background, FT.
      • Store per-scan aligned extras: InB/OutB sum_good, I0/I1/I2.
      • Build per-group (summed) arrays once; edge-trim; store.
      • Export one XDI-like .dat per group (write-only; preserves header blocks).
    """

    def __init__(self):
        self.groups: Dict[str, Group] = {}
        self.e0: Dict[str, float] = {}
        self.exafs: Dict[str, Dict[str, Any]] = {}
        self.metrics: Dict[str, pd.DataFrame] = {}
        self.metrics_all: Optional[pd.DataFrame] = None
        self._channel_eval: Dict[str, Dict[str, Any]] = {}
        self._aligned_data: Dict[str, Dict[str, Any]] = {}
        self._sum_map: Dict[str, List[str]] = {}
        self._reference: Optional[str] = None
        self._label_map: Dict[str, str] = {}
        self._label_display_map: Dict[str, str] = {}
        self._ka1_token: str = "Ka1_spectra"
        self._ft_params: Dict[str, Any] = {}
        self._default_kweight: int = None
        
    def print_e0_values(self):
        """
        Print E0 values for all loaded labels.
        """
        if not hasattr(self, "e0") or not self.e0:
            print("No E0 values found. Run align_multiple() first.")
            return

        for label, e0 in self.e0.items():
            print(f"{label}: E0 = {e0:.4f} eV")
    # ------------------- Data Loader -------------------
    class XASDataLoader:
        """
        Loader for whitespace-delimited XAS text files with '#' headers.
    
        Keeps:
          - header: cleaned lines (legacy)
          - header_raw: raw lines with tabs/spacing preserved (for ROI/Scanned Regions)
        """
    
        def __init__(self, filename: str):
            self.filename = filename
            self.header: List[str] = []
            self.header_raw: List[str] = []
            self.data: Optional[pd.DataFrame] = None
            self.columns: List[str] = []
    
        def load(self) -> "EXAFSAll.XASDataLoader":
            # Read header block
            with open(self.filename, "r", encoding="utf-8", errors="replace", newline="") as f:
                for line in f:
                    if line.startswith("#"):
                        self.header_raw.append(line.rstrip("\n"))  # exact, minus trailing newline
                        self.header.append(line.strip())           # cleaned
                    else:
                        break
    
            # More tolerant CSV parsing: python engine + whitespace delimiter
            self.data = pd.read_csv(
                self.filename,
                comment="#",
                #delim_whitespace=True,   # or 
                sep=r"\s+",
                header=None,
                engine="python",
                skip_blank_lines=True,
                # on_bad_lines="skip",   # uncomment temporarily if diagnosing malformed lines
            )
    
            if self.data is None or self.data.empty:
                raise ValueError(f"No numeric data found in '{self.filename}'.")
    
            # Parse and assign column names
            self.columns = self._parse_column_names()
            n_data_cols = self.data.shape[1]
    
            if self.columns:
                # If header lists fewer names than actual columns, pad with generic names
                if len(self.columns) < n_data_cols:
                    self.columns += [f"col_{i+1}" for i in range(len(self.columns), n_data_cols)]
                # If header lists more names, truncate
                if len(self.columns) > n_data_cols:
                    self.columns = self.columns[:n_data_cols]
                self.data.columns = self.columns
            else:
                self.data.columns = [f"col_{i+1}" for i in range(n_data_cols)]
    
            return self



        def _parse_column_names(self) -> List[str]:
            # Parse "# Column.N: Name" and sort by N to ensure correct order
            import re
            pairs = []
            for line in self.header:
                m = re.match(r"#\s*Column\.(\d+)\s*:\s*(.+)", line)
                if m:
                    idx = int(m.group(1))
                    name = m.group(2).strip()
                    pairs.append((idx, name))
            if not pairs:
                return []
            pairs.sort(key=lambda t: t[0])
            return [name for _, name in pairs]
    
        def get(self, name: str) -> np.ndarray:
            if name not in self.data.columns:
                raise KeyError(
                    f"Column '{name}' not found. Available (first few): {list(self.data.columns)[:6]}..."
                )
            # Coerce to float, surfacing NaNs for any non-numeric cells instead of crashing
            arr = pd.to_numeric(self.data[name], errors="coerce").to_numpy(dtype=float)
            return arr
    # ------------------- μ(E) basics -------------------
    @staticmethod
    def transmission(I0: Iterable[float], I2: Iterable[float]) -> np.ndarray:
        """μ(E) = ln(I0 / I2)"""
        return np.log(np.asarray(I0, float) / np.asarray(I1, float))

    @staticmethod
    def fluorescence(If: Iterable[float], I0: Iterable[float]) -> np.ndarray:
        """If/I0"""
        return np.asarray(If, float) / np.asarray(I0, float)

    @staticmethod
    def _safe_savgol(y: Iterable[float], preferred_win: int = 11, polyorder: int = 3) -> np.ndarray:
        """Savitzky–Golay with short-array/NaN guards."""
        y = np.asarray(y, float)
        n = np.isfinite(y).sum()
        if n < polyorder + 2:
            return y
        x = np.arange(len(y))
        mask = np.isfinite(y)
        if not mask.all():
            y = y.copy()
            y[~mask] = np.interp(x[~mask], x[mask], y[mask])
        win = min(preferred_win, len(y))
        if win % 2 == 0:
            win -= 1
        if win < polyorder + 2:
            win = polyorder + 3 if (polyorder + 2) % 2 == 0 else polyorder + 2
        if win > len(y):
            return y
        return savgol_filter(y, window_length=win, polyorder=polyorder, mode="interp")

    def _determine_e0_first_derivative(
        self, energy: Iterable[float], mu: Iterable[float], smooth: bool = True, window: int = 9, polyorder: int = 3
    ) -> float:
        """Estimate E0 from the max of the first derivative of μ(E)."""
        energy = np.asarray(energy, float)
        mu = np.asarray(mu, float)
        mask = np.isfinite(energy) & np.isfinite(mu)
        if mask.sum() < 5:
            return float(np.nanmedian(energy))
        x, y = energy[mask], mu[mask]
        if smooth:
            n = len(x)
            win = min(window if window % 2 == 1 else window + 1, n if n % 2 == 1 else n - 1)
            if win < polyorder + 2:
                win = polyorder + 3 if (polyorder + 2) % 2 == 0 else polyorder + 2
            if win >= max(5, polyorder + 2):
                y = savgol_filter(y, win, polyorder, mode="interp")
        d1 = np.gradient(y, x)
        if not np.any(np.isfinite(d1)):
            return float(np.nanmedian(x))
        return float(x[int(np.nanargmax(d1))])

    # ------------------- Fluorescence channel evaluation -------------------
    
    def evaluate_fluo_channels(
        self,
        energy: Iterable[float],
        data: pd.DataFrame,
        channels: List[str],
        pre_window: Tuple[float, float] = (-150, -30),
        post_window: Tuple[float, float] = (0, 250),
        min_signal: float = 1e-6,
        # --- legacy thresholds (diagnostics only unless use_legacy_gates=True) ---
        min_edge_jump: float = 0.12,
        min_corr: float = 0.6,
        # --- BASIC gates (the ones that decide accept/reject) ---
        use_basic_gates_only: bool = True,
        max_rms: float = 1.0e-1,        # pre-edge robust RMS
        max_hf_ratio: float = 100.0,     # post/pre residual noise inflation
        min_snr_post: float = 100.0,      # edge_jump / rms_post_resid
        # --- optional post-edge/legacy gates (OFF unless enabled) ---
        use_legacy_gates: bool = False,     # edge_jump + corr
        use_post_checks: bool = False,      # residual-based post-edge checks
        max_post_slope: float = 1.0e-3,
        max_rms_post_resid: float = 2.5e-3,
        min_corr_post: float = 0.55,
    ) -> Tuple[List[str], pd.DataFrame, float]:
        """
        Score fluorescence channels and select 'good' ones.
    
        Status decision is based on:
          - rms (pre-edge robust noise)
          - hf_ratio (post/pre high-frequency noise inflation)
          - snr_post (edge jump vs post-edge residual noise)
    
        Other metrics are reported for diagnostics.
        """
        def mad_to_rms(arr: np.ndarray) -> float:
            """Robust RMS estimate via MAD scaled for Gaussian (1.4826 * MAD)."""
            if arr.size == 0:
                return np.nan
            med = np.nanmedian(arr)
            mad = np.nanmedian(np.abs(arr - med))
            return float(1.4826 * mad)
    
        energy = np.asarray(energy, float)
        if len(energy) != len(data):
            raise ValueError("Length of energy and data must match.")
    
        # diag_cols = [
        #     "edge_jump", "rms", "corr",
        #     "rms_post_resid", "corr_post", "post_slope",
        #     "snr_post", "hf_ratio", "status"
        # ]
        diag_cols = [
             "rms", "corr",
            "snr_post", "status"
        ]
        if not channels:
            return [], pd.DataFrame(columns=diag_cols), float(np.nanmedian(energy))
    
        # Shared E0 from summed signal for stability
        mu_sum = data[channels].sum(axis=1).values
        e0 = self._determine_e0_first_derivative(energy, mu_sum)
    
        mu_stack = np.array([data[ch].values for ch in channels], dtype=float)
        ref_mu = self._safe_savgol(np.nanmedian(mu_stack, axis=0))
    
        pre_mask = (energy > e0 + pre_window[0]) & (energy < e0 + pre_window[1])
        post_mask = (energy > e0 + post_window[0]) & (energy < e0 + post_window[1])
    
        rows, good_channels = [], []
    
        for ch in channels:
            mu = data[ch].values.astype(float)
            row = {"channel": ch}
    
            # Liveness
            finite_mu = mu[np.isfinite(mu)]
            if finite_mu.size == 0 or np.nanmax(np.abs(finite_mu)) <= min_signal:
                row.update({k: np.nan for k in diag_cols if k not in ("status",)})
                row["status"] = "dead"
                rows.append(row)
                continue
    
            # Robust normalization to avoid spikes dominating scale
            den = np.nanpercentile(np.abs(mu), 99.5)
            if not np.isfinite(den) or den == 0.0:
                row.update({k: np.nan for k in diag_cols if k not in ("status",)})
                row["status"] = "dead"
                rows.append(row)
                continue
    
            mu_norm = mu / den
    
            # Window check
            if pre_mask.sum() < 5 or post_mask.sum() < 5:
                row.update({k: np.nan for k in diag_cols if k not in ("status",)})
                row["status"] = "bad_energy_window"
                rows.append(row)
                continue
    
            pre = mu_norm[pre_mask]
            post = mu_norm[post_mask]
    
            pre_mean = np.nanmean(pre)
            post_mean = np.nanmean(post)
            edge_jump = float(post_mean - pre_mean)
    
            # Smooth once for residuals and correlations
            mu_smooth = self._safe_savgol(mu_norm)
    
            # Pre-edge robust noise (BASIC gate)
            rms_pre = mad_to_rms(pre)
    
            # Residuals to isolate high-frequency noise
            pre_resid = pre - mu_smooth[pre_mask]
            post_resid = post - mu_smooth[post_mask]
            rms_pre_resid = mad_to_rms(pre_resid)
            rms_post_resid = mad_to_rms(post_resid)
    
            # BASIC metrics
            snr_post = (edge_jump / rms_post_resid) if (np.isfinite(edge_jump) and np.isfinite(rms_post_resid) and rms_post_resid > 0) else np.nan
            hf_ratio = (rms_post_resid / rms_pre_resid) if (np.isfinite(rms_post_resid) and np.isfinite(rms_pre_resid) and rms_pre_resid > 0) else np.nan
    
            # Diagnostics: correlations and post-edge slope
            m = np.isfinite(mu_smooth) & np.isfinite(ref_mu)
            corr = float(np.corrcoef(mu_smooth[m], ref_mu[m])[0, 1]) if m.sum() > 2 else np.nan
            mp = post_mask & np.isfinite(mu_smooth) & np.isfinite(ref_mu)
            corr_post = float(np.corrcoef(mu_smooth[mp], ref_mu[mp])[0, 1]) if mp.sum() > 2 else np.nan
    
            x_post = energy[post_mask]
            y_post = mu_norm[post_mask]
            if np.isfinite(x_post).all() and np.isfinite(y_post).all() and x_post.size > 5:
                post_slope = float(np.polyfit(x_post, y_post, 1)[0])
            else:
                post_slope = np.nan
    
            # --- DECISION LOGIC ---
            # Start with BASIC gates
            # basic_ok = (
            #     (np.isfinite(rms_pre) and (rms_pre < max_rms)) and
            #     (np.isfinite(hf_ratio) and (hf_ratio < max_hf_ratio)) and
            #     (np.isfinite(snr_post) and (snr_post > min_snr_post))
            # )
            # Start with BASIC gates
            basic_ok = (
                (np.isfinite(snr_post) and (snr_post > min_snr_post))
            )
            # Legacy gates (optional): require finite and pass
            if use_legacy_gates:
                legacy_ok = (
                    (np.isfinite(edge_jump) and (edge_jump > min_edge_jump)) and
                    (np.isfinite(corr) and (corr > min_corr))
                )
            else:
                legacy_ok = True
            
            # Post-edge detail gates (optional): require finite and pass
            if use_post_checks:
                post_ok = (
                    (np.isfinite(post_slope) and (abs(post_slope) < max_post_slope)) and
                    (np.isfinite(rms_post_resid) and (rms_post_resid < max_rms_post_resid)) and
                    (np.isfinite(corr_post) and (corr_post > min_corr_post))
                )
            else:
                post_ok = True
            
            is_good = bool(basic_ok and legacy_ok and post_ok)
            
            row.update(
                # edge_jump=float(edge_jump) if np.isfinite(edge_jump) else np.nan,
                rms=float(rms_pre) if np.isfinite(rms_pre) else np.nan,
                corr=float(corr) if np.isfinite(corr) else np.nan,
                # rms_post_resid=float(rms_post_resid) if np.isfinite(rms_post_resid) else np.nan,
                # corr_post=float(corr_post) if np.isfinite(corr_post) else np.nan,
                # post_slope=float(post_slope) if np.isfinite(post_slope) else np.nan,
                # hf_ratio=float(hf_ratio) if np.isfinite(hf_ratio) else np.nan,
                snr_post=float(snr_post) if np.isfinite(snr_post) else np.nan,
                status="good" if is_good else "rejected",
            )
            
            if is_good:
                good_channels.append(ch)
            
            rows.append(row)
    
        metrics = pd.DataFrame(rows).set_index("channel")
        return good_channels, metrics, e0
        
    # ------------------- Sum channels -------------------
    @staticmethod
    def sum_channels(
        data: pd.DataFrame,
        channels: Optional[List[str]] = None,
        metrics: Optional[pd.DataFrame] = None,
        use_status: str = "good",
        exclude_channels: Optional[List[str]] = None,
        fill_value: float = 0.0,
        strict: bool = False,
        allow_empty: bool = True,
        normalize: bool = False,
    ) -> Tuple[np.ndarray, List[str]]:
        """Sum (or average) selected DataFrame columns, with NaN handling."""
        if channels is None:
            if metrics is None:
                raise ValueError("Need channels or metrics")
            channels = metrics.index[metrics["status"] == use_status].tolist()
        else:
            channels = list(channels)

        if exclude_channels is not None:
            channels = [ch for ch in channels if ch not in set(exclude_channels)]

        if not channels:
            if allow_empty:
                return np.zeros(len(data)), []
            raise ValueError("No channels selected")

        mu_sum, used_channels = np.zeros(len(data)), []
        for ch in channels:
            if ch not in data.columns:
                if strict:
                    raise KeyError(f"{ch} missing")
                continue
            vals = np.nan_to_num(
                data[ch].to_numpy(dtype=float),
                nan=fill_value, posinf=fill_value, neginf=fill_value,
            )
            mu_sum += vals
            used_channels.append(ch)

        if normalize and used_channels:
            mu_sum /= len(used_channels)
        return mu_sum, used_channels

    # ------------------- Safe autobk -------------------
    def _safe_autobk(self, grp: Group, *, e0=None, user_kws: Optional[Dict[str, Any]] = None) -> None:
        """
        Robust autobk:
          - Try user kwargs first.
          - If fails (e.g., covariance NoneType), retry with safe defaults and calc_uncertainties=False.
        """
        from larch.xafs import autobk as _autobk
        kws = dict(user_kws or {})
        try:
            _autobk(grp, e0=e0, **kws)
            return
        except Exception:
            fallback = dict(kws)
            fallback.setdefault("rbkg", 1.0)
            fallback.setdefault("kmin", 0.0)
            fallback.setdefault("kmax", 10.0)
            fallback.setdefault("dk", 0.25)
            fallback["calc_uncertainties"] = False
            _autobk(grp, e0=e0, **fallback)



# ----------------------------------------------------------------
#                   LARCH_PARAMS['autobk']
# ------------------- Align multiple datasets --------------------
    def align_multiple(
        self,
        files: Dict[str, str],
        max_rms: float = 1.0e-3,
        max_hf_ratio: float = 100.0,     # post/pre residual noise inflation
        min_snr_post: float = 100.0, 
        reference: Optional[str] = None,
        larch_params: Optional[Dict[str, Dict[str, Any]]] = None,
        inb_filter: str = "Ka1_spectra",
        ion_chamber: Optional[str] = None,
        summed_groups: Optional[List[Union[List[str], Dict[str, Any]]]] = None,
        sum_names: Optional[List[str]] = None,
        plot_mode: Optional[str] = None,
        include_group_sums: bool = True,
        plot_good_channels_now: bool = False,
        channel_side: str = None,
        auto_rbkg: bool = False,
        # interactive export controls (kept)
        prompt_save: bool = False,
        include_mu_norm_export: bool = False,
        # ---------------- UPDATED: explicit deglitch controls ----------------
        deglitch_mode: Optional[str] = None,      # one of {None, 'pfy', 'both'}
        deglitch_region: Optional[str] = None,    # one of {"pre","xanes","exafs","custom"}
        deglitch_xrange: Optional[Union[Tuple[float, float], Dict[str, Tuple[float, float]]]] = None,
    ):
        """
        Align multiple XAS datasets (BioXAS) with robust PFY + Transmission support.
    
        OPTION A (this implementation):
          • Each SUM(...) group is computed in the RAW (unaligned) energy frame.
          • Group grid = shortest overlap among group members (no extrapolation).
          • E0 per group:
              - If reference is None -> use E0 of first member in the group.
              - If reference provided (numeric or label) -> use the global reference E0.
          • Transmission aggregation for sums uses RAW arrays and the same E0 as PFY-like sum.
    
        Flow (PFY singles):
          1) Load data
          2) Evaluate metrics & identify good channels
          3) Sum the PFY good channels  -> single PFY μ(E) per scan
          4) Compute E0 (from summed PFY) for region gating if deglitch_mode is not None
          5) Deglitch PFY (optional; modes below)
          6) Pre-edge/autobk/FT/WT on the chosen μ(E)
    
        Deglitching (explicit modes):
          - deglitch_mode=None: no deglitching (legacy). PFY μ(E) is In/I0 when ion_chamber is given,
            otherwise raw In (summed PFY).
          - deglitch_mode='pfy': compute In/I0 and deglitch the ratio.
                In_n = In / I0
                μ(E) := degl.process(energy, In_n, region=..., x_range=..., E0=PFY-E0)
            If ion_chamber was not specified or not found, falls back to default I0Detector_DarkCorrect.
          - deglitch_mode='both': deglitch In and I0 separately, then normalize:
                In_D = degl.process(energy, In,  ...)
                I0_D = degl.process(energy, I0,  ...)
                μ(E) := In_D / clip(I0_D, eps, +∞)
    
          'deglitch_region' ∈ {"pre","xanes","exafs","custom"}.
          For "custom", provide 'deglitch_xrange' as (xmin, xmax) or dict of named windows.
          E0 is auto-computed from the summed PFY (pre-deglitch) using pre_edge().
        """
        # ----------------- Imports -----------------
        import numpy as np
        import pandas as pd
        from typing import Any, Dict, List, Optional, Tuple, Union
        from larch import Group
        from larch.xafs import pre_edge, xftf, xftr, cauchy_wavelet
    
        # --- Validate larch params ---
        if not (larch_params and ("xftf" in larch_params)):
            raise RuntimeError(
                "align_multiple requires larch_params['xftf'] so χ(R) is computed and stored.\n"
                "Example LARCH_PARAMS:\n"
                "  {'pre_edge': {'nnorm': 2}, 'autobk': {'rbkg': 1.0, 'kmin': 0, 'kmax': 12, 'dk': 1.0, 'calc_uncertainties': False},\n"
                "   'xftf': {'kmin': 2, 'kmax': 12, 'kweight': 2, 'dk': 1.0, 'window': 'hanning'}}"
            )
    
        # --- Validate deglitch inputs ---
        allowed_modes = {None, "pfy", "both", "none"}
        if deglitch_mode not in allowed_modes:
            raise ValueError("deglitch_mode must be one of {None, 'pfy', 'both', 'none'}.")
    
        # normalize 'none' -> None
        if deglitch_mode == "none":
            deglitch_mode = None
    
        if deglitch_mode is not None:
            allowed_regions = {"pre", "xanes", "exafs", "custom"}
            if not isinstance(deglitch_region, str) or deglitch_region.lower() not in allowed_regions:
                raise ValueError("With deglitch_mode set, deglitch_region must be one of {'pre','xanes','exafs','custom'}.")
            deglitch_region = deglitch_region.lower()
            if deglitch_region == "custom" and deglitch_xrange is None:
                raise ValueError("With deglitch_region='custom', provide deglitch_xrange as (xmin,xmax) or a dict of named ranges.")
    
        # -------------- Cache params --------------
        self._ft_params = dict(larch_params["xftf"])
        self._default_kweight = int(self._ft_params.get("kweight", {}))
        self._pre_edge_params = dict(larch_params.get("pre_edge", {}))
        self._autobk_params   = dict(larch_params.get("autobk", {}))
    
        # -------------- Reset caches --------------
        self.metrics = {}
        self.metrics_all = None
        if not hasattr(self, "exafs") or self.exafs is None:
            self.exafs = {}
        self._channel_eval = getattr(self, "_channel_eval", {})
    
        # -------------- Toggle cache --------------
        self._deglitch_enabled = bool(deglitch_mode is not None)
    
        # -------------- Constants / containers ----
        self._ka1_token = inb_filter if inb_filter else "Ka1_spectra"
        ka1_token = self._ka1_token
        _ka1_inb_hits_total = 0
        _ka1_outb_hits_total = 0
        I0_KEY = "I0Detector_DarkCorrect"
        I1_KEY = "I1Detector_DarkCorrect"
        I2_KEY = "I2Detector_DarkCorrect"
        eps = 1e-12
    
        # Safe-label map
        safe_label_map: Dict[str, str] = {}
        seen: Dict[str, int] = {}
        for lbl in files.keys():
            base = self._safe_label(lbl)
            if base not in seen:
                safe = base
                seen[base] = 1
            else:
                seen[base] += 1
                safe = f"{base}_{seen[base]}"
            safe_label_map[lbl] = safe
    
        groups: Dict[str, Group] = {}
        e0_dict: Dict[str, float] = {}
        aligned_data: Dict[str, Dict[str, Any]] = {}
        sum_label_to_members: Dict[str, List[str]] = {}
        label_display_map: Dict[str, str] = {}
    
        # Validate inputs
        side = (channel_side or "InB").strip()
        if side not in ("InB", "OutB"):
            raise ValueError(f"channel_side must be 'InB' or 'OutB', got {side!r}")
    
        # =========================
        # Helpers
        # =========================
        def _interp(src_x, src_y, dst_x):
            if src_x is None or src_y is None or len(src_x) == 0 or len(src_y) == 0:
                return np.full_like(dst_x, np.nan, dtype=float)
            order = np.argsort(src_x)
            xs = np.asarray(src_x)[order]
            ys = np.asarray(src_y)[order]
            return np.interp(dst_x, xs, ys, left=np.nan, right=np.nan)
    
        def _first_col(df_cols, key):
            cols = [c for c in df_cols if key in c]
            return cols[0] if cols else None
    
        def _safe_autobk_local(grp: Group, *, e0=None, user_kws: Optional[Dict[str, Any]] = None) -> None:
            from larch.xafs import autobk as _local_autobk
            kws = dict(user_kws or {})
            try:
                _local_autobk(grp, e0=e0, **kws)
            except Exception:
                fallback = dict(kws)
                fallback.setdefault("rbkg", 1.0)
                fallback.setdefault("kmin", 0.0)
                fallback.setdefault("kmax", 10.0)
                fallback.setdefault("dk", 0.25)
                fallback["calc_uncertainties"] = False
                _local_autobk(grp, e0=e0, **fallback)
    
        def _optimize_rbkg_rigorous(
            group: Group, *,
            e0: Optional[float],
            ft_kws: Dict[str, Any],
            autobk_kws: Dict[str, Any],
            bounds=(0.6, 2.2),
            coarse_steps: int = 61,
            topk: int = 3,
            refine_halfwidth: float = 0.25,
            max_iter: int = 80,
            tol: float = 5e-4,
            w_R: float = 0.75, w_B: float = 0.05, w_reg: float = 0.20,
            r_gate_mode: str = "fixed",
            rmax_bg: float = 1.0,
        ) -> float:
            from larch.xafs import autobk as _abk, xftf as _xftf
            eps_local = 1e-18
            kweight_eval = int(ft_kws.get("kweight", autobk_kws.get("kweight", 2)))
    
            def _snapshot(g: Group):
                return {k: getattr(g, k, None) for k in ("mu","bkg","chi","k","r","chir","chir_mag","chir_re","chir_im","e0")}
            def _restore(g: Group, snap: Dict[str, Any]):
                for k, v in snap.items():
                    if v is None:
                        if hasattr(g, k):
                            try:
                                delattr(g, k)
                            except Exception:
                                pass
                    else:
                        setattr(g, k, v)
    
            def _eval_obj(rbkg_val: float, gate_mode: str, gate_fixed: float):
                snap = _snapshot(group)
                try:
                    abk = dict(autobk_kws); abk["rbkg"] = float(rbkg_val); abk.setdefault("kweight", kweight_eval)
                    _abk(group, e0=e0, **abk)
                    ft = dict(ft_kws); ft.setdefault("kweight", kweight_eval)
                    _xftf(group, **ft)
    
                    r = getattr(group, "r", None); cm = getattr(group, "chir_mag", None)
                    E = getattr(group, "energy", None); b = getattr(group, "bkg", None); m = getattr(group, "mu", None)
                    if any(x is None for x in (r, cm, E, b, m)):
                        return np.inf, np.nan, np.nan
    
                    r = np.asarray(r); cm = np.asarray(cm); E = np.asarray(E); b = np.asarray(b); m = np.asarray(m)
    
                    Rgate = (max(0.5, float(rbkg_val)) if gate_mode == "adaptive" else float(gate_fixed))
                    rmask = (r >= 0.0) & (r < Rgate) & np.isfinite(cm)
                    if np.any(rmask):
                        leak_raw = np.trapz(np.abs(cm[rmask]), x=r[rmask])
                        leak_scale = np.nanpercentile(np.abs(cm), 95) + eps_local
                        leak = leak_raw / (leak_scale * (Rgate + eps_local))
                    else:
                        leak = 1e3
    
                    mask_f = np.isfinite(E) & np.isfinite(b)
                    if np.sum(mask_f) >= 5:
                        d2b = np.gradient(np.gradient(b[mask_f]))
                        curv_raw = np.nanmedian(np.abs(d2b))
                        mu_scale = np.nanpercentile(np.abs(m[mask_f]), 95) + eps_local
                        curv = curv_raw / mu_scale
                    else:
                        curv = 0.0
    
                    reg = (float(rbkg_val) - 1.0) ** 2
                    obj = w_R * leak + w_B * curv + w_reg * reg
                    return float(obj), float(leak), float(curv)
                except Exception:
                    return np.inf, np.nan, np.nan
                finally:
                    _restore(g, snap)
    
            a, b = float(bounds[0]), float(bounds[1])
            grid = np.linspace(a, b, int(coarse_steps))
            obj_grid = np.array([_eval_obj(x, r_gate_mode, rmax_bg)[0] for x in grid])
    
            span = np.nanmax(obj_grid) - np.nanmin(obj_grid)
            if (not np.isfinite(span)) or (span < 1e-3):
                obj_grid = np.array([_eval_obj(x, "adaptive", rmax_bg)[0] for x in grid])
                span2 = np.nanmax(obj_grid) - np.nanmin(obj_grid)
                if (not np.isfinite(span2)) or (span2 < 1e-3):
                    a, b = 0.5, 2.5
                    grid = np.linspace(a, b, max(coarse_steps, 61))
                    obj_grid = np.array([_eval_obj(x, r_gate_mode, rmax_bg)[0] for x in grid])
    
            order = np.argsort(obj_grid)
            seeds = grid[order[:max(1, topk)]]
    
            def _refine(seed: float):
                left = max(a, seed - refine_halfwidth)
                right = min(b, seed + refine_halfwidth)
                if right - left < 4 * tol:
                    left = max(a, seed - 2 * refine_halfwidth)
                    right = min(b, seed + 2 * refine_halfwidth)
                phi = (np.sqrt(5.0) - 1.0) / 2.0
                c = right - phi * (right - left)
                d = left + phi * (right - left)
                fc, _, _ = _eval_obj(c, r_gate_mode, rmax_bg)
                fd, _, _ = _eval_obj(d, r_gate_mode, rmax_bg)
                it = 0
                while (right - left) > tol and it < max_iter:
                    if fc < fd:
                        right, d, fd = d, c, fc
                        c = right - phi * (right - left)
                        fc, _, _ = _eval_obj(c, r_gate_mode, rmax_bg)
                    else:
                        left, c, fc = c, d, fd
                        d = left + phi * (right - left)
                        fd, _, _ = _eval_obj(d, r_gate_mode, rmax_bg)
                    it += 1
                cand = [(c, fc), (d, fd),
                        (left, _eval_obj(left, r_gate_mode, rmax_bg)[0]),
                        (right, _eval_obj(right, r_gate_mode, rmax_bg)[0])]
                return min(cand, key=lambda t: t[1])
    
            best_x, best_f = None, np.inf
            for s in seeds:
                x_star, f_star = _refine(float(s))
                if f_star < best_f:
                    best_x, best_f = x_star, f_star
            return float(best_x)
    
        def _store_xafs_outputs(grp: Group, label: str) -> None:
            self.exafs.setdefault(label, {})
            self.exafs[label]["ft"] = {
                "k": getattr(grp, "k", None),
                "chi": getattr(grp, "chi", None),
                "r": getattr(grp, "r", None),
                "chir": getattr(grp, "chir", None),
                "chir_mag": getattr(grp, "chir_mag", None),
                "chir_re": getattr(grp, "chir_re", None),
                "chir_im": getattr(grp, "chir_im", None),
                "kweight": int(self._ft_params.get("kweight", getattr(grp, "kweight", 2))),
            }
            self.exafs[label]["wavelet"] = {
                "k": getattr(grp, "k", None),
                "r": getattr(grp, "wcauchy_r", None),
                "wcauchy_mag": getattr(grp, "wcauchy_mag", None),
                "wcauchy_re": getattr(grp, "wcauchy_re", None),
                "wcauchy_im": getattr(grp, "wcauchy_im", None),
                "kweight": int(self._ft_params.get("kweight", getattr(grp, "kweight", 2))),
            }
            self.exafs[label]["meta"] = {
                "e0": getattr(grp, "e0", None),
                "filename": getattr(grp, "filename", None),
                "filepath": getattr(grp, "filepath", None),
                "rbkg_opt": getattr(grp, "_rbkg_opt", None),
            }
    
        # Region-aware deglitch of a single y-series (the summed PFY or ratio)
        def _deglitch_series(x, y, *, region, x_range, e0):
            try:
                degl = XASDeglitcher(window=5, threshold=6)
            except NameError:
                degl = getattr(self, "XASDeglitcher", None)
                degl = degl(window=5, threshold=6) if callable(degl) else None
            if degl is None:
                return y
            y_corr = degl.process(x, y, region=region, x_range=x_range, E0=e0, return_indices=False)
            return y_corr
    
        # =========================
        # PROCESS INDIVIDUAL SCANS
        # =========================
        for orig_label, fname in files.items():
            label = safe_label_map[orig_label]
            try:
                loader = self.XASDataLoader(fname).load()
                energy = loader.get("energy eV")
    
                # detector presence
                all_cols = loader.columns
                inb_hits  = sum(1 for c in all_cols if (ka1_token in c) and ("InB"  in c))
                outb_hits = sum(1 for c in all_cols if (ka1_token in c) and ("OutB" in c))
                _ka1_inb_hits_total  += inb_hits
                _ka1_outb_hits_total += outb_hits
                pfy_any = (inb_hits + outb_hits) > 0
    
                # Ion chambers
                def _col_or_nan(col_name):
                    return loader.data[col_name].to_numpy(float) if col_name in loader.data.columns else np.full(len(energy), np.nan)
                i0_col = _first_col(loader.columns, I0_KEY)
                i1_col = _first_col(loader.columns, I1_KEY)
                i2_col = _first_col(loader.columns, I2_KEY)
                I0_raw = _col_or_nan(i0_col) if i0_col else np.full(len(energy), np.nan)
                I1_raw = _col_or_nan(i1_col) if i1_col else np.full(len(energy), np.nan)
                I2_raw = _col_or_nan(i2_col) if i2_col else np.full(len(energy), np.nan)
    
                # PFY lists
                side_channels = [c for c in loader.columns if (ka1_token in c) and (channel_side in c)] if pfy_any else []
                inb_all  = [c for c in loader.columns if (ka1_token in c) and ("InB"  in c)]
                outb_all = [c for c in loader.columns if (ka1_token in c) and ("OutB" in c)]
    
                # --- Evaluate 'good' channels on original data ---
                if len(inb_all) > 0:
                    good_inb, metrics_inb, _ = self.evaluate_fluo_channels(
                        energy, loader.data, inb_all,
                        max_rms=max_rms, max_hf_ratio=max_hf_ratio, min_snr_post=min_snr_post
                    )
                else:
                    good_inb, metrics_inb = [], pd.DataFrame()
    
                if len(outb_all) > 0:
                    good_outb, metrics_outb, _ = self.evaluate_fluo_channels(
                        energy, loader.data, outb_all,
                        max_rms=max_rms, max_hf_ratio=max_hf_ratio, min_snr_post=min_snr_post
                    )
                else:
                    good_outb, metrics_outb = [], pd.DataFrame()
    
                good_side, metrics_side, e0_fluo = [], pd.DataFrame(), None
                if pfy_any and len(side_channels) > 0:
                    good_side, metrics_side, e0_fluo = self.evaluate_fluo_channels(
                        energy, loader.data, side_channels,
                        max_rms=max_rms, max_hf_ratio=max_hf_ratio, min_snr_post=min_snr_post
                    )
    
                # --- SUM the PFY 'good' channels (from original data) ---
                sum_pfy_raw = None  # raw PFY sum (In) on native grid
                if pfy_any and len(side_channels) > 0 and len(good_side) > 0:
                    sum_pfy_raw, _ = self.sum_channels(loader.data, metrics=metrics_side, use_status="good")
    
                # --- Consistent sort/uniq (for all arrays that follow) ---
                idx = np.argsort(energy)
                energy       = energy[idx]
                I0_raw       = I0_raw[idx];  I1_raw = I1_raw[idx];  I2_raw = I2_raw[idx]
                if sum_pfy_raw is not None:
                    sum_pfy_raw = sum_pfy_raw[idx]
    
                energy, uidx = np.unique(energy, return_index=True)
                I0_raw       = I0_raw[uidx];  I1_raw = I1_raw[uidx];  I2_raw = I2_raw[uidx]
                if sum_pfy_raw is not None:
                    sum_pfy_raw = sum_pfy_raw[uidx]
    
                # --- Prepare I0 for normalization ---
                # Fallback to default I0 (I0Detector_DarkCorrect) if ion_chamber not specified or not found.
                I0_for_norm = I0_raw.copy()
                if ion_chamber:
                    I0_cols = [c for c in loader.columns if (ion_chamber in c) and ("Detector_DarkCorrect" in c)]
                    I0_sel = loader.get(I0_cols[0]) if I0_cols else None
                    if I0_sel is not None:
                        I0_sel = np.asarray(I0_sel)[idx][uidx]
                        if np.any(np.isfinite(I0_sel)):
                            I0_for_norm = I0_sel
                I0_for_norm = np.clip(I0_for_norm, eps, np.inf)
    
                # --- Compute E0 from the summed PFY (required for region gating) ---
                e0_use_for_region = None
                if sum_pfy_raw is not None:
                    gtmp_e0 = Group()
                    gtmp_e0.energy = energy.copy()
                    gtmp_e0.mu = sum_pfy_raw.copy()
                    pre_edge(gtmp_e0, **larch_params.get("pre_edge", {}))
                    if getattr(gtmp_e0, "e0", None) is not None and np.isfinite(getattr(gtmp_e0, "e0")):
                        e0_use_for_region = float(getattr(gtmp_e0, "e0"))
    
                # If region needs E0 but we could not compute from PFY, fail clearly
                if (deglitch_mode is not None) and (deglitch_region in {"pre", "xanes", "exafs"}) and (e0_use_for_region is None):
                    raise ValueError(
                        f"Failed to compute E0 from PFY sum for deglitch region '{deglitch_region}' in '{orig_label}'. "
                        "Check PFY data quality or use deglitch_region='custom'."
                    )
    
                # --- Apply deglitch to the PFY path (NOT per-channel) according to mode ---
                mu_pfy_for_pipeline = None
                if sum_pfy_raw is not None:
                    if deglitch_mode is None:
                        # Legacy: no deglitch; normalize only if ion_chamber was provided
                        if ion_chamber is not None:
                            mu_pfy_for_pipeline = self.fluorescence(sum_pfy_raw, I0_for_norm)
                        else:
                            mu_pfy_for_pipeline = sum_pfy_raw
                    elif deglitch_mode == "pfy":
                        # Deglitch the ratio In/I0 (use ion_chamber if available; otherwise fallback to default I0)
                        In_n = sum_pfy_raw / I0_for_norm
                        mu_pfy_for_pipeline = _deglitch_series(
                            x=energy, y=In_n,
                            region=deglitch_region, x_range=deglitch_xrange, e0=e0_use_for_region
                        )
                    elif deglitch_mode == "both":
                        # Deglitch In and I0 separately, then ratio
                        In_D = _deglitch_series(
                            x=energy, y=sum_pfy_raw,
                            region=deglitch_region, x_range=deglitch_xrange, e0=e0_use_for_region
                        )
                        I0_D = _deglitch_series(
                            x=energy, y=I0_for_norm,
                            region=deglitch_region, x_range=deglitch_xrange, e0=e0_use_for_region
                        )
                        I0_D = np.clip(I0_D, eps, np.inf)
                        mu_pfy_for_pipeline = In_D / I0_D
                    else:
                        raise ValueError(f"Unexpected deglitch_mode: {deglitch_mode!r}")
    
                # --- Build InB/OutB sums for headers/export (from original good lists) ---
                if len(inb_all) > 0:
                    inb_sum_scan, _  = self.sum_channels(loader.data, metrics=metrics_inb,  use_status="good", allow_empty=True)
                    inb_sum_scan = inb_sum_scan[idx][uidx]
                else:
                    inb_sum_scan = np.zeros_like(energy)
    
                if len(outb_all) > 0:
                    outb_sum_scan, _ = self.sum_channels(loader.data, metrics=metrics_outb, use_status="good", allow_empty=True)
                    outb_sum_scan = outb_sum_scan[idx][uidx]
                else:
                    outb_sum_scan = np.zeros_like(energy)
    
                # --- PRIMARY modality selection ---
                primary_mode = "pfy" if (mu_pfy_for_pipeline is not None) else "trans"
    
                grp = Group()
                grp.energy = energy
                if primary_mode == "pfy":
                    # Use the (possibly deglitched) PFY μ(E) for everything downstream
                    grp.mu = mu_pfy_for_pipeline
                    grp.mu_raw = mu_pfy_for_pipeline.copy()
                else:
                    # transmission μ on native grid
                    I0s = np.clip(I0_raw, eps, np.inf)
                    I1s = np.clip(I1_raw, eps, np.inf)
                    mu_trans_raw_primary = np.log(I0s / I1s)
                    grp.mu = mu_trans_raw_primary
                    grp.mu_raw = mu_trans_raw_primary.copy()
    
                grp.filename = label
                grp.filepath = fname
    
                # --- Primary pipeline on the chosen μ(E) ---
                pre_edge(grp, **larch_params.get("pre_edge", {}))
                e0_use = getattr(grp, "e0", e0_use_for_region if e0_use_for_region is not None else getattr(grp, "e0", None))
    
                if auto_rbkg:
                    chosen_rbkg = _optimize_rbkg_rigorous(
                        grp, e0=e0_use,
                        ft_kws=larch_params.get("xftf", {}),
                        autobk_kws=larch_params.get("autobk", {}),
                        bounds=(0.6, 2.2), coarse_steps=61, topk=3,
                        r_gate_mode="fixed", rmax_bg=1.0,
                        w_R=0.90, w_B=0.10, w_reg=0.0,
                    )
                    abk_kws = dict(larch_params.get("autobk", {}))
                    abk_kws["rbkg"] = float(chosen_rbkg)
                    abk_kws.setdefault("kweight", int(larch_params.get("xftf", {}).get("kweight", 2)))
                    _safe_autobk_local(grp, e0=e0_use, user_kws=abk_kws)
                    setattr(grp, "_rbkg_opt", float(chosen_rbkg))
                else:
                    _safe_autobk_local(grp, e0=e0_use, user_kws=larch_params.get("autobk", {}))
    
                xftf(grp, **larch_params["xftf"])
                if "xftr" in larch_params:
                    xftr(grp, **larch_params["xftr"])
                cauchy_wavelet(grp, kweight=int(larch_params["xftf"].get("kweight", self._default_kweight)))
    
                # store PFY/primary FT+wavelet
                _store_xafs_outputs(grp, label)
    
                # --- Transmission processing (secondary, only if PFY is primary) ---
                mu_trans_raw = None
                mu_trans_norm = None
                mu_trans_flat = None
                if primary_mode == "pfy":
                    try:
                        I0s = np.clip(I0_raw, eps, np.inf)
                        I1s = np.clip(I1_raw, eps, np.inf)
                        mu_trans_raw = np.log(I0s / I1s)
    
                        grpT = Group()
                        grpT.energy = energy.copy()
                        grpT.mu = mu_trans_raw.copy()
                        grpT.mu_raw = mu_trans_raw.copy()
                        grpT.filename = f"{label}__trans"
                        grpT.filepath = fname
    
                        pre_edge(grpT, **larch_params.get("pre_edge", {}))
                        e0T = getattr(grpT, "e0", None)
    
                        if auto_rbkg:
                            chosen_rbkg_T = _optimize_rbkg_rigorous(
                                grpT, e0=e0T,
                                ft_kws=larch_params.get("xftf", {}),
                                autobk_kws=larch_params.get("autobk", {}),
                                bounds=(0.6, 2.2), coarse_steps=61, topk=3,
                                r_gate_mode="fixed", rmax_bg=1.0,
                            )
                            abkT = dict(larch_params.get("autobk", {}))
                            abkT["rbkg"] = float(chosen_rbkg_T)
                            abkT.setdefault("kweight", int(larch_params.get("xftf", {}).get("kweight", 2)))
                            _safe_autobk_local(grpT, e0=e0T, user_kws=abkT)
                            setattr(grpT, "_rbkg_opt", float(chosen_rbkg_T))
                        else:
                            _safe_autobk_local(grpT, e0=e0T, user_kws=larch_params.get("autobk", {}))
    
                        xftf(grpT, **larch_params["xftf"])
                        if "xftr" in larch_params:
                            xftr(grpT, **larch_params["xftr"])
                        cauchy_wavelet(grpT, kweight=int(larch_params.get("xftf", {}).get("kweight", self._default_kweight)))
                        _store_xafs_outputs(grpT, f"{label}__trans")
    
                        setattr(grp, "_trans", grpT)
                        self.exafs.setdefault(label, {})
                        self.exafs[label]["ft_trans"] = {
                            "k": getattr(grpT, "k", None),
                            "chi": getattr(grpT, "chi", None),
                            "r": getattr(grpT, "r", None),
                            "chir": getattr(grpT, "chir", None),
                            "chir_mag": getattr(grpT, "chir_mag", None),
                            "chir_re": getattr(grpT, "chir_re", None),
                            "chir_im": getattr(grpT, "chir_im", None),
                            "kweight": int(self._ft_params.get("kweight", getattr(grpT, "kweight", 2))),
                        }
                        self.exafs[label]["wavelet_trans"] = {
                            "k": getattr(grpT, "k", None),
                            "r": getattr(grpT, "wcauchy_r", None),
                            "wcauchy_mag": getattr(grpT, "wcauchy_mag", None),
                            "wcauchy_re": getattr(grpT, "wcauchy_re", None),
                            "wcauchy_im": getattr(grpT, "wcauchy_im", None),
                            "kweight": int(self._ft_params.get("kweight", getattr(grpT, "kweight", 2))),
                        }
    
                        mu_trans_norm = getattr(grpT, "norm", None)
                        mu_trans_flat = getattr(grpT, "flat", None)
                    except Exception as eT:
                        print(f"[WARN] Transmission processing failed for '{label}': {eT}")
                else:
                    mu_trans_raw  = grp.mu_raw.copy()
                    mu_trans_norm = getattr(grp, "norm", None)
                    mu_trans_flat = getattr(grp, "flat", None)
                    _store_xafs_outputs(grp, f"{label}__trans")
                    setattr(grp, "_trans", grp)
                    self.exafs.setdefault(label, {})
                    self.exafs[label]["ft_trans"] = {
                        "k": getattr(grp, "k", None),
                        "chi": getattr(grp, "chi", None),
                        "r": getattr(grp, "r", None),
                        "chir": getattr(grp, "chir", None),
                        "chir_mag": getattr(grp, "chir_mag", None),
                        "chir_re": getattr(grp, "chir_re", None),
                        "chir_im": getattr(grp, "chir_im", None),
                        "kweight": int(self._ft_params.get("kweight", getattr(grp, "kweight", 2))),
                    }
                    self.exafs[label]["wavelet_trans"] = {
                        "k": getattr(grp, "k", None),
                        "r": getattr(grp, "wcauchy_r", None),
                        "wcauchy_mag": getattr(grp, "wcauchy_mag", None),
                        "wcauchy_re": getattr(grp, "wcauchy_re", None),
                        "wcauchy_im": getattr(grp, "wcauchy_im", None),
                        "kweight": int(self._ft_params.get("kweight", getattr(grp, "kweight", 2))),
                    }
    
                # Book-keeping (note: 'metrics_side' etc. are from original evaluation)
                self.metrics[label] = metrics_side.copy() if isinstance(metrics_side, pd.DataFrame) else pd.DataFrame()
                self._channel_eval[label] = {
                    "energy": energy,
                    "data": loader.data,            # raw table (channel-level), PFY deglitch was on summed signal only
                    "channels": side_channels,
                    "good": good_side,
                    "metrics": metrics_side,
                    "e0": e0_use_for_region if pfy_any else None,
                    "uidx": uidx,    # <-- add this line
                }
                if plot_good_channels_now and pfy_any and len(side_channels) > 0:
                    self.plot_good_channels(label)
    
                groups[label] = grp
                e0_dict[label] = getattr(grp, "e0", None)
                label_display_map[label] = self._safe_label(orig_label)
    
                aligned_data[label] = {
                    "energy_raw": grp.energy,
                    "inb_sum_raw": inb_sum_scan,
                    "outb_sum_raw": outb_sum_scan,
                    "I0_raw": I0_raw, "I1_raw": I1_raw, "I2_raw": I2_raw,
                    "filepath": fname,
                    "label_safe": label,
                    "label_display": label_display_map.get(label, label),
    
                    # PRIMARY diagnostics (these are ON the post-deglitch pipeline μ if PFY)
                    "mutrans": grp.mu_raw,
                    "mu_raw": grp.mu_raw,
                    "pre_edge": getattr(grp, "pre_edge", None),
                    "post_edge": getattr(grp, "post_edge", None),
                    "bkg": getattr(grp, "bkg", None),
                    "mu_bkgsub": (grp.mu - getattr(grp, "bkg", 0)) if getattr(grp, "bkg", None) is not None else None,
                    "mu_norm": getattr(grp, "norm", None),
                    "mu_flat": getattr(grp, "flat", None),
    
                    # Transmission diagnostics (additive)
                    "mu_trans_raw":  mu_trans_raw,
                    "mu_trans_norm": mu_trans_norm,
                    "mu_trans_flat": mu_trans_flat,
    
                    # placeholders; aligned later
                    # "energy": ..., "I0_aligned": ..., etc.
    
                    # channel summaries (from original evaluation)
                    "InB_channels_all": inb_all,
                    "InB_channels_good": good_inb,
                    "OutB_channels_all": outb_all,
                    "OutB_channels_good": good_outb,
                }
    
            except Exception as e_file:
                print(f"[ERROR][align_multiple:single] label='{orig_label}', file='{fname}': {e_file}")
                raise
    
        # =========================
        # PICK GLOBAL REFERENCE
        # =========================
        def _resolve_reference(reference, groups, e0_dict, safe_label_map):
            if reference is None:
                non_trans = [k for k in groups.keys() if not (isinstance(k, str) and k.endswith("__trans"))]
                ref_label = (non_trans[0] if non_trans else list(groups.keys())[0])
                return ref_label, float(e0_dict[ref_label])
            try:
                e0_val = float(reference)
                if np.isfinite(e0_val):
                    return None, e0_val
            except (TypeError, ValueError):
                pass
            if hasattr(reference, "e0"):
                e0_attr = float(getattr(reference, "e0"))
                ref_label = getattr(reference, "label", None)
                if isinstance(ref_label, str):
                    ref_label = safe_label_map.get(ref_label, ref_label)
                else:
                    ref_label = None
                return ref_label, e0_attr
            if isinstance(reference, str):
                ref_label = safe_label_map.get(reference, reference)
                if ref_label not in groups:
                    raise KeyError(f"Reference '{reference}' not found among safe labels: {list(groups.keys())}")
                return ref_label, float(e0_dict[ref_label])
            raise TypeError("reference must be one of: None, str label, numeric E0, or object exposing `.e0`.")
    
        ref_label, e0_ref = _resolve_reference(reference, groups, e0_dict, safe_label_map)
        self._reference_label = ref_label
        self._reference_e0 = float(e0_ref)
    
        # =========================
        # ALIGN extras to reference (singles only; sums remain raw by design)
        # =========================
        for label, grp in groups.items():
            if label not in aligned_data:
                continue
            try:
                shift = e0_dict[label] - e0_ref
                E_aligned = grp.energy - shift
    
                rec_tmp = aligned_data[label]
                def _I(src):
                    return _interp(rec_tmp["energy_raw"], src, E_aligned)
    
                I0_a  = _I(rec_tmp["I0_raw"])
                I1_a  = _I(rec_tmp["I1_raw"])
                I2_a  = _I(rec_tmp["I2_raw"])
                InB_a = _I(rec_tmp["inb_sum_raw"])
                OutB_a= _I(rec_tmp["outb_sum_raw"])
    
                I0s = np.clip(I0_a, eps, np.inf)
                I1s = np.clip(I1_a, eps, np.inf)
                mu_trans_aligned = np.log(I0s / I1s)
    
                aligned_data[label].update({
                    "energy": E_aligned,
                    "InB_sum_good": InB_a,
                    "OutB_sum_good": OutB_a,
                    "I0_aligned": I0_a,
                    "I1_aligned": I1_a,
                    "I2_aligned": I2_a,
    
                    # carry PRIMARY diagnostics
                    "mutrans": grp.mu_raw,
                    "mu_raw": grp.mu_raw,
                    "pre_edge": getattr(grp, "pre_edge", None),
                    "post_edge": getattr(grp, "post_edge", None),
                    "bkg": getattr(grp, "bkg", None),
                    "mu_bkgsub": (grp.mu - getattr(grp, "bkg", 0)) if getattr(grp, "bkg", None) is not None else None,
                    "mu_norm": getattr(grp, "norm", None),
                    "mu_flat": getattr(grp, "flat", None),
    
                    "mu_trans": mu_trans_aligned,
                })
            except Exception as e_align:
                print(f"[ERROR][align extras] label='{label}', file='{aligned_data[label].get('filepath','?')}': {e_align}")
                raise
    
        # =========================
        # CREATE SUMMED GROUPS (RAW energy frame; PFY preferred, fallback to Transmission)
        # =========================
        if include_group_sums and summed_groups:
    
            # -----------------------
            # Helper utilities
            # -----------------------
            def _ensure_mono(E, Y=None):
                """Ensure E is monotonically increasing; sort Y accordingly if provided."""
                if E is None or len(E) == 0:
                    return E if Y is None else (E, Y)
                E = np.asarray(E).ravel()
                if not np.all(np.diff(E) >= 0):
                    order = np.argsort(E)
                    E_sorted = E[order]
                    if Y is None:
                        return E_sorted
                    Y = np.asarray(Y).ravel()
                    Y_sorted = Y[order]
                    return E_sorted, Y_sorted
                return (E if Y is None else (np.asarray(E).ravel(), np.asarray(Y).ravel()))
    
            def _get_overlap(energies):
                """Given a list of monotonically increasing energy arrays, return (E_min, E_max) for common overlap."""
                if not energies:
                    return None, None
                starts, ends = [], []
                for e in energies:
                    if e is None or len(e) == 0:
                        continue
                    e = np.asarray(e).ravel()
                    e = _ensure_mono(e)
                    starts.append(e[0])
                    ends.append(e[-1])
                if not starts or not ends:
                    return None, None
                E_min = max(starts)
                E_max = min(ends)
                if E_min >= E_max:
                    return None, None
                return E_min, E_max
    
            def _pick_raw_series(rec: dict):
                """Return raw-like μ(E): prefer PFY mu_raw; fallback to trans raw; then mutrans."""
                y = rec.get("mu_raw")
                if y is not None:
                    return y
                y = rec.get("mu_trans_raw")
                if y is not None:
                    return y
                y = rec.get("mutrans")
                return y
    
            def _interp_to_grid(E_src, Y_src, E_grid):
                """Safe interpolation from (E_src, Y_src) to E_grid."""
                if E_src is None or Y_src is None:
                    return None
                E_src, Y_src = _ensure_mono(E_src, Y_src)
                if E_src is None or len(E_src) == 0 or len(Y_src) == 0:
                    return None
                try:
                    return np.interp(E_grid, E_src, np.asarray(Y_src, dtype=float))
                except Exception:
                    return None
    
            use_first_member_e0 = True
    
            for idx_s, item in enumerate(summed_groups):
                try:
                    # Resolve members and group label
                    if isinstance(item, dict):
                        members_orig = item.get("members", [])
                        user_label = item.get("label", None)
                    else:
                        members_orig = item
                        user_label = (sum_names[idx_s] if (sum_names and idx_s < len(sum_names)) else None)
    
                    members = [safe_label_map.get(m, m) for m in members_orig]
                    if not members:
                        raise ValueError("Empty members in summed_groups entry.")
                    for m in members:
                        if m not in groups:
                            raise ValueError(f"Dataset '{m}' not found among processed labels.")
    
                    # Build per-group overlap grid in RAW frame
                    member_Es = []
                    for m in members:
                        # Prefer the Group's energy (raw), fallback to stored raw energy
                        Ei = getattr(groups[m], "energy", None)
                        if Ei is None:
                            Ei = aligned_data[m].get("energy_raw")
                        if Ei is None:
                            continue
                        Ei = _ensure_mono(Ei)
                        member_Es.append(Ei)
    
                    E_min, E_max = _get_overlap(member_Es)
                    if E_min is None or E_max is None:
                        raise ValueError(f"No common energy overlap among {members}.")
    
                    # Slice each member's RAW energy to overlap and pick the shortest as the group's grid
                    overlap_slices = {}
                    for m, Ei in zip(members, member_Es):
                        mask = (Ei >= E_min) & (Ei <= E_max)
                        Ei_clip = Ei[mask]
                        overlap_slices[m] = Ei_clip
    
                    shortest_member = min(members, key=lambda m: len(overlap_slices.get(m, [])))
                    group_grid = overlap_slices[shortest_member]
                    group_grid = np.asarray(group_grid, dtype=float).ravel()
                    if group_grid is None or len(group_grid) == 0:
                        raise ValueError(f"Overlap grid empty for group {members}.")
    
                    group_grid = np.unique(group_grid)
                    if len(group_grid) < 5:
                        print(f"[WARN] Very short overlap grid ({len(group_grid)} pts) for group {members}.")
    
                    # PFY-like RAW aggregation on the RAW group grid
                    mu_raw_accum, cnt_raw = None, 0
                    for name in members:
                        # Use RAW frame arrays
                        Ei = aligned_data[name].get("energy_raw")
                        yi = _pick_raw_series(aligned_data[name])
                        if (Ei is None) or (yi is None) or (len(Ei) == 0) or (len(yi) == 0):
                            continue
                        yi_interp = _interp_to_grid(Ei, yi, group_grid)
                        if yi_interp is None:
                            continue
                        mu_raw_accum = yi_interp if mu_raw_accum is None else (mu_raw_accum + yi_interp)
                        cnt_raw += 1
    
                    if cnt_raw == 0:
                        raise RuntimeError(f"No valid RAW μ arrays to average for: {members}")
    
                    mu_raw_sum = mu_raw_accum / cnt_raw
    
                    # Create SUM Group (PFY-like) — Larch processing
                    sum_label_raw = user_label if user_label else f"SUM({'+'.join(members)})"
                    sum_label = self._safe_label(sum_label_raw)
    
                    gsum = Group()
                    gsum.energy = group_grid  # RAW frame
                    gsum.mu = mu_raw_sum
                    gsum.mu_raw = mu_raw_sum.copy()
                    gsum.filename = sum_label
                    gsum.filepath = None
    
                    # Decide group's E0 (Option A policy)
                    forced_e0 = None
                    try:
                        if reference is None and use_first_member_e0:
                            forced_e0 = float(e0_dict[members[0]])
                        else:
                            forced_e0 = float(self._reference_e0)
                    except Exception:
                        forced_e0 = None
    
                    if forced_e0 is not None and np.isfinite(forced_e0):
                        pre_edge(gsum, e0=forced_e0, **larch_params.get("pre_edge", {}))
                        gsum.e0 = forced_e0
                    else:
                        pre_edge(gsum, **larch_params.get("pre_edge", {}))  # fallback
    
                    if auto_rbkg:
                        chosen_rbkg = _optimize_rbkg_rigorous(
                            gsum, e0=gsum.e0,
                            ft_kws=larch_params.get("xftf", {}),
                            autobk_kws=larch_params.get("autobk", {}),
                            bounds=(0.6, 2.2), coarse_steps=61, topk=3,
                            r_gate_mode="fixed", rmax_bg=1.0,
                            w_R=0.90, w_B=0.10, w_reg=0.0,
                        )
                        abk_sum = dict(larch_params.get("autobk", {}))
                        abk_sum["rbkg"] = float(chosen_rbkg)
                        abk_sum.setdefault("kweight", int(larch_params.get("xftf", {}).get("kweight", 2)))
                        _safe_autobk_local(gsum, e0=gsum.e0, user_kws=abk_sum)
                        setattr(gsum, "_rbkg_opt", float(chosen_rbkg))
                    else:
                        _safe_autobk_local(gsum, e0=gsum.e0, user_kws=larch_params.get("autobk", {}))
    
                    xftf(gsum, **larch_params["xftf"])
                    if "xftr" in larch_params:
                        xftr(gsum, **larch_params["xftr"])
                    cauchy_wavelet(gsum, kweight=int(larch_params.get("xftf", {}).get("kweight", self._default_kweight)))
    
                    _store_xafs_outputs(gsum, sum_label)
    
                    # Transmission-like aggregation on RAW frame
                    def _collect_interp_raw(key):
                        out = []
                        for name in members:
                            E_m = aligned_data[name].get("energy_raw")
                            Y_m = aligned_data[name].get(key)
                            if E_m is None or Y_m is None:
                                continue
                            y_interp = _interp_to_grid(E_m, Y_m, group_grid)
                            if y_interp is not None:
                                out.append(y_interp.astype(float))
                        return out
    
                    L_I0  = _collect_interp_raw("I0_raw")
                    L_I1  = _collect_interp_raw("I1_raw")
                    L_I2  = _collect_interp_raw("I2_raw")
                    L_InB = _collect_interp_raw("inb_sum_raw")
                    L_Out = _collect_interp_raw("outb_sum_raw")
    
                    def _stack_or_none(L):
                        if not L:
                            return None
                        return np.vstack([np.asarray(a).ravel() for a in L])
    
                    S_I0  = _stack_or_none(L_I0)
                    S_I1  = _stack_or_none(L_I1)
                    S_I2  = _stack_or_none(L_I2)
                    S_InB = _stack_or_none(L_InB)
                    S_Out = _stack_or_none(L_Out)
    
                    npts = len(group_grid)
    
                    # Determine "keep" region based on valid I0/I1 availability per point (require >= 2 traces)
                    def _counts_and_keep():
                        valid_counts = np.zeros(npts, dtype=int)
                        if S_I0 is not None and S_I1 is not None:
                            c0 = np.sum(np.isfinite(S_I0), axis=0) if S_I0 is not None else np.zeros(npts, int)
                            c1 = np.sum(np.isfinite(S_I1), axis=0) if S_I1 is not None else np.zeros(npts, int)
                            valid_counts = np.minimum(c0, c1)
                        min_valid = 2
                        keep = (valid_counts >= min_valid)
                        return keep, min_valid
    
                    keep, min_valid = _counts_and_keep()
                    if np.any(keep):
                        idx_keep = np.where(keep)[0]
                        lo, hi = int(idx_keep[0]), int(idx_keep[-1]) + 1
                    else:
                        lo, hi = 0, npts
    
                    E_group = group_grid[lo:hi]
    
                    def _nanmedian(S):
                        return np.nanmedian(S[:, lo:hi], axis=0) if S is not None else np.full_like(E_group, np.nan, float)
    
                    def _nanmean(S):
                        return np.nanmean(S[:, lo:hi], axis=0) if S is not None else np.full_like(E_group, np.nan, float)
    
                    I0_group  = _nanmedian(S_I0)
                    I1_group  = _nanmedian(S_I1)
                    I2_group  = _nanmedian(S_I2)
                    InB_group = _nanmean(S_InB)
                    OutB_group= _nanmean(S_Out)
    
                    I0_safe = np.clip(I0_group, eps, np.inf)
                    I1_safe = np.clip(I1_group, eps, np.inf)
                    Itran_group  = np.log(I0_safe / I1_safe)
    
                    # Transmission processing for SUM (use SAME E0 as PFY-like sum)
                    mu_trans_norm_g = None
                    mu_trans_flat_g = None
                    try:
                        gsumT = Group()
                        gsumT.energy = E_group
                        gsumT.mu = Itran_group.copy()
                        gsumT.mu_raw = Itran_group.copy()
                        gsumT.filename = f"{sum_label}__trans"
                        gsumT.filepath = None
    
                        if getattr(gsum, "e0", None) is not None and np.isfinite(getattr(gsum, "e0")):
                            pre_edge(gsumT, e0=float(gsum.e0), **larch_params.get("pre_edge", {}))
                            gsumT.e0 = float(gsum.e0)
                        else:
                            pre_edge(gsumT, **larch_params.get("pre_edge", {}))
    
                        if auto_rbkg:
                            chosen_rbkg_T = _optimize_rbkg_rigorous(
                                gsumT, e0=gsumT.e0,
                                ft_kws=larch_params.get("xftf", {}),
                                autobk_kws=larch_params.get("autobk", {}),
                                bounds=(0.6, 2.2), coarse_steps=61, topk=3,
                                r_gate_mode="fixed", rmax_bg=1.0,
                            )
                            abkT = dict(larch_params.get("autobk", {}))
                            abkT["rbkg"] = float(chosen_rbkg_T)
                            abkT.setdefault("kweight", int(larch_params.get("xftf", {}).get("kweight", 2)))
                            _safe_autobk_local(gsumT, e0=gsumT.e0, user_kws=abkT)
                            setattr(gsumT, "_rbkg_opt", float(chosen_rbkg_T))
                        else:
                            _safe_autobk_local(gsumT, e0=gsumT.e0, user_kws=larch_params.get("autobk", {}))
    
                        xftf(gsumT, **larch_params["xftf"])
                        if "xftr" in larch_params:
                            xftr(gsumT, **larch_params["xftr"])
                        cauchy_wavelet(gsumT, kweight=int(larch_params.get("xftf", {}).get("kweight", self._default_kweight)))
                        _store_xafs_outputs(gsumT, f"{sum_label}__trans")
    
                        setattr(gsum, "_trans", gsumT)
    
                        self.exafs.setdefault(sum_label, {})
                        self.exafs[sum_label]["ft_trans"] = {
                            "k": getattr(gsumT, "k", None),
                            "chi": getattr(gsumT, "chi", None),
                            "r": getattr(gsumT, "r", None),
                            "chir": getattr(gsumT, "chir", None),
                            "chir_mag": getattr(gsumT, "chir_mag", None),
                            "chir_re": getattr(gsumT, "chir_re", None),
                            "chir_im": getattr(gsumT, "chir_im", None),
                            "kweight": int(self._ft_params.get("kweight", getattr(gsumT, "kweight", 2))),
                        }
                        self.exafs[sum_label]["wavelet_trans"] = {
                            "k": getattr(gsumT, "k", None),
                            "r": getattr(gsumT, "wcauchy_r", None),
                            "wcauchy_mag": getattr(gsumT, "wcauchy_mag", None),
                            "wcauchy_re": getattr(gsumT, "wcauchy_re", None),
                            "wcauchy_im": getattr(gsumT, "wcauchy_im", None),
                            "kweight": int(self._ft_params.get("kweight", getattr(gsumT, "kweight", 2))),
                        }
    
                        mu_trans_norm_g = getattr(gsumT, "norm", None)
                        mu_trans_flat_g = getattr(gsumT, "flat", None)
                    except Exception as eTsum:
                        print(f"[WARN] Transmission processing failed for SUM '{sum_label}': {eTsum}")
    
                    # Store SUM record (RAW frame)
                    aligned_data[sum_label] = {
                        "energy": E_group,  # RAW frame for sums
    
                        # PFY-like primary defaults
                        "mutrans": gsum.mu_raw[lo:hi],
                        "mu_raw": gsum.mu_raw[lo:hi],
                        "pre_edge": getattr(gsum, "pre_edge", None),
                        "post_edge": getattr(gsum, "post_edge", None),
                        "bkg": getattr(gsum, "bkg", None),
                        "mu_bkgsub": (gsum.mu - getattr(gsum, "bkg", 0))[lo:hi] if getattr(gsum, "bkg", None) is not None else None,
                        "mu_norm": getattr(gsum, "norm", None)[lo:hi] if getattr(gsum, "norm", None) is not None else None,
                        "mu_flat": getattr(gsum, "flat", None)[lo:hi] if getattr(gsum, "flat", None) is not None else None,
    
                        "filepath": None,
                        "label_safe": sum_label,
                        "label_display": label_display_map.get(sum_label, sum_label),
    
                        # Transmission series (median/mean on RAW)
                        "mu_trans": Itran_group,
                        "mu_trans_raw": Itran_group,
                        "mu_trans_norm": mu_trans_norm_g,
                        "mu_trans_flat": mu_trans_flat_g,
    
                        # For exporter / headers
                        "I0_group": I0_group, "I1_group": I1_group, "I2_group": I2_group,
                        "InB_group_mean": InB_group, "OutB_group_mean": OutB_group,
                        "If_in_group": (InB_group / np.clip(I0_group, eps, np.inf)) if InB_group is not None else None,
                        "If_out_group": (OutB_group / np.clip(I0_group, eps, np.inf)) if OutB_group is not None else None,
                        "Itran_group": Itran_group,
                        "edge_trim": {"lo": lo, "hi": hi, "min_valid": min_valid},
    
                        "group_members": members,
                    }
    
                    groups[sum_label] = gsum
                    e0_dict[sum_label] = gsum.e0
                    sum_label_to_members[sum_label] = members
                    label_display_map[sum_label] = sum_label_raw
    
                except Exception as e_sum:
                    print(f"[ERROR][align_multiple:group] group_index={idx_s}, members={item}: {e_sum}")
                    raise
    
        # =========================
        # Ka1 token warnings
        # =========================
        if (_ka1_inb_hits_total == 0) and (_ka1_outb_hits_total == 0):
            print(f"[WARN] Ka1 token '{self._ka1_token}' matched 0 channels for BOTH InB and OutB across all files (transmission-only datasets will be used).")
        elif (_ka1_inb_hits_total == 0):
            print(f"[WARN] Ka1 token '{self._ka1_token}' matched 0 InB channels across all files.")
        elif (_ka1_outb_hits_total == 0):
            print(f"[WARN] Ka1 token '{self._ka1_token}' matched 0 OutB channels across all files.")
    
        # =========================
        # STORE STATE
        # =========================
        self.groups = groups
        self.e0 = e0_dict
        self._aligned_data = aligned_data
        self._reference = reference
        self._sum_map = sum_label_to_members
        self._plot_mode = plot_mode or "sums_plus_unsummed"
        self._label_map = safe_label_map
        self._label_display_map = label_display_map
    
        try:
            if self.metrics:
                self.metrics_all = pd.concat(self.metrics, names=["label", "channel"])
            else:
                self.metrics_all = None
        except Exception:
            self.metrics_all = None
    
        # ---------------- interactive export (optional) ----------------
        if prompt_save:
            choice = None
            while True:
                print("Save what? [s]ingles / [S]ums / [b]oth (required): ", end="")
                raw = (input() or "").strip().lower()
                if raw in {"s", "single", "singles"}:
                    choice = "singles"; break
                if raw in {"sum", "sums"}:
                    choice = "sums";    break
                if raw in {"b", "both"}:
                    choice = "both";    break
                print("Invalid choice. Please enter 's', 'S', or 'b'.")
    
            outdir = None
            while True:
                raw_dir = (input("Output directory (required): ") or "").strip()
                if raw_dir != "":
                    outdir = raw_dir; break
                print("Output directory cannot be empty. Please provide a path.")
    
            try:
                self.export_groups_xdi_dat(outdir=outdir, include_mu_norm=include_mu_norm_export, which=choice)
            except TypeError:
                aligned_backup = getattr(self, "_aligned_data", None)
                sum_map = getattr(self, "_sum_map", {}) or {}
                sum_keys = set(sum_map.keys())
                try:
                    if choice == "sums":
                        self._aligned_data = {k: v for k, v in aligned_backup.items() if k in sum_keys}
                    elif choice == "singles":
                        self._aligned_data = {k: v for k, v in aligned_backup.items() if k not in sum_keys}
                    self.export_groups_xdi_dat(outdir=outdir, include_mu_norm=include_mu_norm_export)
                finally:
                    self._aligned_data = aligned_backup
    
        # =========================
        # RETURN (restore old behavior)
        # =========================
        if plot_mode == "aligned_mu":
            return aligned_data, e0_dict   # old contract
        return self.groups, self.e0
    
        # =========================
        # ALIGN extras to reference (singles only; sums remain raw by design)
        # =========================
        for label, grp in groups.items():
            if label not in aligned_data:
                continue
            try:
                shift = e0_dict[label] - e0_ref
                E_aligned = grp.energy - shift
    
                rec_tmp = aligned_data[label]
                def _I(src):
                    return _interp(rec_tmp["energy_raw"], src, E_aligned)
    
                I0_a  = _I(rec_tmp["I0_raw"])
                I1_a  = _I(rec_tmp["I1_raw"])
                I2_a  = _I(rec_tmp["I2_raw"])
                InB_a = _I(rec_tmp["inb_sum_raw"])
                OutB_a= _I(rec_tmp["outb_sum_raw"])
    
                I0s = np.clip(I0_a, eps, np.inf)
                I1s = np.clip(I1_a, eps, np.inf)
                mu_trans_aligned = np.log(I0s / I1s)
    
                aligned_data[label].update({
                    "energy": E_aligned,
                    "InB_sum_good": InB_a,
                    "OutB_sum_good": OutB_a,
                    "I0_aligned": I0_a,
                    "I1_aligned": I1_a,
                    "I2_aligned": I2_a,
    
                    # carry PRIMARY diagnostics
                    "mutrans": grp.mu_raw,
                    "mu_raw": grp.mu_raw,
                    "pre_edge": getattr(grp, "pre_edge", None),
                    "post_edge": getattr(grp, "post_edge", None),
                    "bkg": getattr(grp, "bkg", None),
                    "mu_bkgsub": (grp.mu - getattr(grp, "bkg", 0)) if getattr(grp, "bkg", None) is not None else None,
                    "mu_norm": getattr(grp, "norm", None),
                    "mu_flat": getattr(grp, "flat", None),
    
                    "mu_trans": mu_trans_aligned,
                })
            except Exception as e_align:
                print(f"[ERROR][align extras] label='{label}', file='{aligned_data[label].get('filepath','?')}': {e_align}")
                raise
    
        # =========================
        # CREATE SUMMED GROUPS (RAW energy frame; PFY preferred, fallback to Transmission)
        # =========================
        if include_group_sums and summed_groups:
    
            # -----------------------
            # Helper utilities
            # -----------------------
            def _ensure_mono(E, Y=None):
                """Ensure E is monotonically increasing; sort Y accordingly if provided."""
                if E is None or len(E) == 0:
                    return E if Y is None else (E, Y)
                E = np.asarray(E).ravel()
                if not np.all(np.diff(E) >= 0):
                    order = np.argsort(E)
                    E_sorted = E[order]
                    if Y is None:
                        return E_sorted
                    Y = np.asarray(Y).ravel()
                    Y_sorted = Y[order]
                    return E_sorted, Y_sorted
                return (E if Y is None else (np.asarray(E).ravel(), np.asarray(Y).ravel()))
    
            def _get_overlap(energies):
                """Given a list of monotonically increasing energy arrays, return (E_min, E_max) for common overlap."""
                if not energies:
                    return None, None
                starts, ends = [], []
                for e in energies:
                    if e is None or len(e) == 0:
                        continue
                    e = np.asarray(e).ravel()
                    e = _ensure_mono(e)
                    starts.append(e[0])
                    ends.append(e[-1])
                if not starts or not ends:
                    return None, None
                E_min = max(starts)
                E_max = min(ends)
                if E_min >= E_max:
                    return None, None
                return E_min, E_max
    
            def _pick_raw_series(rec: dict):
                """Return raw-like μ(E): prefer PFY mu_raw; fallback to trans raw; then mutrans."""
                y = rec.get("mu_raw")
                if y is not None:
                    return y
                y = rec.get("mu_trans_raw")
                if y is not None:
                    return y
                y = rec.get("mutrans")
                return y
    
            def _interp_to_grid(E_src, Y_src, E_grid):
                """Safe interpolation from (E_src, Y_src) to E_grid."""
                if E_src is None or Y_src is None:
                    return None
                E_src, Y_src = _ensure_mono(E_src, Y_src)
                if E_src is None or len(E_src) == 0 or len(Y_src) == 0:
                    return None
                try:
                    return np.interp(E_grid, E_src, np.asarray(Y_src, dtype=float))
                except Exception:
                    return None
    
            use_first_member_e0 = True
    
            for idx_s, item in enumerate(summed_groups):
                try:
                    # Resolve members and group label
                    if isinstance(item, dict):
                        members_orig = item.get("members", [])
                        user_label = item.get("label", None)
                    else:
                        members_orig = item
                        user_label = (sum_names[idx_s] if (sum_names and idx_s < len(sum_names)) else None)
    
                    members = [safe_label_map.get(m, m) for m in members_orig]
                    if not members:
                        raise ValueError("Empty members in summed_groups entry.")
                    for m in members:
                        if m not in groups:
                            raise ValueError(f"Dataset '{m}' not found among processed labels.")
    
                    # Build per-group overlap grid in RAW frame
                    member_Es = []
                    for m in members:
                        # Prefer the Group's energy (raw), fallback to stored raw energy
                        Ei = getattr(groups[m], "energy", None)
                        if Ei is None:
                            Ei = aligned_data[m].get("energy_raw")
                        if Ei is None:
                            continue
                        Ei = _ensure_mono(Ei)
                        member_Es.append(Ei)
    
                    E_min, E_max = _get_overlap(member_Es)
                    if E_min is None or E_max is None:
                        raise ValueError(f"No common energy overlap among {members}.")
    
                    # Slice each member's RAW energy to overlap and pick the shortest as the group's grid
                    overlap_slices = {}
                    for m, Ei in zip(members, member_Es):
                        mask = (Ei >= E_min) & (Ei <= E_max)
                        Ei_clip = Ei[mask]
                        overlap_slices[m] = Ei_clip
    
                    shortest_member = min(members, key=lambda m: len(overlap_slices.get(m, [])))
                    group_grid = overlap_slices[shortest_member]
                    group_grid = np.asarray(group_grid, dtype=float).ravel()
                    if group_grid is None or len(group_grid) == 0:
                        raise ValueError(f"Overlap grid empty for group {members}.")
    
                    group_grid = np.unique(group_grid)
                    if len(group_grid) < 5:
                        print(f"[WARN] Very short overlap grid ({len(group_grid)} pts) for group {members}.")
    
                    # PFY-like RAW aggregation on the RAW group grid
                    mu_raw_accum, cnt_raw = None, 0
                    for name in members:
                        # Use RAW frame arrays
                        Ei = aligned_data[name].get("energy_raw")
                        yi = _pick_raw_series(aligned_data[name])
                        if (Ei is None) or (yi is None) or (len(Ei) == 0) or (len(yi) == 0):
                            continue
                        yi_interp = _interp_to_grid(Ei, yi, group_grid)
                        if yi_interp is None:
                            continue
                        mu_raw_accum = yi_interp if mu_raw_accum is None else (mu_raw_accum + yi_interp)
                        cnt_raw += 1
    
                    if cnt_raw == 0:
                        raise RuntimeError(f"No valid RAW μ arrays to average for: {members}")
    
                    mu_raw_sum = mu_raw_accum / cnt_raw
    
                    # Create SUM Group (PFY-like) — Larch processing
                    sum_label_raw = user_label if user_label else f"SUM({'+'.join(members)})"
                    sum_label = self._safe_label(sum_label_raw)
    
                    gsum = Group()
                    gsum.energy = group_grid  # RAW frame
                    gsum.mu = mu_raw_sum
                    gsum.mu_raw = mu_raw_sum.copy()
                    gsum.filename = sum_label
                    gsum.filepath = None
    
                    # Decide group's E0 (Option A policy)
                    forced_e0 = None
                    try:
                        if reference is None and use_first_member_e0:
                            forced_e0 = float(e0_dict[members[0]])
                        else:
                            forced_e0 = float(self._reference_e0)
                    except Exception:
                        forced_e0 = None
    
                    if forced_e0 is not None and np.isfinite(forced_e0):
                        pre_edge(gsum, e0=forced_e0, **larch_params.get("pre_edge", {}))
                        gsum.e0 = forced_e0
                    else:
                        pre_edge(gsum, **larch_params.get("pre_edge", {}))  # fallback
    
                    if auto_rbkg:
                        chosen_rbkg = _optimize_rbkg_rigorous(
                            gsum, e0=gsum.e0,
                            ft_kws=larch_params.get("xftf", {}),
                            autobk_kws=larch_params.get("autobk", {}),
                            bounds=(0.6, 2.2), coarse_steps=61, topk=3,
                            r_gate_mode="fixed", rmax_bg=1.0,
                            w_R=0.90, w_B=0.10, w_reg=0.0,
                        )
                        abk_sum = dict(larch_params.get("autobk", {}))
                        abk_sum["rbkg"] = float(chosen_rbkg)
                        abk_sum.setdefault("kweight", int(larch_params.get("xftf", {}).get("kweight", 2)))
                        _safe_autobk_local(gsum, e0=gsum.e0, user_kws=abk_sum)
                        setattr(gsum, "_rbkg_opt", float(chosen_rbkg))
                    else:
                        _safe_autobk_local(gsum, e0=gsum.e0, user_kws=larch_params.get("autobk", {}))
    
                    xftf(gsum, **larch_params["xftf"])
                    if "xftr" in larch_params:
                        xftr(gsum, **larch_params["xftr"])
                    cauchy_wavelet(gsum, kweight=int(larch_params.get("xftf", {}).get("kweight", self._default_kweight)))
    
                    _store_xafs_outputs(gsum, sum_label)
    
                    # Transmission-like aggregation on RAW frame
                    def _collect_interp_raw(key):
                        out = []
                        for name in members:
                            E_m = aligned_data[name].get("energy_raw")
                            Y_m = aligned_data[name].get(key)
                            if E_m is None or Y_m is None:
                                continue
                            y_interp = _interp_to_grid(E_m, Y_m, group_grid)
                            if y_interp is not None:
                                out.append(y_interp.astype(float))
                        return out
    
                    L_I0  = _collect_interp_raw("I0_raw")
                    L_I1  = _collect_interp_raw("I1_raw")
                    L_I2  = _collect_interp_raw("I2_raw")
                    L_InB = _collect_interp_raw("inb_sum_raw")
                    L_Out = _collect_interp_raw("outb_sum_raw")
    
                    def _stack_or_none(L):
                        if not L:
                            return None
                        return np.vstack([np.asarray(a).ravel() for a in L])
    
                    S_I0  = _stack_or_none(L_I0)
                    S_I1  = _stack_or_none(L_I1)
                    S_I2  = _stack_or_none(L_I2)
                    S_InB = _stack_or_none(L_InB)
                    S_Out = _stack_or_none(L_Out)
    
                    npts = len(group_grid)
    
                    # Determine "keep" region based on valid I0/I1 availability per point (require >= 2 traces)
                    def _counts_and_keep():
                        valid_counts = np.zeros(npts, dtype=int)
                        if S_I0 is not None and S_I1 is not None:
                            c0 = np.sum(np.isfinite(S_I0), axis=0) if S_I0 is not None else np.zeros(npts, int)
                            c1 = np.sum(np.isfinite(S_I1), axis=0) if S_I1 is not None else np.zeros(npts, int)
                            valid_counts = np.minimum(c0, c1)
                        min_valid = 2
                        keep = (valid_counts >= min_valid)
                        return keep, min_valid
    
                    keep, min_valid = _counts_and_keep()
                    if np.any(keep):
                        idx_keep = np.where(keep)[0]
                        lo, hi = int(idx_keep[0]), int(idx_keep[-1]) + 1
                    else:
                        lo, hi = 0, npts
    
                    E_group = group_grid[lo:hi]
    
                    def _nanmedian(S):
                        return np.nanmedian(S[:, lo:hi], axis=0) if S is not None else np.full_like(E_group, np.nan, float)
    
                    def _nanmean(S):
                        return np.nanmean(S[:, lo:hi], axis=0) if S is not None else np.full_like(E_group, np.nan, float)
    
                    I0_group  = _nanmedian(S_I0)
                    I1_group  = _nanmedian(S_I1)
                    I2_group  = _nanmedian(S_I2)
                    InB_group = _nanmean(S_InB)
                    OutB_group= _nanmean(S_Out)
    
                    I0_safe = np.clip(I0_group, eps, np.inf)
                    I1_safe = np.clip(I1_group, eps, np.inf)
                    Itran_group  = np.log(I0_safe / I1_safe)
    
                    # Transmission processing for SUM (use SAME E0 as PFY-like sum)
                    mu_trans_norm_g = None
                    mu_trans_flat_g = None
                    try:
                        gsumT = Group()
                        gsumT.energy = E_group
                        gsumT.mu = Itran_group.copy()
                        gsumT.mu_raw = Itran_group.copy()
                        gsumT.filename = f"{sum_label}__trans"
                        gsumT.filepath = None
    
                        if getattr(gsum, "e0", None) is not None and np.isfinite(getattr(gsum, "e0")):
                            pre_edge(gsumT, e0=float(gsum.e0), **larch_params.get("pre_edge", {}))
                            gsumT.e0 = float(gsum.e0)
                        else:
                            pre_edge(gsumT, **larch_params.get("pre_edge", {}))
    
                        if auto_rbkg:
                            chosen_rbkg_T = _optimize_rbkg_rigorous(
                                gsumT, e0=gsumT.e0,
                                ft_kws=larch_params.get("xftf", {}),
                                autobk_kws=larch_params.get("autobk", {}),
                                bounds=(0.6, 2.2), coarse_steps=61, topk=3,
                                r_gate_mode="fixed", rmax_bg=1.0,
                            )
                            abkT = dict(larch_params.get("autobk", {}))
                            abkT["rbkg"] = float(chosen_rbkg_T)
                            abkT.setdefault("kweight", int(larch_params.get("xftf", {}).get("kweight", 2)))
                            _safe_autobk_local(gsumT, e0=gsumT.e0, user_kws=abkT)
                            setattr(gsumT, "_rbkg_opt", float(chosen_rbkg_T))
                        else:
                            _safe_autobk_local(gsumT, e0=gsumT.e0, user_kws=larch_params.get("autobk", {}))
    
                        xftf(gsumT, **larch_params["xftf"])
                        if "xftr" in larch_params:
                            xftr(gsumT, **larch_params["xftr"])
                        cauchy_wavelet(gsumT, kweight=int(larch_params.get("xftf", {}).get("kweight", self._default_kweight)))
                        _store_xafs_outputs(gsumT, f"{sum_label}__trans")
    
                        setattr(gsum, "_trans", gsumT)
    
                        self.exafs.setdefault(sum_label, {})
                        self.exafs[sum_label]["ft_trans"] = {
                            "k": getattr(gsumT, "k", None),
                            "chi": getattr(gsumT, "chi", None),
                            "r": getattr(gsumT, "r", None),
                            "chir": getattr(gsumT, "chir", None),
                            "chir_mag": getattr(gsumT, "chir_mag", None),
                            "chir_re": getattr(gsumT, "chir_re", None),
                            "chir_im": getattr(gsumT, "chir_im", None),
                            "kweight": int(self._ft_params.get("kweight", getattr(gsumT, "kweight", 2))),
                        }
                        self.exafs[sum_label]["wavelet_trans"] = {
                            "k": getattr(gsumT, "k", None),
                            "r": getattr(gsumT, "wcauchy_r", None),
                            "wcauchy_mag": getattr(gsumT, "wcauchy_mag", None),
                            "wcauchy_re": getattr(gsumT, "wcauchy_re", None),
                            "wcauchy_im": getattr(gsumT, "wcauchy_im", None),
                            "kweight": int(self._ft_params.get("kweight", getattr(gsumT, "kweight", 2))),
                        }
    
                        mu_trans_norm_g = getattr(gsumT, "norm", None)
                        mu_trans_flat_g = getattr(gsumT, "flat", None)
                    except Exception as eTsum:
                        print(f"[WARN] Transmission processing failed for SUM '{sum_label}': {eTsum}")
    
                    # Store SUM record (RAW frame)
                    aligned_data[sum_label] = {
                        "energy": E_group,  # RAW frame for sums
    
                        # PFY-like primary defaults
                        "mutrans": gsum.mu_raw[lo:hi],
                        "mu_raw": gsum.mu_raw[lo:hi],
                        "pre_edge": getattr(gsum, "pre_edge", None),
                        "post_edge": getattr(gsum, "post_edge", None),
                        "bkg": getattr(gsum, "bkg", None),
                        "mu_bkgsub": (gsum.mu - getattr(gsum, "bkg", 0))[lo:hi] if getattr(gsum, "bkg", None) is not None else None,
                        "mu_norm": getattr(gsum, "norm", None)[lo:hi] if getattr(gsum, "norm", None) is not None else None,
                        "mu_flat": getattr(gsum, "flat", None)[lo:hi] if getattr(gsum, "flat", None) is not None else None,
    
                        "filepath": None,
                        "label_safe": sum_label,
                        "label_display": label_display_map.get(sum_label, sum_label),
    
                        # Transmission series (median/mean on RAW)
                        "mu_trans": Itran_group,
                        "mu_trans_raw": Itran_group,
                        "mu_trans_norm": mu_trans_norm_g,
                        "mu_trans_flat": mu_trans_flat_g,
    
                        # For exporter / headers
                        "I0_group": I0_group, "I1_group": I1_group, "I2_group": I2_group,
                        "InB_group_mean": InB_group, "OutB_group_mean": OutB_group,
                        "If_in_group": (InB_group / np.clip(I0_group, eps, np.inf)) if InB_group is not None else None,
                        "If_out_group": (OutB_group / np.clip(I0_group, eps, np.inf)) if OutB_group is not None else None,
                        "Itran_group": Itran_group,
                        "edge_trim": {"lo": lo, "hi": hi, "min_valid": min_valid},
    
                        "group_members": members,
                    }
    
                    groups[sum_label] = gsum
                    e0_dict[sum_label] = gsum.e0
                    sum_label_to_members[sum_label] = members
                    label_display_map[sum_label] = sum_label_raw
    
                except Exception as e_sum:
                    print(f"[ERROR][align_multiple:group] group_index={idx_s}, members={item}: {e_sum}")
                    raise
    
        # =========================
        # Ka1 token warnings
        # =========================
        if (_ka1_inb_hits_total == 0) and (_ka1_outb_hits_total == 0):
            print(f"[WARN] Ka1 token '{self._ka1_token}' matched 0 channels for BOTH InB and OutB across all files (transmission-only datasets will be used).")
        elif (_ka1_inb_hits_total == 0):
            print(f"[WARN] Ka1 token '{self._ka1_token}' matched 0 InB channels across all files.")
        elif (_ka1_outb_hits_total == 0):
            print(f"[WARN] Ka1 token '{self._ka1_token}' matched 0 OutB channels across all files.")
    
        # =========================
        # STORE STATE
        # =========================
        self.groups = groups
        self.e0 = e0_dict
        self._aligned_data = aligned_data
        self._reference = reference
        self._sum_map = sum_label_to_members
        self._plot_mode = plot_mode or "sums_plus_unsummed"
        self._label_map = safe_label_map
        self._label_display_map = label_display_map
    
        try:
            if self.metrics:
                self.metrics_all = pd.concat(self.metrics, names=["label", "channel"])
            else:
                self.metrics_all = None
        except Exception:
            self.metrics_all = None
    
        # ---------------- interactive export (optional) ----------------
        if prompt_save:
            choice = None
            while True:
                print("Save what? [s]ingles / [S]ums / [b]oth (required): ", end="")
                raw = (input() or "").strip().lower()
                if raw in {"s", "single", "singles"}:
                    choice = "singles"; break
                if raw in {"sum", "sums"}:
                    choice = "sums";    break
                if raw in {"b", "both"}:
                    choice = "both";    break
                print("Invalid choice. Please enter 's', 'S', or 'b'.")
    
            outdir = None
            while True:
                raw_dir = (input("Output directory (required): ") or "").strip()
                if raw_dir != "":
                    outdir = raw_dir; break
                print("Output directory cannot be empty. Please provide a path.")
    
            try:
                self.export_groups_xdi_dat(outdir=outdir, include_mu_norm=include_mu_norm_export, which=choice)
            except TypeError:
                aligned_backup = getattr(self, "_aligned_data", None)
                sum_map = getattr(self, "_sum_map", {}) or {}
                sum_keys = set(sum_map.keys())
                try:
                    if choice == "sums":
                        self._aligned_data = {k: v for k, v in aligned_backup.items() if k in sum_keys}
                    elif choice == "singles":
                        self._aligned_data = {k: v for k, v in aligned_backup.items() if k not in sum_keys}
                    self.export_groups_xdi_dat(outdir=outdir, include_mu_norm=include_mu_norm_export)
                finally:
                    self._aligned_data = aligned_backup
    
        # =========================
        # RETURN (restore old behavior)
        # =========================
        if plot_mode == "aligned_mu":
            return aligned_data, e0_dict   # old contract
        return self.groups, self.e0
    
    
    # ------------------- Exporter (write-only) -------------------
    def export_groups_xdi_dat(
        self,
        outdir: str = ".",
        *,
        include_mu_norm: bool = False,
        which: str = "both",   # 'sums' | 'singles' | 'both'
    ):
        """
        Write one .dat per SUMMED GROUP and per INDIVIDUAL scan using arrays computed & stored by align_multiple().
    
        Parameters
        ----------
        outdir : str
            Output directory (created if needed).
        include_mu_norm : bool
            If True, include mu_norm as the last column when available.
        which : {'sums','singles','both'}
            Which set of outputs to write.
    
        SUMMED GROUPS columns:
          1: energy
          2: I0 (group median)
          3: I1 (group median)
          4: I2 (group median)
          5: InB_Sum_good_channel (group mean)
          6: OutB_Sum_good_channel (group mean)
          7: If_in  (InB/I0)
          8: If_out (OutB/I0)
          9: Itran  (ln(I0/I1))
         10: mu_norm (optional, if present)
    
        INDIVIDUAL scans columns (same semantics except arrays are aligned per-scan):
          1: energy
          2: I0 (aligned)
          3: I1 (aligned)
          4: I2 (aligned)
          5: InB_Sum_good_channel (aligned)
          6: OutB_Sum_good_channel (aligned)
          7: If_in  (InB/I0)
          8: If_out (OutB/I0)
          9: Itran  (ln(I0/I1))
         10: mu_norm (optional, if present)
        """
        import os, time
        import numpy as np
        from typing import List, Dict, Any
    
        which = (which or "both").lower()
        valid = {"sums", "singles", "both"}
        if which not in valid:
            raise ValueError(f"'which' must be one of {valid}, got {which!r}")
    
        if not hasattr(self, "_aligned_data"):
            raise RuntimeError("Run align_multiple(...) before exporting.")
    
        os.makedirs(outdir, exist_ok=True)
    
        aligned = self._aligned_data
        sum_map = getattr(self, "_sum_map", {}) or {}
        sum_keys = set(sum_map.keys())
    
        # -----------------------------
        # Shared helpers (no duplication)
        # -----------------------------
        def _parse_header_meta(header_lines_raw: List[str]) -> Dict[str, Any]:
            """Parse metadata preserving tabs/spacing for ROI/Scanned Regions."""
            meta = {
                "element_symbol": None, "element_edge": None, "date": None,
                "sr_current": None, "settling_time": None,
                "amp_i0": None, "amp_i1": None, "amp_i2": None,
                "endstation_name": None,
                "roi_lines": [], "scan_lines": [],
            }
            def _uncomment(s: str) -> str:
                if s.startswith("# "): return s[2:]
                if s.startswith("#"):  return s[1:]
                return s
            raw = [_uncomment(h) for h in header_lines_raw]
    
            SECTION_HEADERS = {
                "Endstation", "Regions Of Interest", "Scanned Regions",
                "InB channels found", "InB good channels", "OutB channels found", "OutB good channels",
                "If_* = fluorescence(sum_good_side, I0)", "Itran = ln(I0/I1)",
            }
            def _starts_new(s: str) -> bool:
                s2 = s.strip()
                if s2 in SECTION_HEADERS: return True
                if s2.startswith("Column."): return True
                if s2.startswith("I0 source column:") or s2.startswith("I1 source column:") or s2.startswith("I2 source column:"): return True
                return False
            def _after_colon_keep(s: str):
                parts = s.split(":", 1)
                return parts[1].lstrip() if len(parts) == 2 else None
    
            i, n = 0, len(raw)
            while i < n:
                line = raw[i]
                ls = line.lstrip()
                if ls.startswith("Element.symbol:"): meta["element_symbol"] = _after_colon_keep(ls)
                elif ls.startswith("Element.edge:"):  meta["element_edge"]  = _after_colon_keep(ls)
                elif ls.startswith("Date:"):          meta["date"]          = _after_colon_keep(ls)
                elif ls.startswith("SR current:"):     meta["sr_current"]    = _after_colon_keep(ls)
                elif ls.startswith("Settling time:"):  meta["settling_time"] = _after_colon_keep(ls)
                elif ls.startswith("I0 amplifier sensitivity:"): meta["amp_i0"] = _after_colon_keep(ls)
                elif ls.startswith("I1 amplifier sensitivity:"): meta["amp_i1"] = _after_colon_keep(ls)
                elif ls.startswith("I2 amplifier sensitivity:"): meta["amp_i2"] = _after_colon_keep(ls)
                elif ls.strip() == "Endstation":
                    j = i + 1
                    if j < n:
                        nxt = raw[j]
                        if (nxt.strip() != "") and (not _starts_new(nxt)):
                            meta["endstation_name"] = nxt; i = j
                elif ls.strip() == "Regions Of Interest":
                    j = i + 1
                    while j < n:
                        l2 = raw[j]
                        if (l2.strip() == "") or _starts_new(l2): break
                        meta["roi_lines"].append(l2); j += 1
                    i = j - 1
                elif ls.strip() == "Scanned Regions":
                    j = i + 1
                    while j < n:
                        l2 = raw[j]
                        if (l2.strip() == "") or _starts_new(l2): break
                        meta["scan_lines"].append(l2); j += 1
                    i = j - 1
                i += 1
            return meta
    
        def _build_header(*, scan_name: str, meta: Dict[str, Any],
                          inb_found_txt: str, inb_good_txt: str,
                          out_found_txt: str, out_good_txt: str,
                          include_mu_norm_eff: bool) -> List[str]:
            hdr = []
            hdr.append("XDI/1.0")
            hdr.append(f"Element.symbol: {meta.get('element_symbol')}")
            hdr.append(f"Element.edge: {meta.get('element_edge')}")
            hdr.append(f"Scan: {scan_name}  #1")
            date_val = meta.get("date") or time.strftime("%Y %m %d %H:%M:%S")
            hdr.append(f"Date: {date_val}")
            if meta.get("sr_current"):    hdr.append(f"SR current: {meta['sr_current']}")
            if meta.get("settling_time"): hdr.append(f"Settling time: {meta['settling_time']}")
            if meta.get("amp_i0"):        hdr.append(f"I0 amplifier sensitivity: {meta['amp_i0']}")
            if meta.get("amp_i1"):        hdr.append(f"I1 amplifier sensitivity: {meta['amp_i1']}")
            if meta.get("amp_i2"):        hdr.append(f"I2 amplifier sensitivity: {meta['amp_i2']}")
            hdr.append("Endstation")
            hdr.append(meta.get("endstation_name") or "")
            hdr.append("Regions Of Interest")
            for l in meta.get("roi_lines", []) or []:
                hdr.append(l)
            hdr.append("Scanned Regions")
            for l in meta.get("scan_lines", []) or []:
                hdr.append(l)
            hdr.append("")
            hdr.append(f"InB channels found: {inb_found_txt or 'none'}")
            hdr.append(f"InB good channels: {inb_good_txt or 'none'}")
            hdr.append(f"OutB channels found: {out_found_txt or 'none'}")
            hdr.append(f"OutB good channels: {out_good_txt or 'none'}")
            hdr.append("If_* = fluorescence(sum_good_side, I0)")
            hdr.append("Itran = ln(I0/I1)")
            hdr.append("I0 source column: I0Detector_DarkCorrect")
            hdr.append("I1 source column: I1Detector_DarkCorrect")
            hdr.append("I2 source column: I2Detector_DarkCorrect")
            hdr.append("")
            hdr.append("Column.1: energy")
            hdr.append("Column.2: I0")
            hdr.append("Column.3: I1")
            hdr.append("Column.4: I2")
            hdr.append("Column.5: InB_Sum_good_channel")
            hdr.append("Column.6: OutB_Sum_good_channel")
            hdr.append("Column.7: If_in (fluorescence(InB_Sum_good_channel, I0))")
            hdr.append("Column.8: If_out (fluorescence(OutB_Sum_good_channel, I0))")
            hdr.append("Column.9: Itran (ln(I0/I1))")
            if include_mu_norm_eff:
                hdr.append("Column.10: mu_norm")
            return hdr
    
        def _write_rows(fout, E, I0, I1, I2, InB, OutB, If_in, If_out, Itran, mu_norm, include_mu_norm_eff: bool):
            n = len(E)
            if include_mu_norm_eff and (mu_norm is not None):
                for k in range(n):
                    fout.write(
                        f"{E[k]:.8f} {I0[k]:.8g} {I1[k]:.8g} {I2[k]:.8g} "
                        f"{InB[k]:.8g} {OutB[k]:.8g} {If_in[k]:.8g} {If_out[k]:.8g} {Itran[k]:.8g} {mu_norm[k]:.8g}\n"
                    )
            else:
                for k in range(n):
                    fout.write(
                        f"{E[k]:.8f} {I0[k]:.8g} {I1[k]:.8g} {I2[k]:.8g} "
                        f"{InB[k]:.8g} {OutB[k]:.8g} {If_in[k]:.8g} {If_out[k]:.8g} {Itran[k]:.8g}\n"
                    )
    
        # ============================================================
        # 1) EXPORT SUMMED GROUPS (if requested)
        # ============================================================
        if which in {"sums", "both"}:
            for sum_label, members in sum_map.items():
                rec_sum = aligned.get(sum_label, None)
                if rec_sum is None:
                    raise RuntimeError(f"Missing group record for '{sum_label}'")
    
                # grab stored arrays (group aggregates)
                E     = rec_sum.get("energy")
                I0_g  = rec_sum.get("I0_group")
                I1_g  = rec_sum.get("I1_group")
                I2_g  = rec_sum.get("I2_group")
                InB_g = rec_sum.get("InB_group_mean")
                OutB_g= rec_sum.get("OutB_group_mean")
                If_in_g  = rec_sum.get("If_in_group")
                If_out_g = rec_sum.get("If_out_group")
                Itran_g  = rec_sum.get("Itran_group")
                mu_norm_g = rec_sum.get("mu_norm", None)
    
                missing = [k for k, v in [
                    ("energy",E),("I0_group",I0_g),("I1_group",I1_g),("I2_group",I2_g),
                    ("InB_group_mean",InB_g),("OutB_group_mean",OutB_g),
                    ("If_in_group",If_in_g),("If_out_group",If_out_g),("Itran_group",Itran_g)
                ] if v is None]
                if missing:
                    raise RuntimeError(f"Missing stored arrays {missing} for group '{sum_label}'. Re-run align_multiple.")
    
                # header channel summaries
                inb_found_txt = ", ".join(rec_sum.get("InB_found", [])) if rec_sum.get("InB_found") else "none"
                inb_good_txt  = ", ".join(rec_sum.get("InB_good",  [])) if rec_sum.get("InB_good")  else "none"
                out_found_txt = ", ".join(rec_sum.get("OutB_found", [])) if rec_sum.get("OutB_found") else "none"
                out_good_txt  = ", ".join(rec_sum.get("OutB_good",  [])) if rec_sum.get("OutB_good")  else "none"
    
                # metadata from first real member
                meta = None
                for m in members:
                    p = aligned.get(m, {}).get("filepath", None)
                    if p and os.path.isfile(p):
                        loader = self.XASDataLoader(p).load()
                        if getattr(loader, "header_raw", None):
                            meta = _parse_header_meta(loader.header_raw)
                        break
                if meta is None:
                    meta = {
                        "element_symbol": "Unknown", "element_edge": "Unknown",
                        "date": time.strftime("%Y %m %d %H:%M:%S"),
                        "sr_current": None, "settling_time": None,
                        "amp_i0": None, "amp_i1": None, "amp_i2": None,
                        "endstation_name": None, "roi_lines": [], "scan_lines": [],
                    }
    
                include_mu_norm_eff = include_mu_norm and (mu_norm_g is not None)
                hdr = _build_header(
                    scan_name=sum_label, meta=meta,
                    inb_found_txt=inb_found_txt, inb_good_txt=inb_good_txt,
                    out_found_txt=out_found_txt, out_good_txt=out_good_txt,
                    include_mu_norm_eff=include_mu_norm_eff
                )
    
                out_path = os.path.join(outdir, f"{self._safe_label(sum_label)}.dat")
                with open(out_path, "w", encoding="utf-8") as fout:
                    for line in hdr:
                        fout.write(f"# {line}\n")
                    _write_rows(fout, E, I0_g, I1_g, I2_g, InB_g, OutB_g, If_in_g, If_out_g, Itran_g, mu_norm_g, include_mu_norm_eff)
    
        # ============================================================
        # 2) EXPORT INDIVIDUAL SCANS (if requested)
        # ============================================================
        if which in {"singles", "both"}:
            individual_labels = [lbl for lbl in aligned.keys() if lbl not in sum_keys]
            for lbl in individual_labels:
                rec = aligned[lbl]
    
                # Required aligned arrays for individuals
                E     = rec.get("energy")
                I0    = rec.get("I0_aligned")
                I1    = rec.get("I1_aligned")
                I2    = rec.get("I2_aligned")
                InB   = rec.get("InB_sum_good")
                OutB  = rec.get("OutB_sum_good")
                mu_norm = rec.get("mu_norm", None)
    
                # Skip if core arrays are missing
                if any(x is None for x in (E, I0, I1, I2, InB, OutB)):
                    continue
    
                # Derived
                eps = 1e-12
                I0s = np.clip(I0, eps, np.inf)
                I1s = np.clip(I1, eps, np.inf)
                If_in  = InB  / I0s
                If_out = OutB / I0s
                Itran  = np.log(I0s / I1s)
    
                # Channel summaries
                inb_found_txt = ", ".join(rec.get("InB_channels_all", [])) if rec.get("InB_channels_all") else "none"
                inb_good_txt  = ", ".join(rec.get("InB_channels_good", [])) if rec.get("InB_channels_good") else "none"
                out_found_txt = ", ".join(rec.get("OutB_channels_all", [])) if rec.get("OutB_channels_all") else "none"
                out_good_txt  = ", ".join(rec.get("OutB_channels_good", [])) if rec.get("OutB_channels_good") else "none"
    
                # Metadata from the original file
                meta = None
                p = rec.get("filepath", None)
                if p and os.path.isfile(p):
                    loader = self.XASDataLoader(p).load()
                    if getattr(loader, "header_raw", None):
                        try:
                            meta = _parse_header_meta(loader.header_raw)
                        except Exception:
                            meta = None
                if meta is None:
                    meta = {
                        "element_symbol": "Unknown", "element_edge": "Unknown",
                        "date": time.strftime("%Y %m %d %H:%M:%S"),
                        "sr_current": None, "settling_time": None,
                        "amp_i0": None, "amp_i1": None, "amp_i2": None,
                        "endstation_name": None, "roi_lines": [], "scan_lines": [],
                    }
    
                include_mu_norm_eff = include_mu_norm and (mu_norm is not None)
                scan_name = rec.get("label_display") or lbl
                hdr = _build_header(
                    scan_name=scan_name, meta=meta,
                    inb_found_txt=inb_found_txt, inb_good_txt=inb_good_txt,
                    out_found_txt=out_found_txt, out_good_txt=out_good_txt,
                    include_mu_norm_eff=include_mu_norm_eff
                )
    
                # Filename uses display label when available
                try:
                    out_label = self._pretty_label_for_save(lbl, use_display_label=True)
                except Exception:
                    out_label = lbl
                out_path = os.path.join(outdir, f"{out_label}.dat")
    
                with open(out_path, "w", encoding="utf-8") as fout:
                    for line in hdr:
                        fout.write(f"# {line}\n")
                    _write_rows(fout, E, I0, I1, I2, InB, OutB, If_in, If_out, Itran, mu_norm, include_mu_norm_eff)
    
    
    # -----------------------------------------------------------------------------
    # Helpers for saving (single copy)
    # -----------------------------------------------------------------------------
    def _safe_label(self, s: str) -> str:
        base = os.path.splitext(os.path.basename(s or "unnamed"))[0]
        out = re.sub(r"[^\w]+", "_", base, flags=re.UNICODE)
        out = re.sub(r"_+", "_", out).strip("_")
        return out or "dataset"
    
    def _resolve_signal_array(self, label_safe: str, ykey: str) -> Optional[np.ndarray]:
        """
        Resolve requested signal array from _aligned_data by a generic ykey.
        Supported ykey aliases:
           mu->mutrans, mutrans, mu_raw, mu_bkgsub, norm->mu_norm, flat->mu_flat,
           bkg, pre_edge, post_edge
        """
        if not hasattr(self, "_aligned_data"):
            raise RuntimeError("No aligned data available. Run align_multiple first.")
        rec = self._aligned_data.get(label_safe)
        if rec is None:
            return None
    
        # Normalize keys
        key_map = {
            "mu": "mutrans",
            "mutrans": "mutrans",
            "mu_raw": "mu_raw",
            "mu_bkgsub": "mu_bkgsub",
            "norm": "mu_norm",
            "flat": "mu_flat",
            "bkg": "bkg",
            "pre_edge": "pre_edge",
            "post_edge": "post_edge",
        }
        store_key = key_map.get(ykey, ykey)
        return rec.get(store_key)
    
    def _pretty_label_for_save(self, label_safe: str, use_display_label: bool = True) -> str:
        if use_display_label and hasattr(self, "_label_display_map"):
            disp = self._label_display_map.get(label_safe)
            if disp:
                return self._safe_label(disp)
        return label_safe
    
    def save_line(self, x: np.ndarray, y: np.ndarray, path: str, header: Optional[str] = None) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            if header:
                for line in header.splitlines():
                    f.write(f"# {line}\n")
            for xi, yi in zip(x, y):
                f.write(f"{xi:.8f} {yi:.8g}\n")
    
    def get_metrics(self, labels: Optional[Iterable[str]] = None, as_frame: bool = True):
        """
        Return per-label fluorescence-channel metrics.
    
        labels=None -> all labels.
        as_frame=True -> concatenated DataFrame (metrics_all subset).
        as_frame=False -> dict[label -> DataFrame].
        """
        import pandas as pd
        if labels is None:
            labels = list(self.metrics.keys())
        elif isinstance(labels, str):
            labels = [labels]
    
        if not as_frame:
            return {lbl: self.metrics.get(lbl) for lbl in labels}
    
        # build subset of metrics_all or concat requested labels
        if getattr(self, "metrics_all", None) is not None:
            wanted = [lbl for lbl in labels if lbl in self.metrics]
            if not wanted:
                return pd.DataFrame()
            return self.metrics_all.loc[wanted]
        else:
            frames = []
            for lbl in labels:
                df = self.metrics.get(lbl)
                if df is not None:
                    df = df.copy()
                    df.insert(0, "label", lbl)
                    frames.append(df)
            return pd.concat(frames, axis=0) if frames else pd.DataFrame()
    
    def save_metrics_csv(self, path: str, labels: Optional[Iterable[str]] = None) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        df = self.get_metrics(labels=labels, as_frame=True)
        df.to_csv(path)
        return os.path.abspath(path)
    
    def save_signals(
        self,
        folder: str = ".",
        labels: Optional[Iterable[str]] = None,
        y: Union[str, List[str]] = "mu",
        filename_template: str = "{y}_{label}.dat",
        use_display_label: bool = True,
        summed_only: bool = False,
        skip_missing: bool = True,
        include_header: bool = True,
    ) -> None:
        """
        Save aligned X(E) with chosen signal(s).
    
        Parameters
        ----------
        folder : str
            Output directory.
        labels : iterable of safe labels (default: all).
        y : str or list[str]
            One or more of: 'mu'/'mutrans', 'mu_raw', 'mu_bkgsub', 'norm', 'flat', 'bkg', 'pre_edge', 'post_edge'.
        filename_template : str
            Template for file name. Available fields: {y}, {label}.
        use_display_label : bool
            Use the display/custom label (filename-safe) when available.
        summed_only : bool
            If True, restrict to sum groups only.
        skip_missing : bool
            If True, skip when the requested y is not available (default). Otherwise, raise.
        include_header : bool
            If True, write a short header with metadata.
        """
        import os
        if not hasattr(self, "_aligned_data"):
            raise RuntimeError("No aligned data available. Run align_multiple first.")
    
        ys = y if isinstance(y, list) else [y]
        all_labels = list(self._aligned_data.keys())
        if labels is not None:
            wanted = [lbl for lbl in labels if lbl in self._aligned_data]
        else:
            wanted = all_labels
    
        if summed_only:
            # Only labels that exist in _sum_map keys
            sum_keys = set(getattr(self, "_sum_map", {}).keys())
            wanted = [lbl for lbl in wanted if lbl in sum_keys]
    
        for lbl in wanted:
            rec = self._aligned_data[lbl]
            energy = rec.get("energy")
            if energy is None:
                continue
            for yk in ys:
                arr = self._resolve_signal_array(lbl, yk)
                if arr is None:
                    if skip_missing:
                        continue
                    raise ValueError(f"Requested '{yk}' not available for label '{lbl}'")
                out_label = self._pretty_label_for_save(lbl, use_display_label=use_display_label)
                fname = filename_template.format(y=yk, label=out_label)
                path = os.path.join(folder, fname)
                header = None
                if include_header:
                    src = rec.get("filepath") or ""
                    disp = rec.get("label_display") or out_label
                    header = (
                        f"label={disp}\n"
                        f"safe_label={lbl}\n"
                        f"source={src}\n"
                        f"signal={yk}\n"
                        f"reference={self._reference}"
                    )
                self.save_line(energy, arr, path, header=header)
    
    # Backward compatibility: keep your old name, route to new function
    def save_mu(self, folder: str = ".", labels: Optional[List[str]] = None) -> None:
        self.save_signals(folder=folder, labels=labels, y="mu")


    # ------------------- Fourier & Wavelet -------------------
    def k_to_r(
        self, label: str, kmin=None, kmax=None, kweight: int = 2, dk: float = 4,
        window: str = "hanning", rmax_out=None
    ) -> Dict[str, np.ndarray]:
        """Forward FT: χ(k) → χ(R)."""
        g = self.groups[label]
        xftf(g, kmin=kmin, kmax=kmax, kweight=kweight, dk=dk, window=window, rmax_out=rmax_out)
        self.exafs.setdefault(label, {})
        self.exafs[label]["k_to_r"] = {
            "R": g.r,
            "chir_mag": np.abs(g.chir),
            "chir_re": g.chir_re,
            "chir_im": g.chir_im,
            "k": g.k,
            "chi": g.chi,
        }
        return self.exafs[label]["k_to_r"]

    def r_to_k(
        self, label: str, rmin: float, rmax: float, dr: float = 0.25, dk: float = 4,
        kweight: int = 2, rmax_out=None, window: str = "hanning"
    ) -> Dict[str, np.ndarray]:
        """Back-transform an R window: χ(R) → χ(k)."""
        g = self.groups[label]
        xftf(g, kweight=kweight, rmax_out=rmax_out, dk=dk, window=window)  # ensure χ(R) is present
        xftr(g, rmin=rmin, rmax=rmax, dr=dr, window=window)               # now back-transform
        self.exafs.setdefault(label, {})
        self.exafs[label]["r_to_k"] = {
            "r": g.r, "chir": g.chir,
            "k": g.q, "chi_from_rwin": g.chiq,
            "rmin": rmin, "rmax": rmax, "kweight": kweight
        }
        return self.exafs[label]["r_to_k"]

    def overlay_k_shells(
        self, label: str, shells: List[Tuple[float, float]],
        kweight: int = 2, dk: float = 1.0,
        window: str = "hanning", rmax_out=None, offset: float = 0.0
    ):
        """Compute and store R-space overlays for multiple (kmin, kmax) shells."""
        g = self.groups[label]
        overlay_data = {}
        for i, (kmin, kmax) in enumerate(shells):
            xftf(g, kmin=kmin, kmax=kmax, kweight=kweight, dk=dk, window=window, rmax_out=rmax_out)
            overlay_data[(kmin, kmax)] = {"r": g.r, "chir": np.abs(g.chir) + i * offset}
        self.exafs.setdefault(label, {})
        self.exafs[label]["overlay_k"] = overlay_data
        return overlay_data

    def overlay_r_shells(
        self, label: str, shells: List[Tuple[float, float]],
        dr: float = 0.25,
        window: str = "hanning", offset: float = 0.0
    ):
        """Compute and store χ(k) overlays for multiple (rmin, rmax) windows."""
        g = self.groups[label]
        overlay_data = {}
        # Ensure χ(R) exists with latest FT
        xftf(g, window=window)
        for i, (rmin, rmax) in enumerate(shells):
            xftr(g, rmin=rmin, rmax=rmax, dr=dr, window=window)
            overlay_data[(rmin, rmax)] = {"k": g.q, "chi": g.chiq + i * offset}
        self.exafs.setdefault(label, {})
        self.exafs[label]["overlay_r"] = overlay_data
        return overlay_data

    def wavelet(self, label: str, kmin=None, kmax=None, dk: float = 1,
                window: str = "hanning", rmax_out=None, kweight: int = 2):
        """Compute FT and Cauchy wavelet transform for χ(k)."""
        g = self.groups[label]
        self.exafs.setdefault(label, {})
        xftf(g, kmin=kmin, kmax=kmax, kweight=kweight, dk=dk, window=window, rmax_out=rmax_out)
        cauchy_wavelet(g, kweight=kweight)
        self.exafs[label]["wavelet"] = {
            "k": g.k, "r": g.wcauchy_r,
            "wcauchy_mag": g.wcauchy_mag, "wcauchy_re": g.wcauchy_re,
        }
        return self.exafs[label]["wavelet"]


    # ------------------- Plotting -------------------
    def plot_good_channels(
        self, label: str,
        show_sum: bool = True,
        show_good_only: bool = False,
        normalize_each: bool = True,
        alpha_rejected: float = 0.25,
        # --- layout controls ---
        ncols: int = 6,
        sharex: bool = True,
        sharey: bool = True,
        figsize_per_ax: Tuple[float, float] = (3.0, 2.0),
        titlesize: int = 8,
        ticksize: int = 7,
        # --- colors ---
        good_color: str = "green",
        bad_color: str = "black",
    ) -> None:
        """
        Clean, safe, corrected version.
        """
    
        import numpy as np
        import matplotlib.pyplot as plt
        import math
        import re
    
        # ==========================
        # Resolve label
        # ==========================
        if label not in getattr(self, "_channel_eval", {}):
            safe = getattr(self, "_label_map", {}).get(label)
            if safe and (safe in getattr(self, "_channel_eval", {})):
                label = safe
    
        if label not in getattr(self, "_channel_eval", {}):
            raise ValueError(f"No channel evaluation found for '{label}'. Run align_multiple() first.")
    
        info = self._channel_eval[label]
        energy = np.asarray(info["energy"], float)
        uidx = info.get("uidx", None)
        data = info["data"]
        channels_all = info.get("channels", [])
        good_list = info.get("good", []) or []
        good_set = set([c for c in good_list if c in data.columns])
        e0 = info.get("e0", None)
    
        # ==========================
        # Channel selection
        # ==========================
        if isinstance(channels_all, str):
            channels_all = []
        else:
            channels_all = [c for c in channels_all if c in data.columns]
    
        # fallback to ROI inference
        if not channels_all:
            ROI_PREFIX_RE = re.compile(r"^([A-Za-z0-9]+_Ka1_spectra)")
            family_prefix = None
    
            # choose a family prefix
            for c in (list(good_set) + list(data.columns)):
                m = ROI_PREFIX_RE.match(c)
                if m:
                    family_prefix = m.group(1)
                    break
    
            # choose InB/OutB side
            side = None
            for c in (list(good_set) + list(data.columns)):
                if "InB" in c:
                    side = "InB"
                    break
                if "OutB" in c:
                    side = "OutB"
                    break
    
            def _ok(col):
                if family_prefix and not col.startswith(family_prefix):
                    return False
                if side and (side not in col):
                    return False
                if re.search(
                    r"(Dead|Rate|Status|Count|Counts|Scaler|Slow|Fast|Live|Real|Time|Std|Err|Sigma|Var)",
                    col,
                    flags=re.I,
                ):
                    return False
                return True
    
            channels_all = [c for c in data.columns if _ok(c)]
    
            def _sort_key(c):
                m = re.search(r"(\d+)$", c)
                return (re.sub(r"\d+$", "", c), int(m.group(1)) if m else 0, c)
    
            channels_all = sorted(set(channels_all), key=_sort_key)
    
        # final choose channels
        channels = [ch for ch in channels_all if ch in good_set] if show_good_only else channels_all[:]
    
        n_ch = len(channels)
        nplots = n_ch + (1 if show_sum else 0)
    
        # ==========================
        # Layout
        # ==========================
        ncols = max(1, int(ncols))
        nrows = int(math.ceil(max(1, nplots) / ncols))
    
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(figsize_per_ax[0] * ncols, figsize_per_ax[1] * nrows),
            sharex=sharex, sharey=sharey,
        )
        axes = np.atleast_1d(axes).ravel()
    
        ymins, ymaxs = [], []
    
        # ==========================
        # SAFE INDEXING FUNCTION
        # ==========================
        def safe_index(arr):
            """Always return array indexed by uidx or fallback safely."""
            arr = np.asarray(arr, float)
            if uidx is None:
                return arr
            try:
                return arr[uidx]
            except Exception:
                return arr[: len(energy)]
    
        # ==========================
        # Plot channels
        # ==========================
        for i, ch in enumerate(channels):
            if i >= len(axes):
                break
    
            ax = axes[i]
    
            raw_y = data[ch].to_numpy(float)
            y = safe_index(raw_y)
    
            if normalize_each:
                m = np.nanmax(np.abs(y))
                if m and np.isfinite(m):
                    y = y / m
    
            color = good_color if ch in good_set else bad_color
            alpha = 1.0 if ch in good_set else alpha_rejected
    
            ax.plot(energy, y, color=color, alpha=alpha, lw=1.0)
    
            if e0 is not None and np.isfinite(e0):
                ax.axvline(e0, color="k", ls="--", lw=0.7)
    
            ax.set_title(ch, fontsize=titlesize)
            ax.tick_params(axis="both", labelsize=ticksize)
            ax.grid(alpha=0.2)
    
            if not normalize_each:
                ymins.append(float(np.nanmin(y)))
                ymaxs.append(float(np.nanmax(y)))
    
        # ==========================
        # Sum panel
        # ==========================
        if show_sum and (nplots - 1 < len(axes)):
            ax = axes[nplots - 1]
    
            if len(good_set) > 0:
                raw_sum = data[list(good_set)].sum(axis=1).to_numpy(float)
            else:
                raw_sum = np.zeros_like(energy)
    
            ysum = safe_index(raw_sum)
    
            if normalize_each:
                m = np.nanmax(np.abs(ysum))
                if m and np.isfinite(m):
                    ysum = ysum / m
    
            sum_line, = ax.plot(energy, ysum, color="tab:red", lw=1.4, label="sum(good)")
            if e0 is not None and np.isfinite(e0):
                ax.axvline(e0, color="k", ls="--", lw=0.7)
    
            ax.set_title("sum(good)", fontsize=titlesize)
            ax.tick_params(axis="both", labelsize=ticksize)
            ax.grid(alpha=0.2)
    
            from matplotlib.lines import Line2D
            ax.legend(
                handles=[
                    Line2D([0], [0], color=good_color, lw=1.4, label="good"),
                    Line2D([0], [0], color=bad_color, lw=1.4, alpha=alpha_rejected, label="bad"),
                    sum_line,
                ],
                fontsize=ticksize, loc="best", frameon=True
            )
    
            if not normalize_each:
                ymins.append(float(np.nanmin(ysum)))
                ymaxs.append(float(np.nanmax(ysum)))
    
        # turn off leftovers
        for j in range(nplots, len(axes)):
            axes[j].axis("off")
    
        # label axes
        for r in range(nrows):
            ax_left = axes[r * ncols]
            if ax_left.has_data():
                ax_left.set_ylabel("Normalized" if normalize_each else "a.u.", fontsize=titlesize)
    
        for c in range(ncols):
            idx = (nrows - 1) * ncols + c
            if idx < len(axes) and axes[idx].has_data():
                axes[idx].set_xlabel("Energy (eV)", fontsize=titlesize)
    
        # match y ranges
        if sharey and (not normalize_each) and ymins and ymaxs:
            ymin, ymax = min(ymins), max(ymaxs)
            for k in range(min(nplots, len(axes))):
                if axes[k].has_data():
                    axes[k].set_ylim(ymin, ymax)
    
        fig.suptitle(f"Fluorescence channels — {label}", y=0.995, fontsize=max(10, titlesize + 2))
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        plt.show()
        plt.close(fig)

    def plot_mu(
        self,
        *labels,  # 0 or more label strings (SUM or member names)
        show_flat: bool = True,
        show_pre_edge: bool = True,
        show_post_edge: bool = True,
        show_bkg: bool = True,
        show_bkgsub: bool = False,
        show_norm: bool = True,
        show_raw: bool = True,
        show_trans: bool = False,                 # overlay trans on same primary if available
        offset_step: Optional[float] = 0.1,
        title: Optional[str] = None,
        mode: Optional[str] = None,
        overlay_both_in_other_modes: bool = False,  # unused for groups per your new rule
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        # group helpers
        show_members: bool = False,
        members_only: bool = False,
        # transmission-only mode
        show_legend: bool =True,
        trans_only: bool = False,
    ):
        """
        Plot μ(E) for aligned datasets.
    
        Behavior:
    
        • GROUPS (multi-dataset):
            - Plot only ONE among raw/flat/norm (priority: raw -> flat -> norm) based on flags & availability.
            - Diagnostics (pre, post, bkg, bkgsub) OFF.
            - If show_members=True, members follow same one-primary rule.
            - If show_trans=True, overlay the same primary in transmission if available.
    
        • SINGLE (PFY):
            - Respect flags: raw, pre, post, bkg, bkgsub, norm, flat can all be shown.
            - SUM background uses the FIRST member’s stored `bkg` so single/group backgrounds match.
            - SUM raw is always plot-able (prefer SUM.mu_raw; else first member mu_raw; else mu_trans_raw fallback).
            - If show_trans=True, overlay trans for raw/flat/norm when available.
    
        • trans_only=True:
            - SINGLE: same logic as PFY single (may plot multiple primaries & diagnostics if present).
            - GROUP: same logic as group PFY (one primary only: raw -> flat -> norm).
    
        • Legend:
            - Ordered so top entry corresponds to the highest plotted curve (largest offset).
            - E₀ marker appended to bottom.
        """
        import numpy as np
        import matplotlib.pyplot as plt
    
        # -------- guards --------
        if not hasattr(self, "_aligned_data"):
            raise RuntimeError("Run align_multiple first; _aligned_data is missing.")
        data = self._aligned_data
        if not isinstance(data, dict) or len(data) == 0:
            raise RuntimeError("No aligned data available for plotting.")
    
        # ---------- local helper: match y length to energy ----------
        def _y_match_length(rec: dict, y):
            """
            Return y with the same length as rec['energy'].
            - If 'edge_trim' exists (lo, hi), try y[lo:hi].
            - Otherwise truncate or right-pad with NaN to match length.
            """
            if y is None:
                return None
            E = rec.get("energy")
            if E is None:
                return y
            n = len(E)
            try:
                if hasattr(y, "__len__") and len(y) == n:
                    return y
            except Exception:
                pass
            tr = rec.get("edge_trim", None)
            if isinstance(tr, dict):
                try:
                    lo = int(tr.get("lo", 0))
                    hi = int(tr.get("hi", len(y)))
                    if 0 <= lo < hi and hasattr(y, "__len__") and len(y) >= hi:
                        y2 = y[lo:hi]
                        if len(y2) == n:
                            return y2
                except Exception:
                    pass
            y = np.asarray(y)
            if len(y) >= n:
                return y[:n]
            else:
                return np.pad(y, (0, n - len(y)), constant_values=np.nan)
    
        # ---------- detect label filter in first positional argument ----------
        # New: parse positional labels from *labels
        labels = list(labels) if labels else []
        labels_filter = None
        if labels:
            # If they passed a single list/tuple/set as first positional, expand it
            if len(labels) == 1 and isinstance(labels[0], (list, tuple, set)):
                labels_filter = [str(x) for x in labels[0]]
            else:
                labels_filter = [str(x) for x in labels]
    
        # ---------- maps ----------
        keyset = set(data.keys())
        orig_to_safe = getattr(self, "_label_map", {}) or {}
        disp_map = getattr(self, "_label_display_map", {}) or {}
        disp_to_safe = {v: k for k, v in disp_map.items()}
        sum_map = getattr(self, "_sum_map", {}) or {}
        sum_labels = set(sum_map.keys())
    
        def _resolve_label(lbl: str) -> Optional[str]:
            if lbl in keyset:
                return lbl
            if lbl in orig_to_safe:
                cand = orig_to_safe[lbl]
                return cand if cand in keyset else None
            if lbl in disp_to_safe:
                return disp_to_safe[lbl]
            try:
                cand = self._safe_label(lbl)
                if cand in keyset:
                    return cand
            except Exception:
                pass
            return None
    
        # ---------- explicit selection ----------
        if labels_filter:
            resolved = [_resolve_label(x) for x in labels_filter]
            missing = [labels_filter[i] for i, r in enumerate(resolved) if r is None]
            labels_selected = [r for r in resolved if r is not None]
            if not labels_selected:
                suggestions = sorted(list(keyset))[:10]
                raise KeyError(
                    f"None of the requested labels were found: {missing}\n"
                    f"Known labels (first 10): {suggestions}"
                )
            # Expand SUM to members if requested
            expanded = []
            for lab in labels_selected:
                if show_members and (lab in sum_labels):
                    members = sum_map.get(lab, [])
                    if members_only:
                        expanded.extend(members)
                    else:
                        expanded.append(lab)
                        expanded.extend(members)
                else:
                    expanded.append(lab)
            seen = set()
            labels_selected = [x for x in expanded if not (x in seen or seen.add(x))]
        else:
            labels_selected = None
    
        # ---------- E0 ----------
        e0_ref = getattr(self, "_reference_e0", None)
        if e0_ref is None:
            ref = getattr(self, "_reference", None)
            e0_ref = float(self.e0[ref]) if isinstance(ref, str) else float(ref)
        lw = 1.8
    
        # ---------- originals ----------
        originals = [
            lab for lab, d in data.items()
            if lab not in sum_labels and d.get("filepath") is not None
        ]
    
        # ---------- offset step ----------
        if offset_step is None:
            spans = []
    
            def _accum(dct):
                # SAFE: do not use "or" with arrays
                raw_like = dct.get("mu_raw")
                if raw_like is None:
                    raw_like = dct.get("mu_trans_raw")
                for arr in (dct.get("mu_norm"), dct.get("mu_flat"), dct.get("mu_bkgsub"), raw_like):
                    if arr is not None:
                        try:
                            spans.append(float(np.nanmax(arr) - np.nanmin(arr)))
                        except Exception:
                            pass
    
            if labels_selected:
                for lab in labels_selected:
                    _accum(data[lab])
            else:
                for d in data.values():
                    _accum(d)
            offset_step = 0.08 * max(spans) if spans else 0.05
    
        # ---------- helpers ----------
        def _has_fluo(rec: dict) -> bool:
            return bool(rec.get("InB_channels_all") or rec.get("OutB_channels_all"))
    
        def _resolved_bkg(label: str, rec: dict):
            """
            For SUM label, use FIRST member's stored bkg so single/group background match.
            Otherwise return rec['bkg'].
            """
            if label in sum_labels:
                members = sum_map.get(label, []) or []
                if len(members) >= 1:
                    mb = data.get(members[0], {}).get("bkg")
                    return _y_match_length(rec, mb)
            return _y_match_length(rec, rec.get("bkg"))
    
        def _resolved_raw(label: str, rec: dict):
            """
            Ensure SUM raw is plottable:
             1) Prefer rec['mu_raw'] (SUM.mu_raw stored).
             2) Else first member's mu_raw.
             3) Else mu_trans_raw (rec or member).
            """
            y = rec.get("mu_raw")
            if y is not None:
                return _y_match_length(rec, y)
            if label in sum_labels:
                members = sum_map.get(label, []) or []
                for m in members:
                    y = data.get(m, {}).get("mu_raw")
                    if y is not None:
                        print("can't find mu")
            y = rec.get("mu_trans_raw")
            if y is not None:
                return _y_match_length(rec, y)
            if label in sum_labels:
                members = sum_map.get(label, []) or []
                for m in members:
                    y = data.get(m, {}).get("mu_trans_raw")
                    if y is not None:
                        return _y_match_length(rec, y)
            return None
    
        # Group one-primary chooser (PFY)
        def _choose_group_primary(rec: dict):
            if show_flat:
                y = rec.get("mu_flat")
                if y is not None:
                    return (_y_match_length(rec, y), "flat", " (flat)")
            if show_norm:
                y = rec.get("mu_norm")
                if y is not None:
                    return (_y_match_length(rec, y), "norm", " (norm)")
            if show_raw:
                y = rec.get("mu_raw")
                if y is not None:
                    return (_y_match_length(rec, y), "raw", " (raw)")
            return (None, None, "")
    
        # Group one-primary chooser (TRANS)
        def _choose_group_primary_trans(rec: dict):
            if show_flat:
                y = rec.get("mu_trans_flat")
                if y is not None:
                    return (_y_match_length(rec, y), "flat", " (trans·flat)")
            if show_norm:
                y = rec.get("mu_trans_norm")
                if y is not None:
                    return (_y_match_length(rec, y), "norm", " (trans·norm)")
            if show_raw:
                y = rec.get("mu_trans_raw")
                if y is not None:
                    return (_y_match_length(rec, y), "raw", " (trans·raw)")
            return (None, None, "")
    
        # Legend manager: ensure top legend entry is highest offset
        legend_handles, legend_labels, legend_orders = [], [], []
    
        def _add_legend(line, label, order_index):
            if (line is None) or (label is None):
                return
            legend_handles.append(line)
            legend_labels.append(label)
            legend_orders.append(order_index)
    
        # ---------------- TRANS-ONLY INTERPRETATION FLAGS ----------------
        if trans_only:
            trans_show_raw  = bool(show_raw)
            trans_show_norm = bool(show_norm)
            trans_show_flat = bool(show_flat)
            trans_show_pre  = bool(show_pre_edge)
            trans_show_post = bool(show_post_edge)
            trans_show_bkg  = bool(show_bkg)
            trans_show_bsub = bool(show_bkgsub)
    
        # ---------- single vs multi ----------
        single_mode = (labels_selected is not None and len(labels_selected) == 1) or \
                      (labels_selected is None and len(originals) <= 1)
    
        # -------------------------------------------------
        #                    SINGLE
        # -------------------------------------------------
        if single_mode:
            labels = labels_selected if labels_selected is not None else ([originals[0]] if originals else [list(data.keys())[0]])
            comp_color = {
                "raw": "tab:green",
                "pre_edge": "tab:purple",
                "post_edge": "tab:gray",
                "bkg": "tab:orange",
                "bkgsub": "tab:pink",
                "norm": "tab:blue",
                "flat": "tab:red",
            }
            plt.figure(figsize=(8, 5))
            used = set()
    
            for i, label in enumerate(labels):
                if label not in data:
                    continue
                d = data[label]
                E = d["energy"]
                off = i * offset_step
                lab_disp = disp_map.get(label, label)
    
                # --- SINGLE · TRANS-ONLY ---
                if trans_only:
                    # RAW
                    if trans_show_raw and (d.get("mu_trans_raw") is not None):
                        y = _y_match_length(d, d["mu_trans_raw"])
                        ln, = plt.plot(E, y + off, lw=lw, color=comp_color["raw"], label=None if "trans raw" in used else "trans raw")
                        if "trans raw" not in used: _add_legend(ln, "trans raw", i); used.add("trans raw")
                    # PRE
                    if trans_show_pre:
                        y = d.get("pre_edge_trans")
                        if y is None:
                            y = d.get("pre_edge")
                        if y is not None:
                            y = _y_match_length(d, y)
                            ln, = plt.plot(E, y + off, lw=lw, ls="--", color=comp_color["pre_edge"], label=None if "pre-edge" in used else "pre-edge")
                            if "pre-edge" not in used: _add_legend(ln, "pre-edge", i); used.add("pre-edge")
                    # POST
                    if trans_show_post:
                        y = d.get("post_edge_trans")
                        if y is None:
                            y = d.get("post_edge")
                        if y is not None:
                            y = _y_match_length(d, y)
                            ln, = plt.plot(E, y + off, lw=lw, ls="--", color=comp_color["post_edge"], label=None if "post-edge" in used else "post-edge")
                            if "post-edge" not in used: _add_legend(ln, "post-edge", i); used.add("post-edge")
                    # BKG
                    if trans_show_bkg:
                        y = d.get("bkg_trans")
                        if y is None:
                            y = _resolved_bkg(label, d)
                        if y is not None:
                            y = _y_match_length(d, y)
                            ln, = plt.plot(E, y + off, lw=lw, color=comp_color["bkg"], label=None if "background" in used else "background")
                            if "background" not in used: _add_legend(ln, "background", i); used.add("background")
                    # BKGSUB
                    if trans_show_bsub:
                        y = d.get("mu_trans_bkgsub")
                        if y is None:
                            y = d.get("mu_bkgsub")
                        if y is not None:
                            y = _y_match_length(d, y)
                            ln, = plt.plot(E, y + off, lw=lw, ls="--", color=comp_color["bkgsub"], label=None if "μ - bkg" in used else "μ - bkg")
                            if "μ - bkg" not in used: _add_legend(ln, "μ - bkg", i); used.add("μ - bkg")
                    # FLAT
                    if trans_show_flat and (d.get("mu_trans_flat") is not None):
                        y = _y_match_length(d, d["mu_trans_flat"])
                        ln, = plt.plot(E, y + off, lw=lw, color=comp_color["flat"], label=None if "trans flat" in used else "trans flat")
                        if "trans flat" not in used: _add_legend(ln, "trans flat", i); used.add("trans flat")
                    # NORM
                    if trans_show_norm and (d.get("mu_trans_norm") is not None):
                        y = _y_match_length(d, d["mu_trans_norm"])
                        ln, = plt.plot(E, y + off, lw=lw, color=comp_color["norm"], label=None if "trans norm" in used else "trans norm")
                        if "trans norm" not in used: _add_legend(ln, "trans norm", i); used.add("trans norm")
    
                # --- SINGLE · PFY ---
                else:
                    # RAW (ensure SUM raw plots)
                    if show_raw:
                        y_raw = _resolved_raw(label, d)
                        if y_raw is not None:
                            ln, = plt.plot(E, y_raw + off, color=comp_color["raw"], lw=lw,
                                           label=None if "raw" in used else "raw")
                            if "raw" not in used: _add_legend(ln, "raw", i); used.add("raw")
                            # overlay trans raw if requested
                            if show_trans and (d.get("mu_trans_raw") is not None):
                                ytr = _y_match_length(d, d["mu_trans_raw"])
                                ln, = plt.plot(E, ytr + off, lw=lw, color=comp_color["raw"], ls="--",
                                               label=None if "trans raw" in used else "trans raw")
                                if "trans raw" not in used: _add_legend(ln, "trans raw", i); used.add("trans raw")
    
                    # Diagnostics
                    if show_pre_edge and (d.get("pre_edge") is not None):
                        y = _y_match_length(d, d["pre_edge"])
                        ln, = plt.plot(E, y + off, color=comp_color["pre_edge"], lw=lw, ls="--",
                                       label=None if "pre-edge" in used else "pre-edge")
                        if "pre-edge" not in used: _add_legend(ln, "pre-edge", i); used.add("pre-edge")
    
                    if show_post_edge and (d.get("post_edge") is not None):
                        y = _y_match_length(d, d["post_edge"])
                        ln, = plt.plot(E, y + off, color=comp_color["post_edge"], lw=lw, ls="--",
                                       label=None if "post-edge" in used else "post-edge")
                        if "post-edge" not in used: _add_legend(ln, "post-edge", i); used.add("post-edge")
    
                    if show_bkg:
                        yb = _resolved_bkg(label, d)
                        if yb is not None:
                            ln, = plt.plot(E, yb + off, color=comp_color["bkg"], lw=lw,
                                           label=None if "background" in used else "background")
                            if "background" not in used: _add_legend(ln, "background", i); used.add("background")
    
                    if show_bkgsub and (d.get("mu_bkgsub") is not None):
                        y = _y_match_length(d, d["mu_bkgsub"])
                        ln, = plt.plot(E, y + off, color=comp_color["bkgsub"], lw=lw, ls="--",
                                       label=None if "μ - bkg" in used else "μ - bkg")
                        if "μ - bkg" not in used: _add_legend(ln, "μ - bkg", i); used.add("μ - bkg")
    
                    # FLAT & NORM (+trans overlays)
                    if show_flat and (d.get("mu_flat") is not None):
                        y = _y_match_length(d, d["mu_flat"])
                        ln, = plt.plot(E, y + off, color=comp_color["flat"], lw=lw,
                                       label=None if "flat" in used else "flat")
                        if "flat" not in used: _add_legend(ln, "flat", i); used.add("flat")
                        if show_trans and (d.get("mu_trans_flat") is not None):
                            ytr = _y_match_length(d, d["mu_trans_flat"])
                            ln, = plt.plot(E, ytr + off, lw=lw, color=comp_color["flat"], ls="--",
                                           label=None if "trans flat" in used else "trans flat")
                            if "trans flat" not in used: _add_legend(ln, "trans flat", i); used.add("trans flat")
    
                    if show_norm and (d.get("mu_norm") is not None):
                        y = _y_match_length(d, d["mu_norm"])
                        ln, = plt.plot(E, y + off, color=comp_color["norm"], lw=lw,
                                       label=None if "norm" in used else "norm")
                        if "norm" not in used: _add_legend(ln, "norm", i); used.add("norm")
                        if show_trans and (d.get("mu_trans_norm") is not None):
                            ytr = _y_match_length(d, d["mu_trans_norm"])
                            ln, = plt.plot(E, ytr + off, lw=lw, color=comp_color["norm"], ls="--",
                                           label=None if "trans norm" in used else "trans norm")
                            if "trans norm" not in used: _add_legend(ln, "trans norm", i); used.add("trans norm")
    
            # E0 & legend (highest first; E0 at bottom)
            e0_line = plt.axvline(e0_ref, color="k", ls="--", lw=0.8)
            uniq = {}
            for h, lab, ordv in zip(legend_handles, legend_labels, legend_orders):
                uniq[lab] = (h, ordv)
            ordered = sorted(uniq.items(), key=lambda kv: kv[1][1], reverse=True)
            handles_final = [kv[1][0] for kv in ordered]
            labels_final  = [kv[0]     for kv in ordered]
            handles_final.append(e0_line)
            labels_final.append(rf"E₀={e0_ref}")
            plt.xlabel("Energy (eV)")
            plt.ylabel("μ(E)")
            plt.title(title or (f"{'μ(E) (transmission only)' if trans_only else 'μ(E) diagnostics'} — {disp_map.get(labels[0], labels[0])}"))
            plt.grid(alpha=0.0)
            if show_legend == True:
                if handles_final:
                    plt.legend(handles_final, labels_final, fontsize=9, ncol=2)
            if xlim is not None: plt.xlim(*xlim)
            if ylim is not None: plt.ylim(*ylim)
            plt.tight_layout(); plt.show()
            return
    
        # -------------------------------------------------
        #                    MULTI / GROUPS
        # -------------------------------------------------
        all_original = originals
        member_set = {m for lst in sum_map.values() for m in lst}
        unsummed = [l for l in all_original if l not in member_set]
    
        overlay_order = []
        for s in sum_labels:
            overlay_order.append(s)
            overlay_order.extend(sum_map[s])
    
        mode_in = mode or getattr(self, "_plot_mode", "sums_plus_unsummed")
    
        if labels_selected is not None:
            labels_to_plot = [lab for lab in labels_selected if lab in data]
        else:
            if mode_in == "sums_plus_unsummed":
                labels_to_plot = list(sum_labels) + unsummed
            elif mode_in == "sums_plus_individuals":
                labels_to_plot = overlay_order
            elif mode_in == "all":
                labels_to_plot = all_original + list(sum_labels)
            elif mode_in == "sums_only":
                labels_to_plot = list(sum_labels)
            else:
                labels_to_plot = list(sum_labels) + unsummed
    
        seen = set()
        labels_to_plot = [x for x in labels_to_plot if not (x in seen or seen.add(x))]
        if len(sum_labels) == 0 and len(labels_to_plot) == 0:
            labels_to_plot = all_original
    
        plt.figure(figsize=(9, 6))
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    
        for i, label in enumerate(labels_to_plot):
            if label not in data:
                continue
            d = data[label]
            E = d["energy"]
            color = colors[i % len(colors)]
            off = i * offset_step
            lab_disp = disp_map.get(label, label)
    
            # GROUP · TRANS-ONLY: one primary only
            if trans_only:
                yT, tagT, suffT = _choose_group_primary_trans(d)
                if yT is not None:
                    ln, = plt.plot(E, yT + off, lw=lw, color=color, label=f"{lab_disp}{suffT}")
                    _add_legend(ln, f"{lab_disp}{suffT}", i)
                continue
    
            # GROUP · PFY: one primary only
            yP, tagP, suffP = _choose_group_primary(d)
            if yP is not None:
                ln, = plt.plot(E, yP + off, lw=lw, color=color, label=f"{lab_disp}{suffP}")
                _add_legend(ln, f"{lab_disp}{suffP}", i)
    
                # If show_trans, overlay same primary in trans when available
                if show_trans:
                    if tagP == "flat" and (d.get("mu_trans_flat") is not None):
                        ytr = _y_match_length(d, d["mu_trans_flat"])
                        ln, = plt.plot(E, ytr + off, lw=lw, color=color, ls="--", label=f"{lab_disp} (trans·flat)")
                        _add_legend(ln, f"{lab_disp} (trans·raw)", i)
                    elif tagP == "norm" and (d.get("mu_trans_norm") is not None):
                        ytr = _y_match_length(d, d["mu_trans_norm"])
                        ln, = plt.plot(E, ytr + off, lw=lw, color=color, ls="--", label=f"{lab_disp} (trans·norm)")
                        _add_legend(ln, f"{lab_disp} (trans·norm)", i)
                    elif tagP == "raw" and (d.get("mu_trans_raw") is not None):
                        ytr = _y_match_length(d, d["mu_trans_raw"])
                        ln, = plt.plot(E, ytr + off, lw=lw, color=color, ls="--", label=f"{lab_disp} (trans·raw)")
                        _add_legend(ln, f"{lab_disp} (trans·flat)", i)
                        
        # Footer + legend (highest first; E0 at bottom)
        e0_line = plt.axvline(e0_ref, color="k", ls="--", lw=0.8)
        uniq = {}
        for h, lab, ordv in zip(legend_handles, legend_labels, legend_orders):
            uniq[lab] = (h, ordv)
        ordered = sorted(uniq.items(), key=lambda kv: kv[1][1], reverse=True)
        handles_final = [kv[1][0] for kv in ordered]
        labels_final  = [kv[0]     for kv in ordered]
        handles_final.append(e0_line)
        labels_final.append(rf"E₀={e0_ref}")
        plt.xlabel("Energy (eV)")
        plt.ylabel("μ(E)")
        title_here = title or (
            f"Aligned μ(E): transmission only — {mode_in.replace('_', ' ')}" if trans_only else
            (f"Aligned μ(E): {mode_in.replace('_', ' ')}" if labels_selected is None else "Aligned μ(E): selected")
        )
        plt.title(title_here)
        plt.grid(alpha=0.0)
        # if handles_final:
        #     plt.legend(handles_final, labels_final, fontsize=9)
        if show_legend == True:
            if handles_final:
                plt.legend(handles_final, labels_final, fontsize=9, ncol=2)
        if xlim is not None: plt.xlim(*xlim)
        if ylim is not None: plt.ylim(*ylim)
        plt.tight_layout(); plt.show()

        
    def plot_chi_k(
        self,
        labels: Optional[Iterable[str]] = None,
        kweight: Optional[int] = None,
        show_raw: bool = False,
        show_smooth: bool = True,
        smooth_window: int = 11,
        smooth_order: int = 3,
        stack: str = "overlay",              # "overlay" | "offset" | "subplots"
        offset_step: Optional[float] = None, # vertical spacing for "offset" mode
        offset_baseline: float = 0.0,        # baseline offset to start stacking
        normalize: bool = False,             # normalize each trace amplitude before stacking
        sharey: bool = False,                # for subplots mode: share y axis
        outdir: Optional[str] = None,
        filename: Optional[str] = None,
        dpi: int = 300,
        # group helpers
        show_members: bool = False,
        members_only: bool = False,
        # channel selection
        show_legend: bool = True,
        trans_only: bool = False,            # use attached transmission (grp._trans) if available
    ):
        """
        Plot k-space χ(k) with options to overlay, offset-stack, or subplots.
    
        Notes:
          - If trans_only=True, uses the attached transmission channel (grp._trans) created
            by align_multiple(). If not present for a label, that label is skipped with a warning.
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        try:
            from scipy.signal import savgol_filter
        except Exception:
            raise ImportError("scipy.signal.savgol_filter is required for show_smooth=True.")
    
        # effective k-weight
        kweight_eff = int(getattr(self, "_default_kweight", 2) if kweight is None else kweight)
    
        # normalize labels
        if labels is None:
            labels = list(self.groups.keys())
        elif isinstance(labels, str):
            labels = [labels]
        labels = list(labels)
        if not labels:
            raise ValueError("No labels to plot.")
    
        # expand SUM -> [SUM + members] or [members only]
        sum_map = getattr(self, "_sum_map", {}) or {}
        def _expand_labels(labs):
            out, seen = [], set()
            for lb in labs:
                if show_members and (lb in sum_map):
                    mems = list(sum_map.get(lb, []))
                    to_add = (mems if members_only else [lb] + mems)
                else:
                    to_add = [lb]
                for x in to_add:
                    if x not in seen:
                        out.append(x); seen.add(x)
            return out
        labels = _expand_labels(labels)
    
        # validate smoothing hyperparams
        if show_smooth:
            if smooth_window % 2 == 0:
                smooth_window += 1
            if smooth_window <= smooth_order:
                raise ValueError("smooth_window must be > smooth_order")
    
        # collect traces
        traces = []  # [{"label","k","y_raw","y_smooth"}]
        for lb in labels:
            g = self.groups.get(lb)
            if g is None:
                continue
            # pick source: transmission or primary
            src = getattr(g, "_trans", None) if trans_only else None
            if trans_only and (src is None):
                print(f"[WARN] No transmission channel attached for '{lb}'; skipping.")
                continue
            src = src if src is not None else g
    
            k   = getattr(src, "k", None)
            chi = getattr(src, "chi", None)
            if k is None or chi is None:
                continue
    
            k = np.asarray(k); chi = np.asarray(chi)
            mask = np.isfinite(k) & np.isfinite(chi)
            k, chi = k[mask], chi[mask]
            if k.size == 0:
                continue
    
            y_raw = (k**kweight_eff) * chi
    
            if show_smooth and chi.size >= smooth_window:
                chi_s = savgol_filter(chi, smooth_window, smooth_order)
                y_smooth = (k**kweight_eff) * chi_s
            elif show_smooth:
                y_smooth = None
            else:
                y_smooth = None
    
            if normalize:
                denom = None
                if y_smooth is not None and np.max(np.abs(y_smooth)) > 0:
                    denom = np.max(np.abs(y_smooth))
                elif np.max(np.abs(y_raw)) > 0:
                    denom = np.max(np.abs(y_raw))
                if denom and denom != 0:
                    y_raw = y_raw / denom
                    if y_smooth is not None:
                        y_smooth = y_smooth / denom
    
            traces.append({"label": lb, "k": k, "y_raw": y_raw, "y_smooth": y_smooth})
    
        if not traces:
            raise ValueError("No valid k/chi data to plot after filtering.")
    
        # default offset_step
        if stack == "offset" and offset_step is None:
            ptp_vals = []
            for tr in traces:
                vals = [np.ptp(tr["y_raw"])]
                if tr["y_smooth"] is not None:
                    vals.append(np.ptp(tr["y_smooth"]))
                vals = [v for v in vals if np.isfinite(v)]
                if vals:
                    ptp_vals.append(np.max(vals))
            typical = np.median(ptp_vals) if ptp_vals else 1.0
            offset_step = 0.6 * typical
    
        # plot
        if stack == "subplots":
            n = len(traces)
            fig, axes = plt.subplots(n, 1, figsize=(6, 2.6*n), sharex=True, sharey=sharey)
            if n == 1:
                axes = [axes]
            for i, tr in enumerate(traces):
                ax = axes[i]
                k, y_raw, y_smooth, lb = tr["k"], tr["y_raw"], tr["y_smooth"], tr["label"]
                if show_raw:
                    ax.plot(k, y_raw, label=rf"{lb} raw $k^{kweight_eff}\chi(k)$", lw=1.5)
                if show_smooth:
                    if y_smooth is not None:
                        ax.plot(k, y_smooth, label=rf"{lb} smoothed $k^{kweight_eff}\chi(k)$", lw=1.5)
                    else:
                        ax.plot(k, y_raw, label=rf"{lb} (no smooth; n<{smooth_window})", lw=1.0, ls="--", alpha=0.7)
                if show_legend == True:
                    ax.legend(fontsize=9, ncol=1 if len(traces) < 8 else 2)
                ax.grid(alpha=0.0)
            axes[-1].set_xlabel(r"$k$ (Å$^{-1}$)")
            plt.tight_layout()
        else:
            fig, ax = plt.subplots(figsize=(9, 6))
            for i, tr in enumerate(traces):
                k, y_raw, y_smooth, lb = tr["k"], tr["y_raw"], tr["y_smooth"], tr["label"]
                y_off = offset_baseline + i * offset_step if stack == "offset" else 0.0
                if show_raw:
                    ax.plot(k, y_raw + y_off, label=rf"{lb} raw $k^{kweight_eff}\chi(k)$", lw=1.5)
                if show_smooth:
                    if y_smooth is not None:
                        ax.plot(k, y_smooth + y_off, label=rf"{lb} smoothed $k^{kweight_eff}\chi(k)$", lw=1.5)
                    else:
                        ax.plot(k, y_raw + y_off, label=rf"{lb} (no smooth; n<{smooth_window})", lw=1.0, ls="--", alpha=0.7)
            ax.set_xlabel(r"$k$ (Å$^{-1}$)")
            ax.set_ylabel(rf"$k^{kweight_eff}\chi(k)$" + ("  (stacked)" if stack == "offset" else ""))
            if show_legend == True:
                ax.legend(fontsize=9, ncol=1 if len(traces) < 8 else 2)
            ax.grid(alpha=0.0)
            plt.tight_layout()
    
        # output
        def _sanitize(s: str) -> str:
            return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)
    
        if outdir is None and filename is None:
            plt.show()
            return fig, None
    
        if filename:
            out_path = os.path.abspath(filename if os.path.isabs(filename)
                                       else os.path.join(outdir or ".", filename))
        else:
            lab_str = "ALL" if len(labels) > 3 else "_".join(_sanitize(lb) for lb in labels)
            base = ("K_STACK_" + stack.upper() + "_" if stack != "overlay" else "K_") + lab_str
            base += "_TRANS" if trans_only else ""
            out_path = os.path.abspath(os.path.join(outdir or ".", f"{base}.png"))
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.show()
        return fig, out_path
        

    def plot_I0(
        self,
        labels: Optional[Iterable[str]] = None,
        stack: str = "overlay",              # "overlay" | "offset" | "subplots"
        offset_step: Optional[float] = None, # vertical spacing for "offset" mode
        offset_baseline: float = 0.0,        # baseline offset to start stacking
        normalize: bool = False,             # per-trace normalization (median-based)
        sharey: bool = False,                # for subplots mode: share y axis
        outdir: Optional[str] = None,
        filename: Optional[str] = None,
        dpi: int = 300,
        # group helpers
        show_members: bool = False,
        members_only: bool = False,
        # visuals
        show_legend: bool = True,
        shade_group_trim: bool = True,       # shade kept range for group sums
    ):
        """
        Plot I0 vs Energy for single scans (I0_aligned) and/or SUM groups (I0_group).
        Matches the style and control flow of plot_chi_r (overlay/offset/subplots).
    
        Parameters
        ----------
        labels : list[str] or str or None
            Labels to plot. If None, will plot all available labels in _aligned_data.
            Accepts original or safe labels; will remap via _label_map if present.
        stack : {"overlay", "offset", "subplots"}
            - "overlay": all traces on one axis (legend enabled by default)
            - "offset":  vertical stacking on one axis with per-trace offset
            - "subplots": one axis per label (sharey optional)
        offset_step : float or None
            Vertical spacing for "offset" stacking. If None, auto-derived from data.
        offset_baseline : float
            Starting baseline for the first trace in "offset" stacking.
        normalize : bool
            If True, each trace is divided by its median (robust across scans).
        sharey : bool
            For "subplots", whether to share the y-axis across all subplots.
        outdir, filename : str
            Output controls: saves if filename is provided, else auto name if outdir
            is given. If neither is given, will just show() and return (fig, None).
        dpi : int
            Save/display resolution.
        show_members : bool
            If a label is a SUM group and show_members=True, expand to include
            group members in the plotting list (in addition to group unless members_only=True).
        members_only : bool
            If True with show_members=True, includes only the members (no group trace).
        show_legend : bool
            Show legend for overlay/offset modes, and in subplots if desired.
        shade_group_trim : bool
            If True, for groups with 'edge_trim' (lo,hi) in _aligned_data, shade the retained region.
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt
    
        # ---- normalize/prepare labels ----
        aligned = getattr(self, "_aligned_data", {}) or {}
        label_map = getattr(self, "_label_map", {}) or {}
        sum_map   = getattr(self, "_sum_map", {}) or {}
    
        if labels is None:
            labels = list(aligned.keys())
        elif isinstance(labels, str):
            labels = [labels]
        labels = list(labels)
        if not labels:
            raise ValueError("No labels to plot.")
    
        # map original -> safe if needed
        labels = [label_map.get(lb, lb) for lb in labels]
    
        # expand SUM groups if requested
        def _expand_labels(labs):
            out, seen = [], set()
            for lb in labs:
                if show_members and (lb in sum_map):
                    mems = list(sum_map.get(lb, []))
                    to_add = (mems if members_only else [lb] + mems)
                else:
                    to_add = [lb]
                for x in to_add:
                    if x not in seen:
                        out.append(x); seen.add(x)
            return out
    
        labels = _expand_labels(labels)
    
        # ---- gather traces ----
        traces = []  # each: {"label","E","I0","is_group","trim_lo","trim_hi"}
        for lb in labels:
            rec = aligned.get(lb)
            if rec is None:
                # Not aligned/known: skip silently? No—mirror your chi_r behavior: skip if missing.
                # But log a warning to help debugging.
                print(f"[WARN] Label '{lb}' not found in _aligned_data; skipping.")
                continue
    
            E = rec.get("energy", None)
            # prefer group I0 if present; else single-scan I0_aligned
            if ("I0_group" in rec) and (rec["I0_group"] is not None):
                I0 = rec["I0_group"]
                is_group = True
                lo = rec.get("edge_trim", {}).get("lo", None)
                hi = rec.get("edge_trim", {}).get("hi", None)
            else:
                I0 = rec.get("I0_aligned", None)
                is_group = False
                lo, hi = None, None
    
            if E is None or I0 is None:
                # mirror chi_r: skip missing
                print(f"[WARN] Missing energy or I0 for '{lb}'; skipping.")
                continue
    
            E = np.asarray(E)
            I0 = np.asarray(I0)
            mask = np.isfinite(E) & np.isfinite(I0)
            if not np.any(mask):
                print(f"[WARN] No finite I0 points for '{lb}'; skipping.")
                continue
            E = E[mask]; I0 = I0[mask]
    
            if normalize:
                med = np.nanmedian(I0)
                if not (np.isfinite(med) and med != 0.0):
                    med = 1.0
                I0 = I0 / med
    
            traces.append({
                "label": lb,
                "E": E,
                "I0": I0,
                "is_group": is_group,
                "trim_lo": lo,
                "trim_hi": hi,
            })
    
        if not traces:
            raise ValueError("No valid I0 data to plot after filtering.")
    
        # ---- default offset step (like plot_chi_r logic) ----
        if stack == "offset" and offset_step is None:
            ptp_vals = []
            for tr in traces:
                ptp = np.nanmax(tr["I0"]) - np.nanmin(tr["I0"])
                if np.isfinite(ptp):
                    ptp_vals.append(ptp)
            typical = np.nanmedian(ptp_vals) if ptp_vals else 1.0
            offset_step = 0.6 * typical
    
        # ---- plotting styles ----
        line_style = dict(ls="-", lw=1.0)
    
        # ---- plot ----
        if stack == "subplots":
            n = len(traces)
            fig, axes = plt.subplots(n, 1,figsize= (6, 2.6*n), sharex=True, sharey=sharey, dpi=dpi)
            if n == 1:
                axes = [axes]
            for i, tr in enumerate(traces):
                ax = axes[i]
                E, I0, lb = tr["E"], tr["I0"], tr["label"]
                ax.plot(E, I0, label=lb, **line_style)
    
                # shade kept range for groups (edge-trim)
                if shade_group_trim and tr["is_group"] and (tr["trim_lo"] is not None) and (tr["trim_hi"] is not None):
                    lo, hi = int(tr["trim_lo"]), int(tr["trim_hi"])
                    if 0 <= lo < len(E) and 0 < hi <= len(E) and hi > lo:
                        ax.axvspan(E[lo], E[hi-1], color="tab:green", alpha=0.12, label="edge_trim")
    
                ax.set_ylabel("I0" + (" (norm)" if normalize else ""))
                if show_legend:
                    ax.legend(fontsize=9, ncol=1 if len(traces) < 8 else 2)
                ax.grid(alpha=0.0)
            axes[-1].set_xlabel("Energy (eV)")
            plt.tight_layout()
        else:
            fig, ax = plt.subplots(figsize=(6, 3), dpi=dpi)
            for i, tr in enumerate(traces):
                E, I0, lb = tr["E"], tr["I0"], tr["label"]
                y_off = offset_baseline + i * offset_step if stack == "offset" else 0.0
                ax.plot(E, I0 + y_off, label=lb, **line_style)
    
                if shade_group_trim and tr["is_group"] and (tr["trim_lo"] is not None) and (tr["trim_hi"] is not None):
                    lo, hi = int(tr["trim_lo"]), int(tr["trim_hi"])
                    if 0 <= lo < len(E) and 0 < hi <= len(E) and hi > lo:
                        ax.axvspan(E[lo], E[hi-1], color="tab:green", alpha=0.10)
    
            ax.set_xlabel("Energy (eV)")
            ax.set_ylabel("I0" + (" (stacked)" if stack == "offset" else "") + (" (norm)" if normalize else ""))
            if show_legend:
                ax.legend(fontsize=9, ncol=1 if len(traces) < 8 else 2)
            ax.grid(alpha=0.0)
            plt.tight_layout()
    
        # ---- output handling (same style as plot_chi_r) ----
        def _sanitize(s: str) -> str:
            return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)
    
        if outdir is None and filename is None:
            plt.show()
            return fig, None
    
        if filename:
            out_path = os.path.abspath(filename if os.path.isabs(filename)
                                       else os.path.join(outdir or ".", filename))
        else:
            labs = [t["label"] for t in traces]
            lab_str = "ALL" if len(labs) > 3 else "_".join(_sanitize(lb) for lb in labs)
            base = ("I0_STACK_" + stack.upper() + "_" if stack != "overlay" else "I0_") + lab_str
            out_path = os.path.abspath(os.path.join(outdir or ".", f"{base}.png"))
    
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.show()
        return fig, out_path
    
    def plot_chi_r(
        self,
        labels: Optional[Iterable[str]] = None,
        show_Mag: bool = True,
        show_Real: bool = False,
        show_Imag: bool = False,
        stack: str = "overlay",              # "overlay" | "offset" | "subplots"
        offset_step: Optional[float] = None, # vertical spacing for "offset" mode
        offset_baseline: float = 0.0,        # baseline offset to start stacking
        normalize: bool = False,             # normalize each label's amplitude across selected components
        sharey: bool = False,                # for subplots mode: share y axis
        outdir: Optional[str] = None,
        filename: Optional[str] = None,
        dpi: int = 300,
        # group helpers
        show_members: bool = False,
        members_only: bool = False,
        # channel selection
        trans_only: bool = False,
        show_legend: bool = True,
    ):
        """
        Plot R-space χ(R) components with options to overlay, offset-stack, or subplots.
    
        If trans_only=True, uses attached transmission channel (grp._trans) if present.
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt
    
        if not (show_Mag or show_Real or show_Imag):
            raise ValueError("At least one of show_Mag, show_Real, or show_Imag must be True.")
    
        # normalize labels
        if labels is None:
            labels = list(self.groups.keys())
        elif isinstance(labels, str):
            labels = [labels]
        labels = list(labels)
        if not labels:
            raise ValueError("No labels to plot.")
    
        # expand SUM
        sum_map = getattr(self, "_sum_map", {}) or {}
        def _expand_labels(labs):
            out, seen = [], set()
            for lb in labs:
                if show_members and (lb in sum_map):
                    mems = list(sum_map.get(lb, []))
                    to_add = (mems if members_only else [lb] + mems)
                else:
                    to_add = [lb]
                for x in to_add:
                    if x not in seen:
                        out.append(x); seen.add(x)
            return out
        labels = _expand_labels(labels)
    
        # gather traces
        traces = []  # {"label","r","y_mag","y_real","y_imag"}
        for lb in labels:
            g = self.groups.get(lb)
            if g is None:
                continue
            src = getattr(g, "_trans", None) if trans_only else None
            if trans_only and (src is None):
                print(f"[WARN] No transmission channel attached for '{lb}'; skipping.")
                continue
            src = src if src is not None else g
    
            r       = getattr(src, "r", None)
            chir    = getattr(src, "chir", None)
            chir_re = getattr(src, "chir_re", None)
            chir_im = getattr(src, "chir_im", None)
            if any(v is None for v in (r, chir, chir_re, chir_im)):
                continue
    
            r = np.asarray(r); chir = np.asarray(chir)
            chir_re = np.asarray(chir_re); chir_im = np.asarray(chir_im)
            mask = np.isfinite(r) & np.isfinite(chir) & np.isfinite(chir_re) & np.isfinite(chir_im)
            r, chir, chir_re, chir_im = r[mask], chir[mask], chir_re[mask], chir_im[mask]
            if r.size == 0:
                continue
    
            y_mag  = np.abs(chir) if show_Mag else None
            y_real = chir_re      if show_Real else None
            y_imag = chir_im      if show_Imag else None
    
            if normalize:
                cands = []
                if y_mag  is not None: cands.append(np.max(np.abs(y_mag)))
                if y_real is not None: cands.append(np.max(np.abs(y_real)))
                if y_imag is not None: cands.append(np.max(np.abs(y_imag)))
                denom = np.max(cands) if cands else None
                if denom and denom > 0:
                    if y_mag  is not None: y_mag  = y_mag  / denom
                    if y_real is not None: y_real = y_real / denom
                    if y_imag is not None: y_imag = y_imag / denom
    
            traces.append({"label": lb, "r": r, "y_mag": y_mag, "y_real": y_real, "y_imag": y_imag})
    
        if not traces:
            raise ValueError("No valid R/χ(R) data to plot after filtering.")
    
        # default offset step
        if stack == "offset" and offset_step is None:
            ptp_vals = []
            for tr in traces:
                ranges = []
                if tr["y_mag"]  is not None: ranges.append(np.ptp(tr["y_mag"]))
                if tr["y_real"] is not None: ranges.append(np.ptp(tr["y_real"]))
                if tr["y_imag"] is not None: ranges.append(np.ptp(tr["y_imag"]))
                if ranges:
                    ptp_vals.append(np.max(ranges))
            typical = np.median(ptp_vals) if ptp_vals else 1.0
            offset_step = 0.6 * typical
    
        # styles
        comp_style = {
            "mag":  dict(ls="-",  lw=1.5),
            "real": dict(ls="--", lw=1.5),
            "imag": dict(ls=":",  lw=1.5),
        }
    
        # plot
        if stack == "subplots":
            n = len(traces)
            fig, axes = plt.subplots(n, 1, figsize=(6, 2.6*n), sharex=True, sharey=sharey)
            if n == 1:
                axes = [axes]
            for i, tr in enumerate(traces):
                ax = axes[i]
                r, lb = tr["r"], tr["label"]
                if tr["y_mag"]  is not None: ax.plot(r, tr["y_mag"],  label=f"{lb} |χ(R)|",   **comp_style["mag"])
                if tr["y_real"] is not None: ax.plot(r, tr["y_real"], label=f"{lb} Re[χ(R)]", **comp_style["real"])
                if tr["y_imag"] is not None: ax.plot(r, tr["y_imag"], label=f"{lb} Im[χ(R)]", **comp_style["imag"])
                ax.set_ylabel(r"$\chi(R)$")
                if show_legend == True:
                    ax.legend(fontsize=9, ncol=1 if len(traces) < 8 else 2)
                ax.grid(alpha=0.0)
            axes[-1].set_xlabel(r"$R$ (Å)")
            plt.tight_layout()
        else:
            fig, ax = plt.subplots(figsize=(9, 6))
            for i, tr in enumerate(traces):
                r, lb = tr["r"], tr["label"]
                y_off = offset_baseline + i * offset_step if stack == "offset" else 0.0
                if tr["y_mag"]  is not None: ax.plot(r, tr["y_mag"]  + y_off, label=f"{lb} |χ(R)|",   **comp_style["mag"])
                if tr["y_real"] is not None: ax.plot(r, tr["y_real"] + y_off, label=f"{lb} Re[χ(R)]", **comp_style["real"])
                if tr["y_imag"] is not None: ax.plot(r, tr["y_imag"] + y_off, label=f"{lb} Im[χ(R)]", **comp_style["imag"])
            ax.set_xlabel(r"$R$ (Å)")
            ax.set_ylabel(r"$\chi(R)$" + ("  (stacked)" if stack == "offset" else ""))
            if show_legend == True:
                ax.legend(fontsize=9, ncol=1 if len(traces) < 8 else 2)
            ax.grid(alpha=0.0)
            plt.tight_layout()
    
        # output
        def _sanitize(s: str) -> str:
            return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)
    
        if outdir is None and filename is None:
            plt.show()
            return fig, None
    
        if filename:
            out_path = os.path.abspath(filename if os.path.isabs(filename)
                                       else os.path.join(outdir or ".", filename))
        else:
            lab_str = "ALL" if len(labels) > 3 else "_".join(_sanitize(lb) for lb in labels)
            base = ("R_STACK_" + stack.upper() + "_" if stack != "overlay" else "R_") + lab_str
            base += "_TRANS" if trans_only else ""
            out_path = os.path.abspath(os.path.join(outdir or ".", f"{base}.png"))
    
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.show()
        return fig, out_path
        
    from typing import Union
    
    def plot_wavelet(
        self,
        label: Union[str, Iterable[str]],
        outdir: Optional[str] = None,
        filename: Optional[str] = None,
        filepath: Optional[str] = None,
        dpi: int = 300,
        cmap_mag: str = "viridis",
        cmap_re: str = "viridis",
        # existing options
        stack: bool = False,              # False: per-label figures, True: stacked rows (|W| left, Re right)
        share_clim: bool = False,         # share color limits across rows if stacking
        # group helpers
        show_members: bool = False,
        members_only: bool = False,
        # channel selection
        trans_only: bool = False,
    ):
        """
        Plot Wavelet Transform maps: Magnitude (|W|) and Real part (Re{W}).
    
        If trans_only=True, uses attached transmission channel (grp._trans) when available.
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt
    
        # Helper: plot a single label (one figure with two panels)
        def _plot_single(single_label: str, _outdir, _filename, _filepath, _dpi, _cmap_mag, _cmap_re):
            g = self.groups.get(single_label)
            if g is None:
                raise KeyError(f"Group '{single_label}' not found in self.groups.")
            src = getattr(g, "_trans", None) if trans_only else None
            if trans_only and (src is None):
                raise ValueError(f"No transmission channel attached for '{single_label}'.")
            src = src if src is not None else g
    
            # Required attributes on the chosen source
            required = ["k", "wcauchy_r", "wcauchy_mag", "wcauchy_re"]
            missing = [a for a in required if getattr(src, a, None) is None]
            if missing:
                raise AttributeError(f"Missing required data on group '{single_label}': {', '.join(missing)}")
    
            wc_mag = np.asarray(getattr(src, "wcauchy_mag"))
            wc_re  = np.asarray(getattr(src, "wcauchy_re"))
            r_arr  = np.asarray(getattr(src, "wcauchy_r"))
            k_arr  = np.asarray(getattr(src, "k"))
    
            if wc_mag.ndim != 2 or wc_re.ndim != 2:
                raise ValueError("Wavelet arrays must be 2D (R x k).")
    
            k_min, k_max = float(np.nanmin(k_arr)), float(np.nanmax(k_arr))
            r_min, r_max = float(np.nanmin(r_arr)), float(np.nanmax(r_arr))
    
            kweight_eff = int(getattr(self, "_default_kweight", 2))
    
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
            im1 = axes[0].imshow(
                wc_mag, extent=[k_min, k_max, r_min, r_max], aspect="auto",
                origin="lower", cmap=_cmap_mag, interpolation="nearest",
            )
            axes[0].set_title("Wavelet Transform: Magnitude")
            axes[0].set_xlabel(r"$k$ (Å$^{-1}$)")
            axes[0].set_ylabel(r"$R$ (Å)")
            cbar1 = fig.colorbar(im1, ax=axes[0]); cbar1.set_label("|W|")
    
            vmax = float(np.nanmax(np.abs(wc_re))) if np.isfinite(wc_re).any() else None
            im2 = axes[1].imshow(
                wc_re, extent=[k_min, k_max, r_min, r_max], aspect="auto",
                origin="lower", cmap=_cmap_re, interpolation="nearest",
                vmin=(-vmax if vmax is not None else None), vmax=(vmax if vmax is not None else None),
            )
            axes[1].set_title("Wavelet Transform: Real Part")
            axes[1].set_xlabel(r"$k$ (Å$^{-1}$)")
            axes[1].set_ylabel(r"$R$ (Å)")
            cbar2 = fig.colorbar(im2, ax=axes[1]); cbar2.set_label("Re{W}")
    
            plt.tight_layout(rect=[0, 0, 1, 0.93])
            fig.suptitle(f"Wavelet – {single_label}" + (" [TRANS]" if trans_only else "") + f" (k-weight = {kweight_eff})")
    
            def _sanitize(s: str) -> str:
                return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)
    
            if _filepath:
                out_path = os.path.abspath(_filepath)
                fig.savefig(out_path, dpi=_dpi, bbox_inches="tight")
                plt.show()
                return fig, out_path
    
            if _outdir is None and _filename is None:
                plt.show()
                return fig, None
    
            if _filename:
                out_path = os.path.abspath(_filename if os.path.isabs(_filename) else os.path.join(_outdir or ".", _filename))
            else:
                filename_local = f"Wavelet_{_sanitize(single_label)}" + ("_TRANS" if trans_only else "") + ".png"
                out_path = os.path.abspath(os.path.join(_outdir or ".", filename_local))
            fig.savefig(out_path, dpi=_dpi, bbox_inches="tight")
            plt.show()
            return fig, out_path
    
        # Build labels (+ expand members if requested)
        sum_map = getattr(self, "_sum_map", {}) or {}
        def _expand_labels(labs):
            out, seen = [], set()
            for lb in labs:
                if show_members and (lb in sum_map):
                    mems = list(sum_map.get(lb, []))
                    to_add = (mems if members_only else [lb] + mems)
                else:
                    to_add = [lb]
                for x in to_add:
                    if x not in seen:
                        out.append(x); seen.add(x)
            return out
    
        # Normalize input into a list
        if isinstance(label, str):
            labels = [label]
        else:
            labels = list(label)
        if not labels:
            raise ValueError("No labels to plot.")
        if show_members:
            labels = _expand_labels(labels)
    
        # Option A: individual figures (original style)
        if not stack:
            results = []
            def _sanitize(s: str) -> str:
                return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)
            for lb in labels:
                per_fn = None
                if outdir is not None and filename is None:
                    per_fn = f"Wavelet_{_sanitize(lb)}" + ("_TRANS" if trans_only else "") + ".png"
                fig, path = _plot_single(
                    single_label=lb,
                    _outdir=outdir,
                    _filename=(filename or per_fn),
                    _filepath=None,
                    _dpi=dpi,
                    _cmap_mag=cmap_mag,
                    _cmap_re=cmap_re
                )
                results.append((lb, fig, path))
            return results
    
        # Option B: stacked rows (|W| left, Re right)
        # Gather for all labels
        records = []
        for lb in labels:
            g = self.groups.get(lb)
            if g is None:
                print(f"[WARN] Group '{lb}' not found; skipping.")
                continue
            src = getattr(g, "_trans", None) if trans_only else None
            if trans_only and (src is None):
                print(f"[WARN] No transmission channel attached for '{lb}'; skipping.")
                continue
            src = src if src is not None else g
    
            wc_mag = np.asarray(getattr(src, "wcauchy_mag", None))
            wc_re  = np.asarray(getattr(src, "wcauchy_re",  None))
            r_arr  = np.asarray(getattr(src, "wcauchy_r",   None))
            k_arr  = np.asarray(getattr(src, "k",           None))
            if any(a is None for a in (wc_mag, wc_re, r_arr, k_arr)):
                print(f"[WARN] Missing wavelet attributes for '{lb}'; skipping.")
                continue
            if wc_mag.ndim != 2 or wc_re.ndim != 2:
                print(f"[WARN] Wavelet arrays must be 2D (R x k) for '{lb}'; skipping.")
                continue
    
            k_min, k_max = float(np.nanmin(k_arr)), float(np.nanmax(k_arr))
            r_min, r_max = float(np.nanmin(r_arr)), float(np.nanmax(r_arr))
            records.append({"label": lb, "wc_mag": wc_mag, "wc_re": wc_re, "extent": [k_min, k_max, r_min, r_max]})
    
        if not records:
            raise ValueError("No wavelet datasets to plot.")
    
        mag_vmax = None
        re_vmax  = None
        if share_clim:
            all_mag = np.concatenate([rec["wc_mag"].ravel() for rec in records])
            all_re  = np.concatenate([rec["wc_re" ].ravel() for rec in records])
            finite_mag = all_mag[np.isfinite(all_mag)]
            finite_re  = all_re[np.isfinite(all_re)]
            mag_vmax = float(np.max(np.abs(finite_mag))) if finite_mag.size else 1.0
            re_vmax  = float(np.max(np.abs(finite_re ))) if finite_re.size  else 1.0
    
        kweight_eff = int(getattr(self, "_default_kweight", 2))
        n_rows = len(records)
        fig, axes = plt.subplots(n_rows, 2, figsize=(10, 3.2 * n_rows), squeeze=False)
    
        for i, rec in enumerate(records):
            vmax_m = mag_vmax if share_clim else float(np.nanmax(np.abs(rec["wc_mag"])))
            im1 = axes[i, 0].imshow(
                rec["wc_mag"], extent=rec["extent"], aspect="auto", origin="lower",
                cmap=cmap_mag, interpolation="nearest", vmin=0.0,
                vmax=vmax_m if np.isfinite(vmax_m) else None
            )
            axes[i, 0].set_title(f"{rec['label']} – |W|" + (" [TRANS]" if trans_only else ""))
            axes[i, 0].set_xlabel(r"$k$ (Å$^{-1}$)")
            axes[i, 0].set_ylabel(r"$R$ (Å)")
            cbar1 = fig.colorbar(im1, ax=axes[i, 0]); cbar1.set_label("|W|")
    
            vmax_r = re_vmax if share_clim else float(np.nanmax(np.abs(rec["wc_re"])))
            vmax_r = vmax_r if np.isfinite(vmax_r) else None
            im2 = axes[i, 1].imshow(
                rec["wc_re"], extent=rec["extent"], aspect="auto", origin="lower",
                cmap=cmap_re, interpolation="nearest",
                vmin=(-vmax_r if vmax_r is not None else None), vmax=(vmax_r if vmax_r is not None else None),
            )
            axes[i, 1].set_title(f"{rec['label']} – Re{{W}}" + (" [TRANS]" if trans_only else ""))
            axes[i, 1].set_xlabel(r"$k$ (Å$^{-1}$)")
            axes[i, 1].set_ylabel(r"$R$ (Å)")
            cbar2 = fig.colorbar(im2, ax=axes[i, 1]); cbar2.set_label("Re{W}")
    
        def _sanitize(s: str) -> str:
            return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)
    
        lbl_str = "ALL" if len(labels) > 3 else "_".join(_sanitize(lb) for lb in labels)
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        fig.suptitle(f"Wavelet – {lbl_str}" + (" [TRANS]" if trans_only else "") + f" (k-weight = {kweight_eff})")
    
        if outdir is None and filename is None:
            plt.show()
            return fig, None
    
        if filename:
            out_path = os.path.abspath(filename if os.path.isabs(filename)
                                       else os.path.join(outdir or ".", filename))
        else:
            fname = f"Wavelet_STACK_{lbl_str}" + ("_TRANS" if trans_only else "") + ".png"
            out_path = os.path.abspath(os.path.join(outdir or ".", fname))
    
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.show()
        return fig, out_path

    def plot_overlay_r(self, label: str):
        """Plot previously computed R-shell overlay from `overlay_r_shells`."""
        if label not in self.exafs or "overlay_r" not in self.exafs[label]:
            raise ValueError(f"No R-shell overlay data found for '{label}'. Run overlay_r_shells first.")
        overlay_data = self.exafs[label]["overlay_r"]
        plt.figure(figsize=(6, 4))
        for r_range, d in overlay_data.items():
            plt.plot(d["k"], d["chi"], label=f"{r_range[0]:.2f}–{r_range[1]:.2f} Å")
        plt.xlabel(r"$k$ (Å$^{-1}$)")
        plt.ylabel(r"$\chi(k)$")
        plt.title(f"R-shell overlay – {label}")
        plt.legend()
        plt.grid(alpha=0.0)
        plt.tight_layout()
        plt.show()

    def plot_overlay_k(self, label: str):
        """Plot FT R-shell overlays from `overlay_k_shells`."""
        if label not in self.exafs or "overlay_k" not in self.exafs[label]:
            raise ValueError(f"No overlay_k data found for '{label}'. Run overlay_k_shells first.")
        overlay_data = self.exafs[label]["overlay_k"]
        plt.figure(figsize=(6, 4))
        for (kmin, kmax), d in overlay_data.items():
            plt.plot(d["r"], d["chir"],
                     label=f"k = {kmin:.2f}–{kmax:.2f} Å$^{{-1}}$")
        plt.xlabel(r"$R$ (Å)")
        plt.ylabel(r"$|\chi(R)|$")
        plt.title(f"FT: k-window overlays – {label}")
        plt.legend()
        plt.grid(alpha=0.0)
        plt.tight_layout()
        plt.show()

    def plot_r_to_k(self, label: str, rmin: float, rmax: float, kweight: int):
        """
        Visualize an R-window selection and the corresponding back-transformed χ(k).
        """
        if label not in self.exafs or "r_to_k" not in self.exafs[label]:
            raise ValueError(f"No r_to_k data found for '{label}'. Run r_to_k() first.")
        g = self.groups[label]
        # R-space
        plt.figure(figsize=(6, 4))
        plt.plot(g.r, np.abs(g.chir), lw=1.8, color="tab:blue", label=r"$|\chi(R)|$")
        plt.axvspan(rmin, rmax, color="tab:orange", alpha=0.25,
                    label=f"R window [{rmin:.2f}, {rmax:.2f}] Å")
        plt.xlabel('R (Å)')
        plt.ylabel(r"$|\chi(R)|$")
        plt.title(f"{label}: R-space")
        plt.legend(); plt.grid(alpha=0.0); plt.tight_layout(); plt.show()

        # k-space (back-transform)
        plt.figure(figsize=(6, 4))
        plt.plot(g.q, (g.q ** kweight) * g.chiq, lw=1.8, color="tab:red",
                 label=rf"$k^{kweight}\chi_{{R\rightarrow k}}(k)$")
        plt.xlabel(r"$k$ (Å$^{-1}$)")
        plt.ylabel(rf"$k^{kweight}\chi(k)$")
        plt.title(rf"{label}: $R\rightarrow k$ back-transform [{rmin:.2f}, {rmax:.2f}] Å")
        plt.legend(); plt.grid(alpha=0.0); plt.tight_layout(); plt.show()

    def plot_k_to_r(self, label: str, kmin: float, kmax: float, kweight: int):
        """
        Visualize a k-window selection and the corresponding forward-transformed χ(R).
        """
        if label not in self.exafs or "k_to_r" not in self.exafs[label]:
            raise ValueError(f"No k_to_r data found for '{label}'. Run k_to_r() first.")
        g = self.groups[label]
        # k window highlight
        plt.figure(figsize=(6, 4))
        plt.plot(g.k, (g.k ** kweight) * g.chi, lw=1.8, color="tab:red",
                 label=rf"$k^{kweight}\chi(k)$")
        plt.axvspan(kmin, kmax, color="tab:orange", alpha=0.25,
                    label=f"k window [{kmin:.2f}, {kmax:.2f}] Å$^{{-1}}$")
        plt.xlabel(r"$k$ (Å$^{-1}$)")
        plt.ylabel(rf"$k^{kweight}\chi(k)$")
        plt.title(rf"{label}: $k\rightarrow R$ forward transform highlight")
        plt.legend(); plt.grid(alpha=0.0); plt.tight_layout(); plt.show()
        # R-space
        plt.figure(figsize=(6, 4))
        plt.plot(g.r, np.abs(g.chir), lw=1.8, color="tab:blue", label=r"$|\chi(R)|$")
        plt.xlabel('R (Å)')
        plt.ylabel(r"$|\chi(R)|$")
        plt.title(f"{label}: R-space")
        plt.legend(); plt.grid(alpha=0.0); plt.tight_layout(); plt.show()

    # ------------------- Saving (extras) -------------------
    @staticmethod
    def save_map2d(Z: np.ndarray, fname: str) -> None:
        np.savetxt(fname, Z)

    def save_wavelet(self, folder: str = ".", label: str = "") -> None:
        d = self.exafs.get(label, {}).get("wavelet")
        if not d:
            raise ValueError(f"No wavelet data for {label}")
        path = os.path.join(folder, f"wavelet_{label}.dat")
        self.save_map2d(self.groups[label].wcauchy_mag, path)

    def compute_ft(
        self, labels: Optional[Iterable[str]] = None,
        kmin: float = 2, kmax: float = 12, kweight: int = 2,
        dk: float = 1, window: str = "hanning", rmax_out=None
    ) -> None:
        """Convenience wrapper to compute forward FT for multiple labels."""
        if labels is None:
            labels = list(self.groups.keys())
        elif isinstance(labels, str):
            labels = [labels]
        for label in labels:
            g = self.groups[label]
            xftf(g, kmin=kmin, kmax=kmax, kweight=kweight, dk=dk, window=window, rmax_out=rmax_out)


# =========================================================
# Universal input preparation for .dat files
#   - Supports ignore tokens as standalone words
#   - Supports glob-style patterns in ignore_tokens (e.g., 'WSD*', 'NC*')
#   - Supports separate ignore_patterns (glob list)
#   - Includes include_glob / exclude_glob filters on basename (no extension)
#   - Labels/group names: compact alphanumeric from first file's basename (no prefixes/suffixes)
# =========================================================


# --------------------------
# Utilities
# --------------------------

def _strip_quotes(s: str) -> str:
    return s.strip().strip("'").strip('"')

def _energy_tuple(token: str) -> Tuple[int, ...]:
    """Convert '15-4-5' or '15_7' or '15_6-5' → (15, 4, 5)."""
    parts = re.split(r'[-_]', token)
    return tuple(int(p) for p in parts if p.isdigit())

def _sanitize_label_token(s: str, drop_prefixes: Optional[List[str]] = None) -> str:
    """
    Legacy sanitizer (kept in case you need it elsewhere).
    Not used for label_map anymore; replaced by _alnum_compact on the filename.
    """
    s = _strip_quotes(s)
    drop_prefixes = drop_prefixes or []
    for pref in drop_prefixes:
        if s.startswith(pref):
            s = s[len(pref):]
    s = re.sub(r'\s+', ' ', s).strip()
    s = s.replace(' ', '_')
    s = ''.join(c for c in s if c.isalnum() or c in '-_.')
    return s or "unnamed"

def _alnum_compact(s: str) -> str:
    """
    Return only letters and digits from s, in order (preserve original case).
    Examples:
      '15-6_5_Nd_2'  -> '1565Nd2'
      'norm_ZnHis10_Cryo' -> 'normZnHis10Cryo'
      ' Fe -  K-edge ' -> 'FeKedge'
      '---' -> 'unnamed' (fallback if no alnum)
    """
    s = _strip_quotes(str(s))
    keep = re.findall(r'[A-Za-z0-9]+', s)
    out = ''.join(keep)
    return out if out else "unnamed"

def _matches_any_glob(name: str, patterns: Optional[List[str]]) -> bool:
    """Case-insensitive fnmatch (glob) for any of the patterns."""
    if not patterns:
        return False
    low = name.lower()
    return any(fnmatch.fnmatch(low, p.lower()) for p in patterns)

def _basename_noext(p: str) -> str:
    return os.path.splitext(os.path.basename(p))[0]

def _has_wildcard(s: str) -> bool:
    """Return True if s contains glob wildcard characters."""
    return any(ch in s for ch in ['*', '?', '[', ']'])

# --------------------------
# Ignore logic
# --------------------------

def should_ignore(filename: str,
                  ignore_tokens: Optional[List[str]] = None,
                  ignore_patterns: Optional[List[str]] = None) -> bool:
    """
    Return True if `filename` should be ignored.

    - ignore_tokens: list where each entry is either
        (A) a standalone word token (case-insensitive), using non-alphanumeric boundaries, e.g., 'loc'
            -> matches '... _loc_ ...' but not 'block'
        (B) a glob-style pattern (case-insensitive) if it contains wildcard characters *, ?, []
            -> e.g., 'WSD*', 'NC*', '*Calibration*'
    - ignore_patterns: additional glob-style patterns (case-insensitive), same semantics as (B).

    Both are applied to the **basename without path** (including extension).
    """
    base = _strip_quotes(os.path.basename(filename))

    # 1) Pattern-based ignore (glob on full basename)
    if _matches_any_glob(base, ignore_patterns):
        return True

    # 2) Token-based ignore
    ignore_tokens = ignore_tokens or []
    wildcard_tokens = [t for t in ignore_tokens if _has_wildcard(t)]
    plain_tokens    = [t for t in ignore_tokens if not _has_wildcard(t)]

    # 2a) Glob-like tokens (match on full basename)
    if _matches_any_glob(base, wildcard_tokens):
        return True

    # 2b) Plain tokens as standalone words (word boundary via non-alphanumeric)
    for tok in plain_tokens:
        patt = re.compile(rf'(?<![A-Za-z0-9]){re.escape(tok)}(?![A-Za-z0-9])', re.IGNORECASE)
        if patt.search(base):
            return True

    return False

# --------------------------
# File collection
# --------------------------

def get_dat_files(base_dir: str) -> List[str]:
    """Return all .dat files (filenames only) inside a directory."""
    return [f for f in os.listdir(base_dir) if f.lower().endswith(".dat")]

# --------------------------
# Parsers
# --------------------------

def parse_energy_style(fname: str) -> Optional[Dict]:
    """
    Match energy-style: <energy_token>_Nd_<#>.dat
    Returns dict or None if not matched.
    """
    f = _strip_quotes(fname)
    m = re.match(r"^([\w\-]+)_Nd_(\d+)\.dat$", f, flags=re.IGNORECASE)
    if not m:
        return None
    token = m.group(1)
    nd = int(m.group(2))
    return {"kind": "energy", "name": f, "token": token, "tuple": _energy_tuple(token), "nd": nd}

def parse_scan_style(fname: str) -> Optional[Dict]:
    """
    Match scan-style variants (priority order):

      1) <prefix> Scan<scan>[_<rep>].dat
      2) <prefix> Scan <scan>[_<rep>].dat
      3) <prefix> <scan>[_<rep>].dat
      4) <prefix>_<scan>[_<rep>].dat  (guard: prefix must NOT end with 'Scan')
      5) <prefix>.dat                 (single file treated as scan=1)

    Returns dict or None.
    """
    f = _strip_quotes(fname)

    # 1) 'Scan<scan>' (no space)
    m = re.match(r'^(?P<prefix>.+?)\s+Scan(?P<scan>\d+)\s*(?:_(?P<rep>\d+))?\.dat$', f, flags=re.IGNORECASE)
    if m:
        return {"kind": "scan", "name": f, "prefix": m.group('prefix').strip(),
                "scan": int(m.group('scan')), "rep": int(m.group('rep')) if m.group('rep') else 0}

    # 2) 'Scan <scan>'
    m = re.match(r'^(?P<prefix>.+?)\s+Scan\s+(?P<scan>\d+)\s*(?:_(?P<rep>\d+))?\.dat$', f, flags=re.IGNORECASE)
    if m:
        return {"kind": "scan", "name": f, "prefix": m.group('prefix').strip(),
                "scan": int(m.group('scan')), "rep": int(m.group('rep')) if m.group('rep') else 0}

    # 3) space before number (no 'Scan' keyword)
    m = re.match(r'^(?P<prefix>.+?)\s+(?P<scan>\d+)\s*(?:_(?P<rep>\d+))?\.dat$', f, flags=re.IGNORECASE)
    if m:
        return {"kind": "scan", "name": f, "prefix": m.group('prefix').strip(),
                "scan": int(m.group('scan')), "rep": int(m.group('rep')) if m.group('rep') else 0}

    # 4) underscore before number, but prefix must NOT end with 'Scan'
    m = re.match(r'^(?P<prefix>.+?)(?<![Ss]can)_(?P<scan>\d+)(?:_(?P<rep>\d+))?\.dat$', f, flags=re.IGNORECASE)
    if m:
        return {"kind": "scan", "name": f, "prefix": m.group('prefix').strip(),
                "scan": int(m.group('scan')), "rep": int(m.group('rep')) if m.group('rep') else 0}

    # 5) single file with no numeric suffix → treat as single scan (scan=1)
    m = re.match(r'^(?P<prefix>.+?)\.dat$', f, flags=re.IGNORECASE)
    if m:
        return {"kind": "scan", "name": f, "prefix": m.group('prefix').strip(),
                "scan": 1, "rep": 0}

    return None

# --------------------------
# Main: prepare_inputs_any
# --------------------------

def prepare_inputs_any(base_dir: str,
                       ignore_tokens: Optional[List[str]] = None,
                       drop_prefixes_for_scan_tokens: Optional[List[str]] = None,
                       energy_label_prefix: str = "Nd",          # kept for signature; NOT used
                       scan_label_prefix: Optional[str] = None,  # kept for signature; NOT used
                       prefer_energy_first: bool = True,
                       reset_scan_numbers_per_group: bool = False,
                       include_glob: Optional[List[str]] = None,
                       exclude_glob: Optional[List[str]] = None,
                       ignore_patterns: Optional[List[str]] = None,
                       *,
                       # --- NEW preview controls ---
                       preview_input: bool = True,
                       preview_n_files: int = 150,
                       preview_n_summed: int = 120):
    """
    Universal loader:
      - Includes ALL *.dat files (except those filtered by ignore_tokens / ignore_patterns)
      - Parses both energy-style and scan-style filenames
      - Groups by energy token (energy) or sanitized prefix (scan)
      - Sorts within group (energy: by Nd; scan: by scan,rep)
      - Returns:
          files: {'Scan1': '/path/a.dat', ...}
          filenames_sorted: [ 'a.dat', 'b.dat', ... ]
          groups_by_token: OrderedDict[token] -> [global scan indices]
          label_map: token -> label (alphanumeric-compact of first filename's basename)
          summed_groups: [{members: [global Scan#...], label: ...}, ...]

      If preview_input=True, prints a preview of the prepared inputs using `preview_inputs(...)`.
      Example:
      files, filenames_sorted, groups_by_token, label_map, summed_groups = prepare_inputs_any(
                base_dir=base_dir,
                # exclude_glob = ['*'],   # Exclude data files you want to plot
                include_glob = ['*_FeK_*'],  # Include data files you want to plot
                scan_label_prefix=None,             # kept in signature; labels ignore prefixes by design now
                prefer_energy_first=True,           # energy groups listed first (if any)
                reset_scan_numbers_per_group=False,  # GLOBAL indices (recommended for EXAFS workflows)
                preview_input=False,
            )
    """
    ignore_tokens = ignore_tokens or ['loc', 'wheel', 'jj', 'stage']
    drop_prefixes_for_scan_tokens = drop_prefixes_for_scan_tokens or ["EXAFS_"]  # retained for compatibility

    # Collect and filter *.dat files
    all_files = get_dat_files(base_dir)
    all_files = [f for f in all_files if not should_ignore(f, ignore_tokens, ignore_patterns)]

    # --- apply glob-style include/exclude on basename (no extension) ---
    def _match_any(name: str, patterns: Optional[List[str]]) -> bool:
        if not patterns:
            return False
        low = name.lower()
        return any(fnmatch.fnmatch(low, pat.lower()) for pat in patterns)

    if include_glob:
        all_files = [f for f in all_files if _match_any(os.path.splitext(f)[0], include_glob)]
    if exclude_glob:
        all_files = [f for f in all_files if not _match_any(os.path.splitext(f)[0], exclude_glob)]

    if not all_files:
        raise FileNotFoundError(f"No acceptable .dat files found in {base_dir}")

    # Parse all files
    energy_items: List[Dict] = []
    scan_items: List[Dict] = []
    for f in all_files:
        pe = parse_energy_style(f)
        if pe:
            energy_items.append(pe)
            continue
        ps = parse_scan_style(f)
        if ps:
            scan_items.append(ps)
            continue
        # if neither matched, treat as single-scan by basename:
        base = os.path.splitext(f)[0]
        scan_items.append({"kind": "scan", "name": f, "prefix": base, "scan": 1, "rep": 0})

    # Build grouping for energy-style
    token_to_energy_items: Dict[str, List[Dict]] = {}
    for it in energy_items:
        token_to_energy_items.setdefault(it["token"], []).append(it)
    for tok in token_to_energy_items:
        token_to_energy_items[tok].sort(key=lambda d: d["nd"])  # Nd order

    # Build grouping for scan-style
    prefix_to_scan_items: Dict[str, List[Dict]] = {}
    for it in scan_items:
        prefix_to_scan_items.setdefault(it["prefix"], []).append(it)
    for pref in prefix_to_scan_items:
        prefix_to_scan_items[pref].sort(key=lambda d: (d["scan"], d["rep"]))

    # Decide token order
    energy_tokens = sorted(token_to_energy_items.keys(), key=_energy_tuple)
    scan_tokens = sorted(prefix_to_scan_items.keys(),
                         key=lambda s: _sanitize_label_token(s, drop_prefixes_for_scan_tokens).lower())

    ordered_tokens: List[Tuple[str, str]] = []  # list of (kind, token)
    if prefer_energy_first and energy_tokens:
        ordered_tokens.extend([("energy", tok) for tok in energy_tokens])
        ordered_tokens.extend([("scan", tok) for tok in scan_tokens])
    else:
        ordered_tokens.extend([("scan", tok) for tok in scan_tokens])
        ordered_tokens.extend([("energy", tok) for tok in energy_tokens])

    # Flatten filenames in that order
    filenames_sorted: List[str] = []
    groups_by_token: "OrderedDict[str, List[int]]" = OrderedDict()
    idx = 1
    for kind, tok in ordered_tokens:
        if kind == "energy":
            names = [d["name"] for d in token_to_energy_items[tok]]
        else:
            names = [d["name"] for d in prefix_to_scan_items[tok]]
        start = idx
        filenames_sorted.extend(names)
        idx += len(names)
        groups_by_token[tok] = list(range(start, idx))

    # files dict with global Scan indices
    files = {f"Scan{i}": os.path.join(base_dir, fname)
             for i, fname in enumerate(filenames_sorted, start=1)}

    # ---- label_map: alphanumeric-compact from the FIRST file's basename in each group ----
    label_map: Dict[str, str] = {}
    for kind, tok in ordered_tokens:
        if kind == "energy":
            first_name = token_to_energy_items[tok][0]["name"]
        else:
            first_name = prefix_to_scan_items[tok][0]["name"]
        base_noext = os.path.splitext(first_name)[0]
        label_map[tok] = _alnum_compact(base_noext)

    # summed_groups
    if reset_scan_numbers_per_group:
        # WARNING: only for display; not recommended for align_multiple()
        summed_groups = [
            {"members": [f"Scan{i}" for i in range(1, len(groups_by_token[tok]) + 1)],
             "label": label_map[tok]}
            for tok in groups_by_token.keys()
        ]
    else:
        # GLOBAL indices (recommended for EXAFS)
        summed_groups = [
            {"members": [f"Scan{i}" for i in groups_by_token[tok]],
             "label": label_map[tok]}
            for tok in groups_by_token.keys()
        ]

    # --- NEW: optional preview ---
    if preview_input:
        preview_inputs(
            files=files,
            groups_by_token=groups_by_token,
            label_map=label_map,
            summed_groups=summed_groups,
            n_files=preview_n_files,
            n_summed=preview_n_summed
        )

    return files, filenames_sorted, groups_by_token, label_map, summed_groups



# =========================================================
# Standalone plotters for energy vs μ_trans text files
#   - Wildcards supported in include_samples / exclude_samples (glob-style)
#   - Wildcards supported in include_regex / exclude_regex (auto-translated to regex)
#   - Backward compatible with exact sample matching and raw regex entries
# =========================================================



def plot_energy_mutrans(
    files,
    *,
    stacked=False,
    offset_step=None,
    title=None,
    xlabel="Energy (eV)",
    ylabel="μ (E)",
    legend=True,
    legend_names=None,     # str | list[str] | dict[str,str] | callable(path, pair_no, file_index)->str
    figsize=(9, 6),
    savepath=None,
    xlim=None,
    ylim=None,
    nan_policy="drop",     # "drop" | "keep"
    delimiter=None,        # None=auto (whitespace, comma, tab)
    debug=False,
    # ---- FILTERS ----
    include_signals=None,   # e.g., ['norm', 'flat']
    exclude_signals=None,
    include_samples=None,   # sample keys (basename after prefix). Supports exact and glob wildcards.
    exclude_samples=None,   # supports exact and glob wildcards.
    include_regex=None,     # regex or glob patterns on basename (no extension)
    exclude_regex=None,     # regex or glob patterns on basename (no extension)
    # ---- STYLE ----
    flat_linestyle="-",     # dashed for flat to distinguish, same color as norm
    others_linestyle="-",   # style for other signals (norm/mu/mu_bkgsub/etc.)
):
    """
    Plot one or more datasets where:
      - each file's FIRST column is Energy (x1), SECOND column is μ_trans (y1),
      - if a file has extra columns, they are interpreted as (x2,y2), (x3,y3), ...
        i.e., column pairs [0,1], [2,3], [4,5], ...

    Enhancements:
      • Files named like '{signal}_{label}.dat' (e.g., 'norm_ZnHis10_Cryo.dat',
        'flat_ZnHis10_Cryo.dat') will share the SAME color across signals
        for the SAME {label}. This makes 'flat' and 'norm' visually matched.
      • Filters by signal/sample/regex with wildcard support.
      • When no custom legend mapping is provided, default legend shows
        '{sample} ({signal})' to avoid duplicate legend entries.
    """

    # ---------- tiny helpers ----------
    def _auto_color(i: int) -> str:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        return colors[i % len(colors)]

    def _basename_noext(p: str) -> str:
        return os.path.splitext(os.path.basename(p))[0]

    _KNOWN_PREFIXES = ("norm", "flat", "mu", "mutrans", "mu_raw", "mu_bkgsub", "bkg", "pre_edge", "post_edge")

    def _parse_signal_and_sample(base_noext: str):
        """Return (signal or None, sample_key)."""
        for pref in _KNOWN_PREFIXES:
            if base_noext.startswith(pref + "_"):
                return pref, base_noext[len(pref) + 1 :]
        return None, base_noext

    # ---------- header-aware readers ----------
    def _first_data_line_index(path: str) -> int:
        with open(path, "r", errors="ignore") as f:
            for i, line in enumerate(f):
                s = line.split("#", 1)[0]
                tokens = s.replace(",", " ").replace("\t", " ").split()
                cnt = 0
                for t in tokens:
                    try:
                        float(t)
                        cnt += 1
                    except Exception:
                        pass
                if cnt >= 2:
                    return i
        return 0

    def _load_matrix_with_skip(path: str, skip: int, delim):
        arr = None
        try:
            arr = np.genfromtxt(
                path, comments="#", autostrip=True, delimiter=delim,
                invalid_raise=False, skip_header=skip
            )
            if arr.ndim == 1:
                arr = arr[:, None]
        except Exception:
            arr = None
        if (arr is None) or (arr.size == 0) or (arr.ndim != 2) or (arr.shape[1] < 2):
            try:
                arr = np.loadtxt(path, comments="#", delimiter=delim, ndmin=2, skiprows=skip)
                if arr.ndim == 1:
                    arr = arr[:, None]
            except Exception:
                arr = None
        return arr

    def _numeric_fallback(path: str, skip: int):
        rows = []
        with open(path, "r", errors="ignore") as f:
            for i, line in enumerate(f):
                if i < skip:
                    continue
                line = line.split("#", 1)[0]
                toks = line.replace(",", " ").replace("\t", " ").split()
                nums = []
                for t in toks:
                    try:
                        nums.append(float(t))
                    except Exception:
                        pass
                if len(nums) >= 2:
                    rows.append(nums)
        if not rows:
            return None
        maxlen = max(len(r) for r in rows)
        pad = np.full((len(rows), maxlen), np.nan)
        for i, r in enumerate(rows):
            pad[i, :len(r)] = r
        return pad

    def _read_matrix(path: str):
        skip = _first_data_line_index(path)

        # user delimiter
        if delimiter is not None:
            arr = _load_matrix_with_skip(path, skip, delimiter)
            if arr is not None and arr.shape[1] >= 2:
                return arr

        # auto delimiters
        for delim in (None, ",", "\t"):
            arr = _load_matrix_with_skip(path, skip, delim)
            if arr is not None and arr.shape[1] >= 2:
                return arr

        # numeric fallback
        arr = _numeric_fallback(path, skip)
        if arr is not None and arr.ndim == 1:
            arr = arr[:, None]
        return arr

    def _iter_pairs(arr):
        ncol = arr.shape[1]
        last_even = (ncol // 2) * 2
        for k in range(0, last_even, 2):
            x = arr[:, k]
            y = arr[:, k + 1]
            yield x, y, (k // 2) + 1

    # ---------- wildcard + regex helpers ----------
    def _has_wildcard(s: str) -> bool:
        return any(ch in s for ch in ['*', '?', '[', ']'])

    def _match_any_glob_ci(name: str, patterns) -> bool:
        """Case-insensitive glob matching for any pattern in patterns."""
        if not patterns:
            return False
        low = name.lower()
        return any(fnmatch.fnmatch(low, pat.lower()) for pat in patterns)

    def _compile_mixed_patterns(patterns):
        """
        Given a list of patterns that may be raw regex or glob patterns,
        return a list of compiled regex objects. Glob patterns are translated.
        """
        compiled = []
        if not patterns:
            return compiled
        for pat in patterns:
            if _has_wildcard(pat):
                # translate glob -> regex string, then compile with IGNORECASE
                regex_str = fnmatch.translate(pat)
                compiled.append(re.compile(regex_str, re.IGNORECASE))
            else:
                # treat as raw regex
                compiled.append(re.compile(pat, re.IGNORECASE))
        return compiled

    # ---------- legend resolver ----------
    if isinstance(legend_names, str):
        legend_names_list = [legend_names] * len(files)
    else:
        legend_names_list = None

    def _legend_for(path: str, pair_no: int, file_index: int, sample_key: str, signal: str | None) -> str:
        base = _basename_noext(path)

        # Callable
        if callable(legend_names):
            try:
                return str(legend_names(path, pair_no, file_index))
            except Exception:
                pass

        # Dict (path or basename)
        if isinstance(legend_names, dict):
            if path in legend_names:
                name = str(legend_names[path])
                return name if pair_no == 1 else f"{name} [pair {pair_no}]"
            if base in legend_names:
                name = str(legend_names[base])
                return name if pair_no == 1 else f"{name} [pair {pair_no}]"

        # List aligned with files
        names_list = legend_names_list if legend_names_list is not None else legend_names
        if isinstance(names_list, list):
            if 0 <= file_index < len(names_list):
                name = str(names_list[file_index])
                return name if pair_no == 1 else f"{name} [pair {pair_no}]"

        # Fallback
        if signal:
            return f"{sample_key} ({signal})" if pair_no == 1 else f"{sample_key} ({signal}) [pair {pair_no}]"
        else:
            return sample_key if pair_no == 1 else f"{sample_key} [pair {pair_no}]"

    # ---------- FILTERS ----------
    # sample filters: exact or glob wildcards (case-insensitive)
    def _ok_samples(val: str | None, allow, deny) -> bool:
        # allow/deny can be None or list[str] (exact or glob)
        if allow is not None:
            if val is None:
                return False
            # exact OR wildcard acceptance
            exact_ok = val in allow
            glob_ok = _match_any_glob_ci(val, [p for p in allow if _has_wildcard(p)])
            if not (exact_ok or glob_ok):
                # if allow was provided, it must pass
                return False
        if deny is not None and val is not None:
            exact_bad = val in deny
            glob_bad = _match_any_glob_ci(val, [p for p in deny if _has_wildcard(p)])
            if exact_bad or glob_bad:
                return False
        return True

    # regex filters: raw regex OR glob converted to regex
    compiled_include = _compile_mixed_patterns(include_regex)
    compiled_exclude = _compile_mixed_patterns(exclude_regex)

    def _ok_regex(name: str) -> bool:
        if compiled_include:
            if not any(r.search(name) for r in compiled_include):
                return False
        if compiled_exclude:
            if any(r.search(name) for r in compiled_exclude):
                return False
        return True

    # signal filters: exact only (as before)
    def _ok_signals(val: str | None, allow, deny) -> bool:
        if allow is not None and (val is None or val not in allow):
            return False
        if deny is not None and val in deny:
            return False
        return True

    # ---------- main ----------
    if not files:
        raise ValueError("No files provided.")

    all_traces = []
    y_spans = []

    # Ensure consistent colors per sample label (norm & flat share color)
    sample_color_map = {}
    next_color_index = 0

    for fidx, path in enumerate(files):
        if not os.path.isfile(path):
            print(f"[plot] Warning: file not found, skipping → {path}")
            continue

        base_noext = _basename_noext(path)
        signal, sample_key = _parse_signal_and_sample(base_noext)

        # ---- Apply filters ----
        if not _ok_signals(signal, include_signals, exclude_signals):
            if debug:
                print(f"[filter] skip by signal: base={base_noext}, signal={signal}")
            continue
        if not _ok_samples(sample_key, include_samples, exclude_samples):
            if debug:
                print(f"[filter] skip by sample: base={base_noext}, sample={sample_key}")
            continue
        if not _ok_regex(base_noext):
            if debug:
                print(f"[filter] skip by regex/glob: base={base_noext}")
            continue

        # Assign/remember a color per sample_key
        if sample_key not in sample_color_map:
            sample_color_map[sample_key] = _auto_color(next_color_index)
            next_color_index += 1
        color = sample_color_map[sample_key]

        arr = _read_matrix(path)
        if arr is None or arr.shape[1] < 2:
            print(f"[plot] Could not recover >=2 numeric columns. "
                  f"Try delimiter=',' or delimiter='\\t' and/or debug=True\n  file: {path}")
            continue

        pairs_found = 0
        for x, y, pair_no in _iter_pairs(arr):
            if nan_policy == "drop":
                mask = np.isfinite(x) & np.isfinite(y)
                x = x[mask]; y = y[mask]
            if x.size == 0 or y.size == 0:
                continue

            label = _legend_for(path, pair_no, fidx, sample_key, signal)

            # Choose linestyle based on signal (flat gets different style)
            ls = flat_linestyle if signal == "flat" else others_linestyle

            all_traces.append({
                "label": label,
                "x": x,
                "y": y,
                "color": color,
                "linestyle": ls,
            })
            y_spans.append(float(np.nanmax(y) - np.nanmin(y)))
            pairs_found += 1

        if debug:
            print(f"[read] {path}: shape={arr.shape}, usable_pairs={pairs_found}, "
                  f"parsed_signal={signal}, sample_key={sample_key}, color={sample_color_map[sample_key]}")

    if len(all_traces) == 0:
        raise RuntimeError("No plottable traces found (check files/columns and filters).")

    if offset_step is None:
        offset_step = 0.08 * max(y_spans) if (stacked and len(y_spans) > 0) else 0.0

    # ---------- plot ----------
    plt.figure(figsize=figsize)
    for i, tr in enumerate(all_traces):
        x = tr["x"]; y = tr["y"]; label = tr["label"]
        color = tr["color"]; ls = tr["linestyle"]
        off = (i * offset_step) if stacked else 0.0
        plt.plot(x, y + off, lw=1.8, color=color, ls=ls, label=label)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    if xlim:
        plt.xlim(*xlim)
    if ylim:
        plt.ylim(*ylim)
    plt.grid(alpha=0.0)
    if legend:
        plt.legend(fontsize=9, ncol=2)
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=200)

    plt.show()


def plot_energy_mutrans_from_folder(folder: str, pattern: str = "*.dat", **plot_kwargs):
    """
    Find files by pattern inside 'folder' and plot them using plot_energy_mutrans(...).

    Example:
      plot_energy_mutrans_from_folder("/gpfs/Scratch/ufondup/Se_sample", "*.dat",
                                      stacked=True, title="All *.dat in folder")
    """
    files = sorted(glob.glob(os.path.join(folder, pattern)))
    if not files:
        raise FileNotFoundError(f"No files found in {folder!r} matching {pattern!r}")
    return plot_energy_mutrans(files, **plot_kwargs)

def sample_labels(label_map):
    """
    Extract all labels from a dict and return them
    formatted as a Python list of strings.
    """
    labels = []

    for tok, lbl in label_map.items():
        # print(f" '{lbl}',")   # print exactly like your format
        labels.append(lbl)

    # Final Python list formatted output
    print("[")
    for lbl in labels:
        print(f" '{lbl}',")
    print("]")

    return labels

def preview_inputs(
    files,
    groups_by_token,
    label_map,
    summed_groups,
    n_files=150,
    n_summed=120
):
    """
    Preview the structure of inputs returned by prepare_inputs_any().

    Parameters
    ----------
    files : dict
        Mapping of scan_key -> full_path.
    groups_by_token : dict
        Mapping token -> list of global indices.
    label_map : dict
        Mapping token -> compact label.
    summed_groups : list of dict
        Each dict has: {"label": <label>, "members": [...]}
    n_files : int
        How many file entries to preview.
    n_summed : int
        How many summed groups to preview.

    Returns
    -------
    None (prints formatted summaries)
    """

    # --- Preview: first N Scan entries ---
    print(f"First {n_files} files:")
    for i, (scan_key, full_path) in enumerate(files.items()):
        if i >= n_files:
            break
        print(f"  {scan_key} -> {full_path}")

    # --- Show grouping ---
    print("\nGroups (global indices):")
    for tok, idxs in groups_by_token.items():
        scan_ids = [f"Scan{i}" for i in idxs]
        print(f"  {tok}: {scan_ids}")

    # --- Show the labels (one per group) ---
    print("\nLabel map (token -> compact label):")
    for tok, lbl in label_map.items():
        print(f" '{lbl}',")

    # --- Summed groups (compact preview) ---
    print(f"\nSummed groups (up to {n_summed}):")
    for g in summed_groups[:n_summed]:
        print(f"  {g['label']}: {g['members']}")



class ProjectPathNotFound(FileNotFoundError):
    pass

def project_number(project_id: str, must_exist: bool = True) -> Path:
    """
    Build the BioXAS export path for the given project ID (e.g., '37G12992').
    If must_exist=True, raises ProjectPathNotFound if the directory doesn't exist.
    Example:
    base_dir = bioxas_dir("37G12992", must_exist=True)
    """
    project_id = str(project_id).strip()
    path = Path("/beamlinedata/BIOXAS-SPECTROSCOPY/projects") / f"prj{project_id}" \
           / "main" / "raw" / "acquamanData" / "exportData" / "BioXAS"
    if must_exist and not path.is_dir():
        raise ProjectPathNotFound(f"Directory not found: {path}")
    return path


#-----------------------------------------------------------------#
#                  Deglitcher Code.
#-----------------------------------------------------------------#

class XASDeglitcher:
    def __init__(self, window=5, threshold=9):
        self.window = int(window)
        self.threshold = float(threshold)

    def _local_stats(self, data):
        n = len(data)
        med = np.zeros(n, dtype=float)
        sigma = np.zeros(n, dtype=float)

        for i in range(self.window, n - self.window):
            local = data[i - self.window:i + self.window + 1]
            median = np.median(local)
            mad = np.median(np.abs(local - median))
            med[i] = median
            if mad > 0:
                sigma[i] = 0.165325 * mad
            else:
                sigma[i] = 0.0
        return med, sigma

    def _select_region(self, x, region=None, x_range=None, E0=None):
        """
        Determine which indices belong to the selected XAS region.

        For region="custom", x_range can be:
            tuple  -> (xmin, xmax)
            dict   -> {"name": (xmin, xmax), ...}
        """
        if region is None:
            return np.arange(len(x))

        # custom region
        if region == "custom":
            if isinstance(x_range, tuple) and len(x_range) == 2:
                xmin, xmax = x_range
                return np.where((x >= xmin) & (x <= xmax))[0]
            if isinstance(x_range, dict):
                mask = np.zeros(len(x), dtype=bool)
                for _key, (xmin, xmax) in x_range.items():
                    mask |= (x >= xmin) & (x <= xmax)
                return np.where(mask)[0]
            raise ValueError("x_range must be tuple or dictionary for custom region")

        # standard regions require E0
        if E0 is None:
            raise ValueError("E0 energy must be provided for region selection")

        if region == "pre":
            return np.where(x < E0 - 200)[0]
        if region == "xanes":
            return np.where((x >= E0 - 50) & (x <= E0 + 50))[0]
        if region == "exafs":
            return np.where(x > E0 + 100)[0]

        return np.arange(len(x))

    def detect_glitches(self, x, y, region=None, x_range=None, E0=None):
        y = np.asarray(y, dtype=float)
        x = np.asarray(x, dtype=float)
        n = len(y)

        region_idx = set(self._select_region(x, region, x_range, E0))
        glitch_indices = set()

        med, sigma = self._local_stats(y)

        # amplitude
        for i in range(self.window, n - self.window):
            if i not in region_idx:
                continue
            if sigma[i] == 0:
                continue
            z = abs(y[i] - med[i]) / sigma[i]
            if z > self.threshold:
                glitch_indices.add(i)

        # first derivative
        dy = np.gradient(y, x)
        med_dy = np.median(dy)
        mad_dy = np.median(np.abs(dy - med_dy))
        if mad_dy > 0:
            sigma_dy = 0.165325 * mad_dy
            z_dy = np.abs(dy - med_dy) / sigma_dy
            for i in np.where(z_dy > self.threshold)[0]:
                if i in region_idx:
                    glitch_indices.add(i)

        # second derivative
        d2y = np.gradient(dy, x)
        med_d2y = np.median(d2y)
        mad_d2y = np.median(np.abs(d2y - med_d2y))
        if mad_d2y > 0:
            sigma_d2y = 0.165325 * mad_d2y
            z_d2y = np.abs(d2y - med_d2y) / sigma_d2y
            for i in np.where(z_d2y > self.threshold)[0]:
                if i in region_idx:
                    glitch_indices.add(i)

        indices = sorted(list(glitch_indices))
        x_values = x[indices]
        return indices, x_values

    def correct_glitches(self, x, y, glitch_indices):
        x = np.asarray(x); y = np.asarray(y, float)
        mask = np.ones_like(y, dtype=bool)
        mask[np.asarray(glitch_indices, int)] = False
    
        # Need at least two valid points
        if mask.sum() < 2:
            return y.copy()
    
        f = PchipInterpolator(x[mask], y[mask], extrapolate=False)
        y_corr = y.copy()
        y_corr[~mask] = f(x[~mask])
    
        # Optional: For leading/trailing all-glitch blocks, fallback to nearest
        if np.any(~np.isfinite(y_corr)):
            # fill NaNs by nearest valid
            valid = np.isfinite(y_corr)
            y_corr[~valid] = np.interp(x[~valid], x[valid], y_corr[valid])
        return y_corr

    # def correct_glitches(self, x, y, glitch_indices):
    #     y = np.asarray(y, dtype=float)
    #     x = np.asarray(x, dtype=float)
    #     y_corr = y.copy()
    #     n = len(y_corr)
    #     glitch_indices = set(glitch_indices)

    #     for idx in glitch_indices:
    #         left = idx - 1
    #         right = idx + 1

    #         while left in glitch_indices and left > 0:
    #             left -= 1
    #         while right in glitch_indices and right < n - 1:
    #             right += 1

    #         if 0 <= left < n and 0 <= right < n and right != left:
    #             # linear interpolation across nearest non-glitch neighbors
    #             y_corr[idx] = y_corr[left] + ((x[idx] - x[left]) / (x[right] - x[left])) * (y_corr[right] - y_corr[left])
    #     return y_corr

    def process(self, x, y, region=None, x_range=None, E0=None, return_indices=False):
        idxs, _ = self.detect_glitches(x, y, region=region, x_range=x_range, E0=E0)
        corrected = self.correct_glitches(x, y, idxs)
        if return_indices:
            return corrected, idxs
        return corrected
