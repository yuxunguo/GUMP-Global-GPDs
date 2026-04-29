import numpy as np
from typing import Tuple, Union
from scipy.interpolate import PchipInterpolator
from functools import lru_cache
import time

from eko.couplings import Couplings
from eko.quantities.couplings import CouplingEvolutionMethod, CouplingsInfo
from eko.quantities.heavy_quarks import QuarkMassScheme
from ekore.anomalous_dimensions.unpolarized import time_like as tl
from ekore.anomalous_dimensions.unpolarized import space_like as sl


from . import config

_ALPHAS_TABLE_Q_MIN = 0.6
_ALPHAS_TABLE_Q_MAX = 1.0e5
_ALPHAS_TABLE_NPTS = 2048

@config.Hybrid_Cache
def _alpha_s_interp_table_cached(table_spec: Tuple[float, float, int],
                                 alphas_ref: float = 0.118, ref_scale: float = 91.2,
                                 nf_ref: int = 5,
                                 heavy_quark_masses: Tuple[float, float, float] = (1.275, 4.118, 173.2),
                                 order: Tuple[int, int] = (3, 1)):
    """Build and cache the alpha_s interpolation table on disk."""
    q_min, q_max, npts = table_spec
    heavy_quark_masses_sq = np.power(heavy_quark_masses, 2)
    thresholds_ratios = np.ones_like(heavy_quark_masses_sq)
    couplings_ref = CouplingsInfo(alphas=alphas_ref, alphaem=0.00784, ref=(ref_scale, nf_ref))

    sc = Couplings(
        couplings_ref,
        order,
        CouplingEvolutionMethod.EXACT,
        heavy_quark_masses_sq,
        QuarkMassScheme.POLE,
        thresholds_ratios
    )

    q_grid = np.geomspace(q_min, q_max, npts)
    alpha_grid = np.array([4.0 * np.pi * sc.a_s(q * q) for q in q_grid])
    return np.log(q_grid), alpha_grid

def _raise_if_q_out_of_interp_range(target_scales_arr: np.ndarray):
    in_range = (target_scales_arr >= _ALPHAS_TABLE_Q_MIN) & (target_scales_arr <= _ALPHAS_TABLE_Q_MAX)
    if np.any(~in_range):
        q_out = target_scales_arr[~in_range]
        raise ValueError(
            "target_scales out of interpolation range "
            f"[{_ALPHAS_TABLE_Q_MIN}, {_ALPHAS_TABLE_Q_MAX}] GeV. "
            f"Received min={q_out.min()} GeV, max={q_out.max()} GeV."
        )

def _alpha_s_interpolator(alphas_ref: float = 0.118, ref_scale: float = 91.2,
                          nf_ref: int = 5,
                          heavy_quark_masses: Tuple[float, float, float] = (1.275, 4.118, 173.2),
                          order: Tuple[int, int] = (3, 1)):
    return _alpha_s_interpolator_cached(
        float(alphas_ref),
        float(ref_scale),
        int(nf_ref),
        tuple(np.asarray(heavy_quark_masses, dtype=float).tolist()),
        tuple(order),
    )

@lru_cache(maxsize=32)
def _alpha_s_interpolator_cached(alphas_ref: float, ref_scale: float, nf_ref: int,
                                 heavy_quark_masses: Tuple[float, float, float],
                                 order: Tuple[int, int]):
    table_spec = (_ALPHAS_TABLE_Q_MIN, _ALPHAS_TABLE_Q_MAX, _ALPHAS_TABLE_NPTS)
    log_q_grid, alpha_grid = _alpha_s_interp_table_cached(
        table_spec,
        alphas_ref,
        ref_scale,
        nf_ref,
        heavy_quark_masses,
        order,
    )
    return PchipInterpolator(log_q_grid, alpha_grid, extrapolate=False)

def Alpha_S_Wrapped(target_scales, alphas_ref=0.118, ref_scale=91.2, nf_ref=5,
                heavy_quark_masses=(1.275, 4.118, 173.2), order=(3,1)):
    """
    Compute alpha_s at given scales using an interpolated table.

    Parameters
    ----------
    target_scales : float or array-like
        Energy scale(s) in GeV (can be a single value or a numpy array).
    alphas_ref : float
        Reference alpha_s value at ref_scale (default 0.118 at MZ).
    ref_scale : float
        Reference scale in GeV (default 91.0).
    nf_ref : int
        Number of active flavors at reference scale (default 5).
    heavy_quark_masses : tuple of floats
        Heavy quark masses in GeV (c, b, t).
    order : tuple
        Perturbative order (QCD loops, QED loops), default (3,1).

    Returns
    -------
    alpha_s_values : float or np.ndarray
        alpha_s at target scales.
    """
    target_scales_arr = np.asarray(np.atleast_1d(target_scales), dtype=float)
    if np.any(target_scales_arr <= 0.0):
        raise ValueError("target_scales must be positive.")

    _raise_if_q_out_of_interp_range(target_scales_arr)

    interp = _alpha_s_interpolator(
        alphas_ref=float(alphas_ref),
        ref_scale=float(ref_scale),
        nf_ref=int(nf_ref),
        heavy_quark_masses=tuple(np.asarray(heavy_quark_masses, dtype=float).tolist()),
        order=tuple(order),
    )

    alpha_s_vals = interp(np.log(target_scales_arr))

    # Return scalar if input was scalar
    if alpha_s_vals.size == 1:
        return alpha_s_vals[0]
    return alpha_s_vals


def benchmark_alpha_s_against_evolution(target_scales=None, nf: int = 4, nloop: int = 2,
                                        repeats: int = 5, warmup: bool = True):
    """Benchmark alpha_s implementation here against Evolution.AlphaS.

    The benchmark compares both runtime and numerical agreement on the same Q grid.
    For a fair comparison, this function aligns the wrapped alpha_s reference point
    to Evolution's internal (Alpha_Ref, Ref_Scale).
    """
    from .Evolution import AlphaS as EvolutionAlphaS, Alpha_Ref as EvolutionAlphaRef, Ref_Scale as EvolutionRefScale

    if target_scales is None:
        # Keep scales inside interpolation range and in typical evolution region.
        q = np.geomspace(1.0, 100.0, 300)
    else:
        q = np.asarray(np.atleast_1d(target_scales), dtype=float)

    if np.any(q <= 0):
        raise ValueError("All target scales for benchmark must be positive.")

    # AdimPlus table range guard.
    _raise_if_q_out_of_interp_range(q)

    def wrapped_call():
        return np.asarray(
            Alpha_S_Wrapped(
                q,
                alphas_ref=EvolutionAlphaRef,
                ref_scale=EvolutionRefScale,
                nf_ref=nf,
                order=(nloop, 1),
            )
        )

    def evolution_call():
        return np.asarray(EvolutionAlphaS(nloop, nf, q))

    if warmup:
        # Warmup avoids one-time compilation/setup cost in timing comparison.
        _ = wrapped_call()
        _ = evolution_call()

    def best_runtime_seconds(fn):
        best = float("inf")
        for _ in range(max(1, repeats)):
            t0 = time.perf_counter()
            _ = fn()
            dt = time.perf_counter() - t0
            if dt < best:
                best = dt
        return best

    wrapped_vals = wrapped_call()
    evolution_vals = evolution_call()

    abs_diff = np.abs(wrapped_vals - evolution_vals)
    denom = np.maximum(np.abs(evolution_vals), 1e-14)
    rel_diff = abs_diff / denom

    wrapped_time = best_runtime_seconds(wrapped_call)
    evolution_time = best_runtime_seconds(evolution_call)

    return {
        "n_points": int(q.size),
        "q_min": float(np.min(q)),
        "q_max": float(np.max(q)),
        "nf": int(nf),
        "nloop": int(nloop),
        "wrapped_time_s": float(wrapped_time),
        "evolution_time_s": float(evolution_time),
        "speed_ratio_wrapped_over_evolution": float(wrapped_time / max(evolution_time, 1e-15)),
        "max_abs_diff": float(np.max(abs_diff)),
        "mean_abs_diff": float(np.mean(abs_diff)),
        "max_rel_diff": float(np.max(rel_diff)),
        "mean_rel_diff": float(np.mean(rel_diff)),
    }


if __name__ == '__main__':
    stats = benchmark_alpha_s_against_evolution()
    print("alpha_s benchmark (AdimPlus vs Evolution)")
    print(f"  points: {stats['n_points']}  Q-range: [{stats['q_min']:.3g}, {stats['q_max']:.3g}] GeV")
    print(f"  nf={stats['nf']}  nloop={stats['nloop']}")
    print(f"  time wrapped   : {stats['wrapped_time_s']:.6f} s")
    print(f"  time evolution : {stats['evolution_time_s']:.6f} s")
    print(f"  speed ratio (wrapped/evolution): {stats['speed_ratio_wrapped_over_evolution']:.3f}")
    print(f"  max abs diff   : {stats['max_abs_diff']:.6e}")
    print(f"  mean abs diff  : {stats['mean_abs_diff']:.6e}")
    print(f"  max rel diff   : {stats['max_rel_diff']:.6e}")
    print(f"  mean rel diff  : {stats['mean_rel_diff']:.6e}")
