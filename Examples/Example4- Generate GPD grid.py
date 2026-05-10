# ===========================================================================
# Generate polarized GPDs: H~u, H~d, E~u, E~d on a custom (x, xi, t, Q) grid
# ===========================================================================

from pathlib import Path
import time

import numpy as np
import pandas as pd

from gumpgpd.Minimizer import GPD_theo, tPDF_theo, Para_Comb_off_forward


def build_x_grid_for_xi(
    xi: float,
    xmax: float = 0.999,
    n_outer: int = 40,
    n_inner: int = 20,
    min_pos: float = 1e-6,
) -> np.ndarray:
    """Build x grid for one xi.

    Requirements implemented here:
    - Explicitly include x = xi.
    - Log scale in regions x > xi and 0 < x < xi.
    - Mirror to negative x.
    """
    xi = float(xi)
    if not (0.0 <= xi < 1.0):
        raise ValueError('xi should satisfy 0 <= xi < 1')

    # Region 0 < x < xi (empty for xi = 0).
    if xi > min_pos:
        upper_inner = xi * (1.0 - 1e-12)
        x_inner = np.logspace(np.log10(min_pos), np.log10(upper_inner), num=n_inner)
    else:
        x_inner = np.array([], dtype=float)

    # Region x > xi.
    lower_outer = max(min_pos, xi * (1.0 + 1e-12))
    x_outer = np.logspace(np.log10(lower_outer), np.log10(xmax), num=n_outer)

    # Explicitly include x = xi.
    x_pos = np.unique(np.concatenate([x_inner, np.array([xi], dtype=float), x_outer]))

    # Mirror positive side to negative side, keep zero once.
    x_all = np.unique(np.concatenate([-x_pos[::-1], x_pos]))
    return x_all


if __name__ == '__main__':

    # Requested xi values.
    xiarr = np.array([
        0.012, 0.014, 0.016, 0.018, 0.020,
        0.025, 0.030, 0.035, 0.040, 0.050,
        0.060, 0.080, 0.100, 0.150, 0.200,
        0.250, 0.300
    ])

    # Requested fixed t and Q values.
    # Keep original Q points and include extra points converted from Q^2 = 6, 10, 14.
    t_fixed = -0.1
    q_base = np.array([1.0, 2.0, 4.0])
    q2_extra = np.array([6.0, 10.0, 14.0])
    qarr = np.unique(np.concatenate([q_base, np.sqrt(q2_extra)]))

    # spe mapping in this package: 2 -> H~ and 3 -> E~.
    components = [
        (2, 'u', 'Htilde_u'),
        (2, 'd', 'Htilde_d'),
        (3, 'u', 'Etilde_u'),
        (3, 'd', 'Etilde_d'),
    ]

    frames = []
    for spe, flv, comp in components:
        for xi in xiarr:
            xarr = build_x_grid_for_xi(xi)
            for q in qarr:
                n = xarr.size
                frames.append(
                    pd.DataFrame(
                        {
                            'x': xarr,
                            'xi': np.full(n, xi),
                            't': np.full(n, t_fixed),
                            'Q': np.full(n, q),
                            'spe': np.full(n, spe, dtype=int),
                            'flv': np.full(n, flv),
                            'component': np.full(n, comp),
                        }
                    )
                )

    gpd_input = pd.concat(frames, ignore_index=True)
    gpd_input['_row_id'] = np.arange(len(gpd_input), dtype=int)

    t0 = time.perf_counter()

    # Use tPDF at xi=0 and GPD for xi>0.
    mask_xi0 = np.isclose(gpd_input['xi'].to_numpy(), 0.0)
    result_parts = []

    if (~mask_xi0).any():
        gpd_nonzero = gpd_input.loc[~mask_xi0, ['_row_id', 'x', 'xi', 't', 'Q', 'spe', 'flv']]
        result_parts.append(GPD_theo(gpd_nonzero, Para=Para_Comb_off_forward))

    if mask_xi0.any():
        tpdf_input = gpd_input.loc[mask_xi0, ['_row_id', 'x', 't', 'Q', 'spe', 'flv']]
        result_xi0 = tPDF_theo(tpdf_input, Para=Para_Comb_off_forward)
        result_xi0['xi'] = 0.0
        result_parts.append(result_xi0[['_row_id', 'x', 'xi', 't', 'Q', 'spe', 'flv', 'pred f']])

    result = pd.concat(result_parts, ignore_index=True).sort_values('_row_id').reset_index(drop=True)
    t1 = time.perf_counter()

    # Restore component labels and reorder output columns.
    result['component'] = gpd_input.loc[result['_row_id'], 'component'].values
    result = result[['component', 'x', 'xi', 't', 'Q', 'spe', 'flv', 'pred f']]

    out_path = Path('GUMP_Results') / 'GPD_tilde_ud_x_xi_extra_t-0p1_Q1_2_4_plusQ2_6_10_14.csv'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)

    print(f'Generated {len(result)} rows.')
    print(f'Elapsed time: {t1 - t0:.2f} s')
    print(f'Saved to: {out_path}')
