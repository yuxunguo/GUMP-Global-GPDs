"""Tutorial 3: evaluate DVMP TFFs and rho/J/psi cross sections."""

import numpy as np
import pandas as pd

from gumpgpd.DVMP_xsec import R_fitted
from gumpgpd.Minimizer import (
    DVMPxsec_theo,
    Para_Unp_off_forward,
    TFF_theo,
    close_pool,
    get_pool,
)


P_ORDER = 2  # 1 = LO, 2 = NLO
MESON_NAMES = {1: "rho0", 3: "J/psi"}


def print_tffs(xB: float, t: float, Q: float, meson: int) -> None:
    """Evaluate the complex H- and E-type transition form factors."""
    h_tff, e_tff = TFF_theo(
        xB,
        t,
        Q,
        Para_Unp=Para_Unp_off_forward,
        meson=meson,
        p_order=P_ORDER,
        muset=1.0,  # factorization scale mu = muset * Q
        flv="All",  # use quark and gluon channels
    )
    print(f"\n{MESON_NAMES[meson]} TFFs at xB={xB:g}, t={t:g}, Q={Q:g}")
    print(f"  H = {np.asarray(h_tff).squeeze()}")
    print(f"  E = {np.asarray(e_tff).squeeze()}")


def cross_section(xB: float, meson: int) -> pd.DataFrame:
    """Evaluate two t points for one meson."""
    points = pd.DataFrame({"y": 0.30, "xB": xB, "t": [-0.10, -0.30], "Q": 3.0})
    return DVMPxsec_theo(
        points, Para_Unp=Para_Unp_off_forward, xsec_norm=1.0,
        meson=meson, p_order=P_ORDER,
    )


def main() -> None:
    # The default pool uses the available CPU processes for parallel evaluation.
    get_pool()

    # TFF access is available for rho0 and J/psi. `flv` can also be changed
    # to "q" (quarks only) or "g" (gluon only) for channel studies.
    print_tffs(xB=1.0e-2, t=-0.20, Q=3.0, meson=1)
    print_tffs(xB=1.0e-3, t=-0.20, Q=3.0, meson=3)

    # rho electroproduction data are often quoted without an L/T separation.
    # R = sigma_L/sigma_T connects the separated and total cross sections.
    r_mean, r_std = R_fitted(Q=3.0, meson=1)
    print(f"\nrho0 R = sigma_L/sigma_T at Q=3 GeV: {r_mean:.6g} +/- {r_std:.3g}")

    # For rho0, the high-level wrapper returns the longitudinal virtual-
    # photon-proton cross section d sigma_L/dt in nb/GeV^2.
    rho_result = cross_section(xB=1.0e-2, meson=1)
    print("\nrho0 longitudinal d sigma_L/dt [nb/GeV^2]")
    print(rho_result.to_string(index=False))

    # For J/psi, the wrapper returns the total d sigma/dt. The meson mass is
    # included in both the skewness and hard-scattering kinematics.
    jpsi_result = cross_section(xB=1.0e-3, meson=3)
    print("\nJ/psi total d sigma/dt [nb/GeV^2]")
    print(jpsi_result.to_string(index=False))

    # xsec_norm rescales the amplitude, so the wrapper multiplies a cross
    # section by xsec_norm**2. Meson code 2 is reserved for phi production,
    # but its charge factor and high-level cross-section branch are currently
    # not implemented.


if __name__ == "__main__":
    main()
    close_pool()
