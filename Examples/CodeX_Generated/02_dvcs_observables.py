"""Tutorial 2: evaluate DVCS CFFs, cross sections, and asymmetries."""

import numpy as np
import pandas as pd

from gumpgpd.DVCS_xsec import dsigma_BH, dsigma_DVCS, dsigma_INT
from gumpgpd.Minimizer import (
    CFF_theo,
    DVCSAsym_theo,
    DVCSxsecHERA_theo,
    DVCSxsec_theo,
    Para_Pol_off_forward,
    Para_Unp_off_forward,
    close_pool,
)


P_ORDER = 2  # 1 = LO, 2 = NLO


def predict(function, points):
    """Evaluate a DVCS DataFrame with the packaged best-fit parameters."""
    return function(
        points, Para_Unp=Para_Unp_off_forward, Para_Pol=Para_Pol_off_forward,
        P_order=P_ORDER,  # note the capital P in the DVCS wrappers
    )


def main() -> None:
    # A DVCS amplitude depends on four complex Compton form factors (CFFs).
    # GUMP derives xi internally from xB, t, and Q. Para_Unp contains H/E;
    # Para_Pol contains Htilde/Etilde.
    xB, t, Q = 0.20, -0.30, 2.5
    cff_names = ("H", "E", "Htilde", "Etilde")
    cffs = CFF_theo(xB, t, Q, Para_Unp_off_forward, Para_Pol_off_forward, porder=P_ORDER)

    print("CFFs at xB=0.20, t=-0.30 GeV^2, Q=2.5 GeV")
    for name, value in zip(cff_names, cffs):
        print(f"  {name:6s} = {np.asarray(value).squeeze()}")

    # The lower-level cross-section functions expose the three terms whose
    # sum is returned by DVCSxsec_theo. This is useful for diagnosing which
    # mechanism dominates a particular point.
    y, phi, pol = 0.50, np.pi / 2, "UU"
    cffs = [np.asarray(value).squeeze() for value in cffs]
    kinematics = (y, xB, t, Q, phi, pol)
    components = {
        "Bethe-Heitler": dsigma_BH(*kinematics),
        "pure DVCS": dsigma_DVCS(*kinematics, *cffs),
        "interference": dsigma_INT(*kinematics, *cffs),
    }
    print("\nCross-section decomposition at phi=90 degrees")
    for name, value in components.items():
        print(f"  {name:14s} = {np.asarray(value).squeeze():.8g}")
    print(f"  {'total':14s} = {sum(np.asarray(v).squeeze() for v in components.values()):.8g}")

    # The lepton-proton interface computes BH + pure DVCS + interference.
    # phi is in radians. Rows with the same (xB,t,Q) share one CFF evaluation,
    # so a phi scan is naturally represented by multiple DataFrame rows.
    dvcs_points = pd.DataFrame(
        {
            "y": [0.50, 0.50, 0.50, 0.50],
            "xB": xB,
            "t": t,
            "Q": Q,
            "phi": np.deg2rad([0.0, 90.0, 180.0, 270.0]),
            "pol": "UU",
        }
    )
    dvcs_result = predict(DVCSxsec_theo, dvcs_points)
    dvcs_result["phi_deg"] = np.rad2deg(dvcs_result["phi"])
    print("\nUnpolarized lepton-proton DVCS cross section")
    print(dvcs_result[["phi_deg", "y", "xB", "t", "Q", "pol", "pred f"]].to_string(index=False))

    # Spin asymmetries are the selected polarized cross-section component
    # divided by UU at the same point. Here LU is the beam-spin asymmetry.
    asymmetry_points = dvcs_points.assign(pol="LU")
    asymmetry_result = predict(DVCSAsym_theo, asymmetry_points)
    asymmetry_result["phi_deg"] = np.rad2deg(asymmetry_result["phi"])
    print("\nLU beam-spin asymmetry (dimensionless)")
    print(asymmetry_result[["phi_deg", "pol", "pred f"]].to_string(index=False))

    # HERA reports the virtual-photon-proton d sigma/dt with the lepton flux
    # removed and phi integrated. Consequently this interface has no phi.
    hera_points = pd.DataFrame([[0.30, 1.0e-3, -0.20, 4.0, "UU"]], columns=["y", "xB", "t", "Q", "pol"])
    hera_result = predict(DVCSxsecHERA_theo, hera_points)
    print("\nHERA virtual-photon-proton DVCS d sigma/dt")
    print(hera_result.to_string(index=False))


if __name__ == "__main__":
    main()
    close_pool()
