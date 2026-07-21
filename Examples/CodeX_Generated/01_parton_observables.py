"""Tutorial 1: evaluate PDFs, tPDFs, GPDs, and GFFs with GUMP.

Read Examples/CodeX_Generated/README.md first for the complete column and
selector reference.
The small tables below are meant to be edited: one row is one requested theory
point. Numerical evaluation can be slow the first time because GUMP performs
QCD evolution and inverse Mellin/Mellin-Barnes integrals.
"""

import pandas as pd

from gumpgpd.Minimizer import (
    GFF_theo,
    GPD_theo,
    PDF_theo,
    Para_Comb_off_forward,
    close_pool,
    get_pool,
    tPDF_theo,
)


# 1 means LO and 2 means NLO. The packaged fit is an NLO fit.
P_ORDER = 2


def show(title: str, result: pd.DataFrame) -> None:
    """Print the requested coordinates beside GUMP's `pred f` result."""
    print(f"\n{title}\n{'-' * len(title)}")
    print(result.to_string(index=False))


def evaluate(title, function, columns, rows) -> None:
    """Build a point table, evaluate it, and print the result."""
    points = pd.DataFrame(rows, columns=columns)
    result = function(points, Para=Para_Comb_off_forward, p_order=P_ORDER)
    show(title, result)


def main() -> None:
    # The default pool uses the available CPU processes for parallel evaluation.
    get_pool()

    # ------------------------------------------------------------------
    # PDF: the xi=0, t=0 forward limit of a GPD.
    #
    # spe=0 requests H (an unpolarized PDF), while spe=2 requests Htilde
    # (a helicity PDF). flv may be u, d, g, S=u+d, or NS=u-d.
    # ------------------------------------------------------------------
    evaluate(
        "Forward PDFs", PDF_theo, ["x", "t", "Q", "spe", "flv"],
        [(0.10, 0.0, 2.0, 0, "u"), (0.10, 0.0, 2.0, 0, "g"), (0.10, 0.0, 2.0, 2, "NS")],
    )

    # ------------------------------------------------------------------
    # tPDF: the same interface, but at nonzero (normally negative) t.
    # E and Etilde do not have ordinary forward-PDF counterparts, but their
    # xi=0, t-dependent distributions can still be requested with spe=1/3.
    # ------------------------------------------------------------------
    evaluate(
        "t-dependent PDFs", tPDF_theo, ["x", "t", "Q", "spe", "flv"],
        [(0.20, -0.30, 2.0, spe, flv) for spe, flv in [(0, "u"), (1, "NS"), (2, "u"), (3, "NS")]],
    )

    # ------------------------------------------------------------------
    # GPD: adding xi moves away from the forward limit. The examples cover
    # all four leading-twist species. x may lie in the DGLAP (|x| > xi) or
    # ERBL (|x| < xi) region; these sample points use x > xi.
    # ------------------------------------------------------------------
    evaluate(
        "Generalized parton distributions", GPD_theo,
        ["x", "xi", "t", "Q", "spe", "flv"],
        [(0.30, 0.10, -0.30, 2.0, spe, flv) for spe, flv in [(0, "u"), (1, "NS"), (2, "u"), (3, "NS")]],
    )

    # ------------------------------------------------------------------
    # GFF: j selects a conformal/Mellin moment. No x or xi column is needed
    # because taking a moment has integrated out the momentum fraction.
    # ------------------------------------------------------------------
    evaluate(
        "Generalized form factors", GFF_theo, ["j", "t", "Q", "spe", "flv"],
        [(0, -0.20, 2.0, spe, "NS") for spe in range(4)],
    )

    # To compare with data, add `f` and `delta f` columns before calling a
    # wrapper. Its output will then also contain the pointwise chi^2 `cost`.


if __name__ == "__main__":
    main()
    close_pool()
