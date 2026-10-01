"""
Module-specific capacity and yield for R3-2.

Implements the equations of Revision Plan section 2.1, which replace the draft's
Eq. (5)-(6) where the power density ``P_D`` was never given a value::

    N_m   = floor( k * A_usable / (L_m * W_m) )
    P_m   = N_m * Pmax_STC,m / 1000                                   [kWp]
    E_m   = P_m * H_POA * (1 + gamma_m * dT) * (1 - L_sys)            [kWh/yr]
    E_m(y)= E_m * (1 - d1_m) * (1 - d_m)^(y-1)

``A_usable`` is always a required argument and is never hard-coded. R1-3 (inner
buffering from the SAM3 polygons) will change it, and the revision plan explicitly
asks for this script to take it as an input rather than baking in 6,256.6 m2.

Two cautions carried over from the revision plan:

* ``L_sys`` covers non-temperature losses only (soiling, mismatch, wiring, inverter).
  Do NOT also multiply by a literature performance ratio that already includes
  temperature loss, or the temperature penalty is counted twice.
* ``dT`` is the irradiance-weighted cell temperature *above STC* (i.e. T_cell - 25),
  not the rise above ambient. The plan's worked example uses 27 degC, which reproduces
  its printed temperature losses of -7.0 / -7.8 / -6.5 % for gamma of
  -0.26 / -0.29 / -0.24 %/degC.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

import pandas as pd

from .pv_modules import BinPolicy, PVModule, list_modules

try:  # pragma: no cover - optional dependency
    import pvlib  # noqa: F401

    HAS_PVLIB = True
except ImportError:  # pragma: no cover
    HAS_PVLIB = False


#: Objectives from Revision Plan section 2.2. "Optimum" is meaningless until one of
#: these is named, because a different module wins under each.
OBJECTIVES: Dict[str, str] = {
    "per_area": "Maximum energy per roof area (area-limited urban roofs; efficiency dominates)",
    "per_kwp": "Maximum energy per kWp (hot climates; temperature coefficient dominates)",
    "self_consumed": (
        "Maximum self-consumed energy (Permen ESDM 2/2024 gives no credit for export, "
        "so the economic optimum is sized to demand, not to the roof)"
    ),
}


@dataclass(frozen=True)
class YieldAssumptions:
    """Every non-datasheet number in the R3-2 calculation, in one place.

    All fields are surfaced as dashboard widgets so a reviewer can see, and a reader
    can reproduce, exactly what was assumed.
    """

    #: Packing factor: module area divided by usable area. Sensitivity range in the
    #: revision plan's work plan is 0.75-0.90.
    packing_factor: float = 0.85

    #: Annual plane-of-array irradiation, kWh/m2/yr. The default matches the
    #: 1,642 kWh/m2 already used elsewhere in this codebase (MAGELANG_SOLAR_DATA).
    h_poa_kwh_m2_yr: float = 1642.0

    #: Irradiance-weighted cell temperature above STC (T_cell - 25), degC.
    delta_t_cell_c: float = 27.0

    #: Non-temperature system losses: soiling, mismatch, wiring, inverter.
    system_losses: float = 0.14

    #: Applied uniformly to every manufacturer (Revision Plan section 2.3).
    bin_policy: BinPolicy = "top"

    lifetime_years: int = 25

    #: Multiplier on H_POA from the R3-3 slope/orientation work. Left at 1.0 for the
    #: paper's headline numbers, because R3-3 is scoped as "Limitation only".
    orientation_factor: float = 1.0

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def h_poa_from_psh(peak_sun_hours: float) -> float:
    """Convert the peak-sun-hours convention used elsewhere here to kWh/m2/yr."""
    return peak_sun_hours * 365.0


# --------------------------------------------------------------------------- core

def n_modules(a_usable_m2: float, module: PVModule, packing_factor: float = 0.85) -> int:
    """Whole modules that fit on the usable area.

    This is the fallback when a real panel layout is unavailable. If R3-1 (Raffi's
    panel-layout item) produces an actual per-roof module count, pass that count
    straight to :func:`capacity_kwp` instead -- it is strictly better than a packing
    factor.
    """
    if a_usable_m2 < 0:
        raise ValueError("a_usable_m2 must be non-negative")
    if not 0 < packing_factor <= 1:
        raise ValueError("packing_factor must be in (0, 1]")
    return int(math.floor(packing_factor * a_usable_m2 / module.area_m2))


def capacity_kwp(n: int, module: PVModule, bin_policy: BinPolicy = "top") -> float:
    """Installed DC capacity, kWp."""
    return n * module.pmax_w(bin_policy) / 1000.0


def temperature_factor(gamma_pmax: float, delta_t_cell_c: float) -> float:
    """``1 + gamma * dT``: the fraction of STC output left after thermal derating."""
    return 1.0 + (gamma_pmax / 100.0) * delta_t_cell_c


def annual_energy_kwh(
    kwp: float,
    h_poa_kwh_m2_yr: float,
    gamma_pmax: float,
    delta_t_cell_c: float,
    system_losses: float,
) -> float:
    """First-year AC energy, kWh/yr.

    Closed annual form of the plan's hourly sum. It is exact when ``delta_t_cell_c``
    is the irradiance-weighted mean, which is why the parameter is defined that way
    rather than as a simple average temperature.
    """
    if not 0 <= system_losses < 1:
        raise ValueError("system_losses must be in [0, 1)")
    # Units: kWp * (kWh/m2/yr) / (1 kW/m2 at STC) = kWh/yr. The STC reference
    # irradiance is 1 kW/m2, so it divides out and leaves kwp * H_POA.
    return (
        kwp
        * h_poa_kwh_m2_yr
        * temperature_factor(gamma_pmax, delta_t_cell_c)
        * (1.0 - system_losses)
    )


def lifetime_energy_kwh(e_year1_kwh: float, module: PVModule, years: int = 25) -> float:
    """Total energy over ``years``, applying the datasheet degradation curve."""
    return sum(e_year1_kwh * module.retention_at_year(y) for y in range(1, years + 1))


def annual_energy_series(e_year1_kwh: float, module: PVModule, years: int = 25) -> List[float]:
    """Per-year energy, for plotting the degradation curve."""
    return [e_year1_kwh * module.retention_at_year(y) for y in range(1, years + 1)]


# ---------------------------------------------------------------------- evaluation

def evaluate_module(
    a_usable_m2: float,
    module: PVModule,
    assumptions: Optional[YieldAssumptions] = None,
    annual_demand_kwh: Optional[float] = None,
) -> Dict[str, object]:
    """Full R3-2 result for one module on one roof area."""
    a = assumptions or YieldAssumptions()
    n = n_modules(a_usable_m2, module, a.packing_factor)
    kwp = capacity_kwp(n, module, a.bin_policy)
    h_poa = a.h_poa_kwh_m2_yr * a.orientation_factor

    e1 = annual_energy_kwh(
        kwp, h_poa, module.gamma_pmax, a.delta_t_cell_c, a.system_losses
    )
    e_life = lifetime_energy_kwh(e1, module, a.lifetime_years)

    row: Dict[str, object] = {
        "code": module.code,
        "manufacturer": module.manufacturer,
        "model": module.model,
        "cell_tech": module.cell_tech,
        "pmax_w": module.pmax_w(a.bin_policy),
        "area_m2": round(module.area_m2, 4),
        "eta_pct": round(module.eta_pct(a.bin_policy), 2),
        "gamma_pmax": module.gamma_pmax,
        "n_modules": n,
        "capacity_kwp": round(kwp, 1),
        "kwp_per_m2": round(kwp / a_usable_m2, 4) if a_usable_m2 else 0.0,
        "annual_kwh": round(e1, 0),
        "annual_mwh": round(e1 / 1000.0, 1),
        "kwh_per_kwp": round(e1 / kwp, 0) if kwp else 0.0,
        "kwh_per_m2": round(e1 / a_usable_m2, 1) if a_usable_m2 else 0.0,
        "temp_loss_pct": round((temperature_factor(module.gamma_pmax, a.delta_t_cell_c) - 1) * 100, 1),
        "lifetime_mwh": round(e_life / 1000.0, 1),
        "source_status": module.source_status,
    }

    if annual_demand_kwh:
        self_consumed = min(e1, annual_demand_kwh)
        row["self_consumed_kwh"] = round(self_consumed, 0)
        # Roof area this module needs to cover the demand -- the quantity that matters
        # once export earns nothing (Permen ESDM 2/2024, Art. 13).
        row["area_for_demand_m2"] = (
            round(a_usable_m2 * annual_demand_kwh / e1, 1) if e1 > 0 else float("inf")
        )
        row["demand_met_pct"] = round(100.0 * self_consumed / annual_demand_kwh, 1)

    return row


def compare_modules(
    a_usable_m2: float,
    codes: Optional[Sequence[str]] = None,
    assumptions: Optional[YieldAssumptions] = None,
    annual_demand_kwh: Optional[float] = None,
) -> pd.DataFrame:
    """The IV.B module comparison table.

    Modules whose datasheet does not support the requested bin policy are skipped
    rather than silently substituted, so the bin policy stays uniform.
    """
    a = assumptions or YieldAssumptions()
    rows = []
    for module in list_modules(list(codes) if codes else None):
        if not module.bin_available(a.bin_policy):
            continue
        rows.append(evaluate_module(a_usable_m2, module, a, annual_demand_kwh))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("capacity_kwp", ascending=False).reset_index(drop=True)


def skipped_for_bin(
    codes: Optional[Sequence[str]] = None, bin_policy: BinPolicy = "top"
) -> List[str]:
    """Modules excluded from :func:`compare_modules` because of the bin policy."""
    return [
        m.code
        for m in list_modules(list(codes) if codes else None)
        if not m.bin_available(bin_policy)
    ]


def best_module(df: pd.DataFrame, objective: str = "per_area") -> Optional[pd.Series]:
    """Arg-max over the comparison table for one of :data:`OBJECTIVES`.

    This is a deterministic enumeration over a handful of fully specified options, so
    it finds the exact optimum and stays traceable to datasheets. That is the answer
    to Reviewer 3's suggestion that an AI approach be used here -- see Revision Plan
    section 7; AI becomes useful in R3-4, for inferring unknown roof geometry.
    """
    if df.empty:
        return None
    if objective not in OBJECTIVES:
        raise ValueError("objective must be one of %s" % sorted(OBJECTIVES))
    if objective == "per_area":
        return df.loc[df["kwh_per_m2"].idxmax()]
    if objective == "per_kwp":
        return df.loc[df["kwh_per_kwp"].idxmax()]
    if "area_for_demand_m2" not in df.columns:
        raise ValueError(
            "objective 'self_consumed' needs annual_demand_kwh passed to compare_modules()"
        )
    return df.loc[df["area_for_demand_m2"].idxmin()]


def best_modules(df: pd.DataFrame, annual_demand_kwh: Optional[float] = None) -> Dict[str, object]:
    """All three answers at once, for the IV.B paragraph."""
    out: Dict[str, object] = {}
    for key in OBJECTIVES:
        if key == "self_consumed" and not annual_demand_kwh:
            continue
        row = best_module(df, key)
        if row is not None:
            out[key] = {"code": row["code"], "model": row["model"], "why": OBJECTIVES[key]}
    return out


def spread_summary(df: pd.DataFrame) -> Dict[str, float]:
    """How much module choice actually moves the answer.

    This is the sentence the IV.B paragraph needs: without it the table reads as a
    catalogue rather than a result.
    """
    if df.empty:
        return {}
    cap = df["capacity_kwp"]
    yld = df["kwh_per_kwp"]
    temp = df["temp_loss_pct"]
    return {
        "capacity_spread_pct": round((cap.max() / cap.min() - 1) * 100, 1),
        "capacity_min_kwp": float(cap.min()),
        "capacity_max_kwp": float(cap.max()),
        "yield_spread_pct": round((yld.max() / yld.min() - 1) * 100, 1),
        "temp_loss_spread_pp": round(float(temp.max() - temp.min()), 1),
    }


def packing_sensitivity(
    a_usable_m2: float,
    codes: Optional[Sequence[str]] = None,
    assumptions: Optional[YieldAssumptions] = None,
    packing_factors: Sequence[float] = (0.75, 0.80, 0.85, 0.90),
) -> pd.DataFrame:
    """Capacity vs packing factor -- step 5 of the revision plan's work plan."""
    base = assumptions or YieldAssumptions()
    rows = []
    for k in packing_factors:
        a = YieldAssumptions(**{**base.as_dict(), "packing_factor": k})
        df = compare_modules(a_usable_m2, codes, a)
        if df.empty:
            continue
        rows.append(
            {
                "packing_factor": k,
                "min_kwp": float(df["capacity_kwp"].min()),
                "max_kwp": float(df["capacity_kwp"].max()),
                "best_module": df.loc[df["capacity_kwp"].idxmax(), "code"],
            }
        )
    return pd.DataFrame(rows)


def power_density_kwp_m2(
    module: PVModule, bin_policy: BinPolicy = "top", packing_factor: float = 0.85
) -> float:
    """Traceable replacement for the draft's undefined ``P_D`` in Eq. (5).

    The repo previously carried three different, unsourced values at once
    (0.15 kWp/m2, 0.20 kWp/m2 and 200 W/m2). This derives one number from a named
    datasheet instead: module power over module area, derated by the packing factor.
    """
    return module.pmax_w(bin_policy) / 1000.0 / module.area_m2 * packing_factor
