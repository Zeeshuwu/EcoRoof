"""
PV module catalogue for R3-2 (module-specific power output).

Reviewer 3 asked for "an additional study estimating the optimal power output using
specifications from several solar panel manufacturers". The draft's Eq. (5) is

    P = A_usable * P_D

and P_D is never given a value. This module replaces that undefined constant with a
datasheet-traceable catalogue, so every number downstream can be traced to a
manufacturer PDF.

Source of truth
---------------
``00_Review_TechSolarPanel_Spec.docx`` ("Hyatma's Work for Revision -> Manufacturer
Data Sheets"). Where the Revision Plan PDF section 3 disagrees, the docx wins and the
disagreement is recorded in a ``# CONFLICT:`` comment plus the module's ``notes``.

Fairness rules (Revision Plan section 2.3) enforced here:
  * one bin policy applied to every manufacturer (top / mid / bottom), never mixed
  * front-side STC ratings only (bifacial gain is ~0 on flush-mounted roofs)
  * efficiency is DERIVED from Pmax / module area, never copied from the datasheet
    headline, so it cannot drift from the dimensions used to pack the roof
  * NOCT and NMOT are different definitions and are NOT compared directly; the
    ``noct_kind`` field records which one the datasheet actually states
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

BinPolicy = str  # "top" | "mid" | "bottom"

VALID_BINS: Tuple[str, ...] = ("top", "mid", "bottom")


@dataclass(frozen=True)
class PVModule:
    """One commercially available module, as stated on its datasheet.

    Only datasheet facts are stored. Everything derivable (area, efficiency) is a
    property so it cannot fall out of sync with the dimensions.
    """

    code: str                       # F1, F1_ALT, F2 ... as used in the revision plan
    manufacturer: str
    model: str
    cell_tech: str

    pmax_max_w: float               # top of the power bin (W)
    pmax_min_w: Optional[float]     # bottom of the bin; None when the datasheet only
                                    # states "up to X W"

    length_mm: float
    width_mm: float
    thickness_mm: float

    eta_datasheet_pct: float        # headline "eta max" as printed, for cross-checking
    gamma_pmax: float               # temperature coefficient of Pmax, %/degC (negative)

    noct_c: Optional[float]         # None when the datasheet does not state one
    noct_kind: str                  # "noct" | "nmot" | "nmot_suspected" | "unstated"

    deg_year1_pct: Optional[float]  # first-year degradation, %
    deg_annual_pct: float           # year 2..30 degradation, %/yr

    source: str
    source_status: str              # "verified" | "verify"
    notes: str = ""

    # ---------------------------------------------------------------- geometry

    @property
    def area_m2(self) -> float:
        """Module area from datasheet dimensions."""
        return (self.length_mm / 1000.0) * (self.width_mm / 1000.0)

    # ---------------------------------------------------------------- ratings

    def bin_available(self, policy: BinPolicy = "top") -> bool:
        """Whether this module can honour the requested bin policy."""
        if policy == "top":
            return True
        return self.pmax_min_w is not None

    def pmax_w(self, policy: BinPolicy = "top") -> float:
        """Rated front-side STC power for the requested bin policy."""
        if policy not in VALID_BINS:
            raise ValueError("bin policy must be one of %s, got %r" % (VALID_BINS, policy))
        if policy == "top":
            return self.pmax_max_w
        if self.pmax_min_w is None:
            raise ValueError(
                "%s (%s) states only an upper bin (up to %g W), so the %r bin is "
                "unknown. Use the top bin policy or supply the full bin range."
                % (self.code, self.model, self.pmax_max_w, policy)
            )
        if policy == "bottom":
            return self.pmax_min_w
        return (self.pmax_min_w + self.pmax_max_w) / 2.0  # "mid"

    def eta_pct(self, policy: BinPolicy = "top") -> float:
        """Module efficiency DERIVED from Pmax and area (Revision Plan section 2.3)."""
        return self.pmax_w(policy) / (self.area_m2 * 1000.0) * 100.0

    def eta_discrepancy_pp(self, policy: BinPolicy = "top") -> float:
        """Derived efficiency minus the datasheet headline, in percentage points.

        A large value means the datasheet row mixes product variants and needs
        re-checking against the manufacturer PDF before it goes into the paper.
        """
        return self.eta_pct(policy) - self.eta_datasheet_pct

    # ---------------------------------------------------------------- ageing

    @property
    def deg_year1_effective_pct(self) -> float:
        """First-year degradation, falling back to the annual rate when unstated."""
        return self.deg_annual_pct if self.deg_year1_pct is None else self.deg_year1_pct

    def retention_at_year(self, year: int) -> float:
        """Fraction of nameplate output still available in ``year`` (1-based).

        Compound decay, matching Revision Plan section 2.1:
        ``E(y) = E * (1 - d1) * (1 - d)^(y-1)``.

        Note that manufacturers print *linear* warranty figures, so the two differ
        slightly: Trina's "0.4 %/yr, 87.4 % at yr 30" is 100 - 1 - 0.4*29 linear,
        whereas the compound form above gives 88.1 %. Compound is used here because
        the revision plan specifies it and because it is the convention in the
        degradation literature [A3, Jordan & Kurtz].
        """
        if year < 1:
            raise ValueError("year must be >= 1")
        return (1.0 - self.deg_year1_effective_pct / 100.0) * (
            1.0 - self.deg_annual_pct / 100.0
        ) ** (year - 1)

    @property
    def label(self) -> str:
        return "%s %s %s" % (self.code, self.manufacturer, self.model)


# ---------------------------------------------------------------------------
# Catalogue
#
# Selection rationale for the paper (Revision Plan section 3): the four largest
# module suppliers of 2025 (JinkoSolar, LONGi, Trina Solar, JA Solar) per InfoLink
# [E3], plus one all-back-contact specialist (AIKO) and one heterojunction supplier
# (REC). Together they span temperature coefficients from -0.24 to -0.29 %/degC.
# ---------------------------------------------------------------------------

_CATALOGUE: List[PVModule] = [
    PVModule(
        code="F1",
        manufacturer="JinkoSolar",
        model="Tiger Neo 3.0 48QL6 (dual glass)",
        cell_tech="N-type TOPCon (HOT 4.0)",
        pmax_max_w=485.0,
        # CONFLICT: the docx states only "up to 485" with no lower bin. Rather than
        # invent a bottom bin, leave it None; mid/bottom policies refuse this module
        # until the bin range is read off the official PDF.
        pmax_min_w=None,
        length_mm=1762.0, width_mm=1134.0, thickness_mm=30.0,
        eta_datasheet_pct=24.27,
        gamma_pmax=-0.26,
        # CONFLICT: the docx cell reads "<=70 degC", which is an operating-temperature
        # range, not NOCT. The Revision Plan PDF says "on datasheet PDF". Treated as
        # unstated until the PDF is opened.
        noct_c=None,
        noct_kind="unstated",
        deg_year1_pct=None,          # not stated separately in the docx
        deg_annual_pct=0.35,
        source="jinkosolar.eu - Tiger Neo 3.0 page, 48QL6 Dual Glass datasheet",
        source_status="verify",
        notes=(
            "Bin range and NOCT both need reading off the official PDF. "
            "Datasheet cell <=70 degC is an operating range, not NOCT."
        ),
    ),
    PVModule(
        code="F1_ALT",
        manufacturer="JinkoSolar",
        model="Tiger Neo 54HL4R-(V) (previous generation)",
        cell_tech="N-type TOPCon",
        pmax_max_w=460.0, pmax_min_w=435.0,
        length_mm=1762.0, width_mm=1134.0, thickness_mm=30.0,
        eta_datasheet_pct=23.02,
        gamma_pmax=-0.29,
        noct_c=45.0, noct_kind="noct",
        deg_year1_pct=1.0, deg_annual_pct=0.4,   # 87.4 % at yr 30
        source="JKM435-460N-54HL4R-(V)-F8-EN.pdf",
        source_status="verified",
        notes=(
            "Previous product generation. Excluded from the headline comparison by "
            "default: Revision Plan section 2.3 forbids mixing product generations."
        ),
    ),
    PVModule(
        code="F2",
        manufacturer="LONGi",
        model="Hi-MO X10 Explorer LR7-54HVH 475-490M",
        cell_tech="HPBC 2.0 back-contact",
        pmax_max_w=490.0, pmax_min_w=475.0,
        length_mm=1800.0, width_mm=1134.0, thickness_mm=30.0,
        eta_datasheet_pct=24.0,
        gamma_pmax=-0.260,
        noct_c=45.0, noct_kind="noct",
        deg_year1_pct=1.0, deg_annual_pct=0.35,  # 88.85 % at yr 30
        source="static.longi.com datasheet",
        source_status="verify",
        notes=(
            "Datasheet footer reads 20241 BGV02 Draft; a higher-power Scientist "
            "variant exists. Pick one variant and stay consistent."
        ),
    ),
    PVModule(
        code="F3",
        manufacturer="Trina Solar",
        model="Vertex S+ TSM-NEG9R.28",
        cell_tech="N-type i-TOPCon, dual glass",
        pmax_max_w=475.0, pmax_min_w=450.0,
        length_mm=1762.0, width_mm=1134.0, thickness_mm=30.0,
        eta_datasheet_pct=23.8,
        gamma_pmax=-0.29,
        noct_c=43.0, noct_kind="noct",
        deg_year1_pct=1.0, deg_annual_pct=0.4,   # 87.4 % at yr 30
        source="Trina datasheet TSM_EN_2026_A (EU)",
        source_status="verified",
    ),
    PVModule(
        code="F4",
        manufacturer="JA Solar",
        model="DeepBlue 4.0 Pro JAM54D40 LB",
        cell_tech="N-type TOPCon, bifacial dual glass",
        # CONFLICT: docx says 435-460 W / 23.7 % / -0.25 %/degC;
        #           Revision Plan PDF section 3 says 450-475 W / 23.8 % / -0.29 %/degC.
        # Per the decision on sources, the docx wins. Note the docx row is internally
        # inconsistent: 23.7 % over 1762x1134 mm implies ~473 W, not 460 W, so the row
        # probably mixes the two variants. eta_discrepancy_pp() surfaces this.
        pmax_max_w=460.0, pmax_min_w=435.0,
        length_mm=1762.0, width_mm=1134.0, thickness_mm=30.0,
        eta_datasheet_pct=23.7,
        gamma_pmax=-0.25,
        noct_c=45.0, noct_kind="noct",
        deg_year1_pct=1.0, deg_annual_pct=0.4,   # 87.4 % at yr 30
        source="jasolar.eu product page (435-460 W version)",
        source_status="verify",
        notes=(
            "docx and Revision Plan PDF disagree on bin, efficiency and gamma. "
            "docx values used. The docx eta (23.7 %) does not match its own bin "
            "(460 W over 1762x1134 mm = 23.02 %); confirm against the official PDF."
        ),
    ),
    PVModule(
        code="F5",
        manufacturer="AIKO",
        model="Neostar 3N54 (AIKO-A-MCE54Mw)",
        cell_tech="ABC all-back-contact",
        pmax_max_w=495.0, pmax_min_w=470.0,
        length_mm=1762.0, width_mm=1134.0, thickness_mm=30.0,
        eta_datasheet_pct=24.8,
        gamma_pmax=-0.26,
        # CONFLICT: docx states "~38.5-39.0 degC", far below any plausible NOCT and
        # matching an NMOT figure instead. Flagged, not silently used.
        noct_c=38.75, noct_kind="nmot_suspected",
        deg_year1_pct=1.0, deg_annual_pct=0.35,
        source="aikosolar.com Neostar 3N54 PDF",
        source_status="verify",
        notes=(
            "ENF lists 3N54 as discontinued; successor Neostar 3P54 (up to 500 W, "
            "25.0 %) may be the right module. Stated 38.5-39.0 degC looks like NMOT, "
            "not NOCT."
        ),
    ),
    PVModule(
        code="F6",
        manufacturer="REC",
        model="Alpha Pure-RX",
        cell_tech="Heterojunction (HJT)",
        pmax_max_w=475.0, pmax_min_w=450.0,
        length_mm=1728.0, width_mm=1205.0, thickness_mm=30.0,
        eta_datasheet_pct=22.8,
        gamma_pmax=-0.24,
        noct_c=44.0, noct_kind="nmot",
        # Datasheet warrants ">92 % at yr 25" without printing an annual rate.
        # Inverting the compound form with a 1 % first year:
        #   d = 1 - (0.92/0.99)^(1/24) = 0.3067 %/yr
        deg_year1_pct=1.0, deg_annual_pct=0.3067,
        source="recgroup.com datasheet (EN, 2025-01)",
        source_status="verified",
        notes=(
            "Annual degradation derived from the >92 % at yr 25 warranty, not printed "
            "directly. Best temperature coefficient but lowest capacity per m2."
        ),
    ),
]

MODULES: Dict[str, PVModule] = {m.code: m for m in _CATALOGUE}

#: The six modules that form the IV.B comparison table. F1_ALT is a previous-generation
#: module kept for reference only -- including it would break the "compare the same
#: product generation" rule in Revision Plan section 2.3.
DEFAULT_COMPARISON: Tuple[str, ...] = ("F1", "F2", "F3", "F4", "F5", "F6")

#: Access date to cite alongside every datasheet in the IEEE reference list.
DATASHEET_ACCESS_DATE = "2026-09-11"


def get_module(code: str) -> PVModule:
    """Look up one module by its revision-plan code (F1, F2, ...)."""
    try:
        return MODULES[code.upper()]
    except KeyError:
        raise KeyError(
            "unknown module code %r; available: %s" % (code, sorted(MODULES))
        ) from None


def list_modules(codes: Optional[List[str]] = None) -> List[PVModule]:
    """Return modules in catalogue order, defaulting to the IV.B comparison set."""
    if codes is None:
        codes = list(DEFAULT_COMPARISON)
    return [get_module(c) for c in codes]


def citation(module: PVModule, ref_number: str = "x") -> str:
    """IEEE-style datasheet citation, per the template in Revision Plan section 3."""
    return (
        '[%s] %s, "%s %s module," Datasheet, %s. [Online]. Available: %s. '
        "Accessed: %s."
        % (
            ref_number,
            module.manufacturer,
            module.model,
            module.cell_tech,
            DATASHEET_ACCESS_DATE[:4],
            module.source,
            DATASHEET_ACCESS_DATE,
        )
    )


def catalogue_issues(policy: BinPolicy = "top", tol_pp: float = 0.1) -> List[str]:
    """Rows that still need checking against a manufacturer PDF.

    The dashboard shows this so unverified cells are visible rather than being
    presented as if they were confirmed.
    """
    issues: List[str] = []
    for m in _CATALOGUE:
        if m.source_status != "verified":
            issues.append(
                "%s: source marked '%s' -- %s" % (m.code, m.source_status, m.notes or m.source)
            )
        if m.bin_available(policy):
            gap = m.eta_discrepancy_pp(policy)
            if abs(gap) > tol_pp:
                issues.append(
                    "%s: derived eta %.2f %% differs from the datasheet headline "
                    "%.2f %% by %+.2f pp" % (m.code, m.eta_pct(policy), m.eta_datasheet_pct, gap)
                )
        if m.noct_kind != "noct":
            issues.append("%s: thermal rating is '%s', not a plain NOCT" % (m.code, m.noct_kind))
    return issues
