"""
Emissions and Environment -- vehicle emissions, fuel, noise, dispersion and exposure
===================================================================================

**The environmental consequence of a traffic pattern.**  Every other module in
this family answers a question about *movement*; this one answers what that
movement costs the air, the ear and the thermometer.  It sits deliberately
downstream of ``traffic_flow``, which manufactures the speeds, accelerations,
idle times and stop counts that arrive here as input, and it is deliberately
apart from ``green_infrastructure``, which is about habitat, canopy and
biodiversity rather than pollution.  Feed it a link with a flow, a mean speed
and a stop count and it will return grams of NOx, litres of fuel, a facade
noise level in dB(A), a kerbside concentration in ug/m3, a population-weighted
exposure, an attributable fraction and a social cost of carbon.

The white paper's network objective uses a deliberately cheap linear edge proxy

    E_e = a * f_e * t_e + b * idle_e + c * stop_e

-- flow times travel time, idle vehicle-hours, and stops -- because an
assignment or signal-optimisation loop cannot afford a second-by-second modal
model inside its inner iteration.  This module implements *both* ends of that
bargain: the modal models that are physically right (COPERT-style speed curves,
MOVES operating-mode bins driven by Vehicle Specific Power, the Akcelik-Biggs
elemental fuel model) *and* the proxy itself, plus the least-squares routine
that calibrates ``(a, b, c)`` against modal output and the rank test that
certifies the proxy still orders links the way the modal model does.  A proxy
you have not calibrated is a guess; a proxy you have is a legitimate surrogate.

Everything is pure standard library.  The error function, the Gaussian-plume
quadrature, the decibel algebra, the barrier diffraction and the rational
emission curves are all built by hand.  No NumPy, no SciPy, no sibling-module
imports -- drop ``emissions_environment_template.py`` into a hackathon sandbox
and go.

Contents
--------
0.  **Validation helpers** -- ``EmissionsError`` and the value guards, plus the
    small symmetric linear solver used by the proxy calibration.
1.  **Units, constants and the pollutant registry** -- speed and mass
    conversions, fuel properties, global warming potentials.
2.  **Speed-dependent emission factors** -- the COPERT-5 rational-polynomial
    curve ``EF(v) = (a v^2 + b v + c + d/v) / (e v^2 + f v + g)``, a base-curve
    table per vehicle category and pollutant, Euro-class reduction factors, and
    the curve minimum that proves the U shape.
3.  **Fleet composition and cold start** -- share-weighted fleet factors and
    the COPERT cold-start excess (the ``beta`` mileage fraction and the
    temperature-dependent cold/hot ratio).
4.  **Vehicle Specific Power and MOVES operating modes** -- ``VSP`` in kW/tonne
    from the road-load coefficients, the full MOVES operating-mode bin
    definition, and a representative bin rate table.
5.  **Speed traces, drive cycles and modal emissions** -- accelerations, idle
    time, stop counting, deterministic synthetic cycles and second-by-second
    modal emission integration.
6.  **Fuel consumption** -- tractive power, the Akcelik-Biggs instantaneous
    model and its four-mode elemental (cruise / idle / accelerate / decelerate)
    decomposition, and fuel from a trace.
7.  **The linear edge proxy** -- ``E = a f t + b idle + c stop``, network
    aggregation, non-negative least-squares calibration against modal output,
    and the Kendall rank-consistency certificate.
8.  **Non-exhaust particulate matter** -- brake, tyre and road-surface wear
    with the EMEP/EEA speed corrections and size fractions.
9.  **Acoustics fundamentals** -- decibel algebra, point and line geometric
    spreading, atmospheric absorption, ground effect and Maekawa barrier
    diffraction.
10. **Road-traffic noise** -- FHWA TNM-lite REMELs and CNOSSOS-EU-lite source
    powers, the equivalent level at a receptor, ``L_Aeq`` / ``L_den`` /
    ``L_night``, facade levels and the annoyance dose-response curves.
11. **Gaussian-plume dispersion** -- Pasquill-Gifford stability classification,
    Briggs sigma curves (rural and urban), plume rise, the point-source plume,
    the crosswind-integrated form and the closed-form finite line source.
12. **Box and street-canyon models** -- the steady-state and transient box
    models and an OSPM-lite canyon with an explicit recirculation vortex.
13. **Background and exposure** -- background addition and population-weighted
    exposure over zones.
14. **Health impact** -- relative risk, attributable fraction, premature deaths
    and years of life lost.
15. **Carbon accounting** -- CO2-equivalent aggregation, discounting and the
    social cost of carbon.
16. **Urban heat and thermal comfort** -- street-canyon sky view factor, mean
    radiant temperature and the UTCI / PET lite indices.
17. **Payload builders** -- JSON-able dicts for a frontend.
18. **Demo**.

Conventions
-----------
* **Emission factors are g/km**; emission masses are grams unless a name says
  otherwise.  ``grams_to_tonnes`` is the only mass conversion you should need.
* **Traffic units are (km, h)**: flow in veh/h, speed in km/h, travel time in
  hours.  **Modal units are SI (m, s)**: trace speeds in m/s, accelerations in
  m/s^2, VSP in kW/tonne.  ``kmh_to_ms`` / ``ms_to_kmh`` bridge the two and
  every function's docstring says which world it lives in.
* **Concentrations are g/m3 internally** because that is what the dispersion
  algebra produces; ``ug_per_m3`` converts for reporting and the health layer
  takes ug/m3 explicitly.
* **Sound levels are A-weighted decibels** re 20 uPa.  ``db_sum`` is energy
  addition performed with ``math.fsum`` over the linear energies, so it is
  exactly commutative and associative.
* The **edge proxy** takes ``flow`` in veh/h, ``travel_time`` in hours, ``idle``
  in vehicle-hours and ``stops`` as a plain count, so ``a`` is grams per
  vehicle-hour of travel, ``b`` is grams per vehicle-hour of idling and ``c``
  is grams per stop.
* **Scale discipline**: urban quantities span many orders of magnitude (a
  concentration of 1e-9 g/m3 and a carbon cost of 1e9 currency units both turn
  up), so no guard in this file compares a data-scaled quantity against an
  absolute epsilon.  Guards are either exact-zero tests or relative to the
  scale of the data they guard.
* Everything stochastic takes an explicit ``seed`` and uses ``random.Random``,
  and no numeric result depends on the iteration order of a set or dict, so
  results are byte-identical across runs and independent of PYTHONHASHSEED.
* British spelling throughout the prose.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Optional interop with the core probabilistic module (erf / normal CDF for the
# dispersion integrals).  Guarded, with a stdlib fallback, so this file remains
# standalone.
# ---------------------------------------------------------------------------
try:                                                    # pragma: no cover
    from algorithms.core.probabilistic.probabilistic_template import (
        erf as _core_erf, norm_cdf as _core_norm_cdf)
    _HAVE_CORE_PROBABILISTIC = True
except ImportError:                                     # pragma: no cover
    _HAVE_CORE_PROBABILISTIC = False

__version__ = "1.0.0"

Grams = float
Concentration = float
Decibels = float
Vector = List[float]
Matrix = List[List[float]]

_EPS = 1e-12
_REL_TOL = 1e-12
_SQRT2 = math.sqrt(2.0)
_SQRT_2PI = math.sqrt(2.0 * math.pi)

__all__ = [
    # validation
    "EmissionsError",
    # units, constants and registries
    "POLLUTANTS", "VEHICLE_CATEGORIES", "EURO_CLASSES", "GWP100",
    "FUEL_PROPERTIES", "kmh_to_ms", "ms_to_kmh", "mph_to_kmh", "kmh_to_mph",
    "grams_to_tonnes", "ug_per_m3", "g_per_m3", "fuel_litres_to_co2_g",
    # emission factors
    "EmissionFactorCurve", "BASE_EMISSION_CURVES", "EURO_REDUCTION_FACTORS",
    "emission_factor_curve", "emission_factor", "euro_reduction_factor",
    "emission_factor_table", "curve_minimum", "hot_emissions",
    # fleet and cold start
    "FleetComposition", "fleet_emission_factor", "fleet_hot_emissions",
    "cold_start_beta", "cold_start_ratio", "ColdStartResult",
    "cold_start_excess",
    # vsp and moves operating modes
    "VSPCoefficients", "LIGHT_DUTY_VSP", "HEAVY_DUTY_VSP",
    "vehicle_specific_power", "road_load_power", "OPMODE_IDS",
    "operating_mode_bin", "opmode_vsp_midpoint", "opmode_label",
    "MOVES_RATE_PARAMS", "moves_opmode_rate", "OpModeDistribution",
    "opmode_distribution",
    # traces, cycles and modal emissions
    "SpeedTrace", "trace_from_speeds", "constant_speed_cycle", "stop_go_cycle",
    "synthetic_drive_cycle", "ModalResult", "modal_emissions",
    # fuel
    "AkcelikBiggsParameters", "PETROL_CAR", "DIESEL_CAR", "URBAN_BUS",
    "tractive_power", "instantaneous_fuel_rate", "FuelResult",
    "four_mode_fuel", "elemental_fuel_model", "fuel_from_trace",
    # edge proxy
    "EdgeProxyCoefficients", "EdgeObservation", "edge_emission_proxy",
    "network_emission_proxy", "ProxyCalibration", "calibrate_edge_proxy",
    "RankConsistency", "proxy_rank_consistency",
    # non-exhaust pm
    "NON_EXHAUST_TSP", "NON_EXHAUST_SIZE_FRACTIONS", "NON_EXHAUST_MASS_FACTOR",
    "brake_wear_speed_factor", "tyre_wear_speed_factor", "NonExhaustResult",
    "non_exhaust_emission_factor", "non_exhaust_emissions",
    # acoustics fundamentals
    "db_sum", "db_subtract", "db_energy_mean", "db_weighted_energy_mean",
    "point_source_attenuation", "line_source_attenuation",
    "geometric_spreading", "atmospheric_absorption", "ground_attenuation",
    "path_difference", "fresnel_number", "maekawa_barrier_attenuation",
    # road traffic noise
    "REMELParameters", "FHWA_REMELS", "fhwa_remel", "fhwa_leq",
    "CnossosParameters", "CNOSSOS_COEFFICIENTS", "cnossos_sound_power",
    "cnossos_line_sound_power", "NOISE_CATEGORIES", "TrafficStream",
    "NoiseResult",
    "road_noise_level", "l_aeq", "l_den", "l_night", "facade_level",
    "percent_highly_annoyed", "percent_annoyed", "percent_sleep_disturbed",
    # dispersion
    "STABILITY_CLASSES", "TERRAIN_TYPES", "pasquill_stability",
    "stability_description", "briggs_sigma_y",
    "briggs_sigma_z", "plume_rise", "gaussian_plume_concentration",
    "crosswind_integrated_concentration", "line_source_concentration",
    "road_link_concentration", "plume_mass_flux",
    # box and canyon
    "box_model_concentration", "box_model_buildup", "StreetCanyon",
    "CanyonResult", "canyon_concentration",
    # background and exposure
    "add_background", "ExposureZone", "population_weighted_exposure",
    "exceedance_population", "annual_mean_from_hourly",
    # health
    "CONCENTRATION_RESPONSE", "HEALTH_POLLUTANTS", "relative_risk",
    "attributable_fraction",
    "premature_deaths", "years_of_life_lost", "HealthImpact",
    "health_impact_assessment",
    # carbon
    "co2_equivalent", "discount_factor", "social_cost_of_carbon",
    "CarbonAccount", "carbon_account",
    # urban heat and comfort
    "sky_view_factor", "sky_view_factor_asymmetric",
    "saturation_vapour_pressure", "vapour_pressure", "wind_speed_at_height",
    "mean_radiant_temperature", "utci_lite", "pet_lite",
    "thermal_stress_category",
    # payloads
    "build_emissions_environment_payload", "build_a2ui_payload",
]


# ===========================================================================
# 0. VALIDATION HELPERS
# ===========================================================================
# These guards validate SIGN, RANGE and FINITENESS -- the things a caller gets
# wrong.  They deliberately do not validate MAGNITUDE.  Feeding a physical
# routine 1e300 ug/m3, a 3100 dB sound level, a 1e-300 m dispersion parameter or
# minus 1e300 years of discounting can surface a bare ``OverflowError``,
# ``ZeroDivisionError`` or a signed infinity from the arithmetic underneath, and
# that is the documented convention rather than a contract this module makes:
# none of those inputs is reachable from any use of the module, and wrapping
# every ``math.exp`` in a magnitude check would cost every real call.  The suite
# pins the current behaviour of each so it is recorded rather than accidental.
# The exception that PROVES the rule is ``annual_mean_from_hourly``, where an
# entirely missing averaging period is an ordinary operational case and so
# raises ``EmissionsError`` properly.
# ---------------------------------------------------------------------------
class EmissionsError(ValueError):
    """Invalid input to an emissions / environment routine."""


def _as_float(x: Any, name: str) -> float:
    """Coerce to a finite float or raise ``EmissionsError``."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise EmissionsError("%s must be a real number (got %r)." % (name, x))
    if math.isnan(v) or math.isinf(v):
        raise EmissionsError("%s must be finite (got %r)." % (name, x))
    return v


def _as_floats(x: Sequence[Any], name: str = "x") -> List[float]:
    """Coerce a sequence to a list of finite floats."""
    try:
        out = [float(v) for v in x]
    except (TypeError, ValueError):
        raise EmissionsError("%s must be a sequence of real numbers." % name)
    for v in out:
        if math.isnan(v) or math.isinf(v):
            raise EmissionsError("%s contains non-finite values." % name)
    return out


def _positive(x: Any, name: str) -> float:
    """Coerce to a strictly positive finite float."""
    v = _as_float(x, name)
    if v <= 0.0:
        raise EmissionsError("%s must be strictly positive (got %g)." % (name, v))
    return v


def _non_negative(x: Any, name: str) -> float:
    """Coerce to a non-negative finite float."""
    v = _as_float(x, name)
    if v < 0.0:
        raise EmissionsError("%s must be non-negative (got %g)." % (name, v))
    return v


def _fraction(x: Any, name: str) -> float:
    """Coerce to a float in [0, 1]."""
    v = _as_float(x, name)
    if v < 0.0 or v > 1.0:
        raise EmissionsError("%s must lie in [0, 1] (got %g)." % (name, v))
    return v


def _check_len(x: Sequence[Any], min_len: int, name: str) -> None:
    """Raise unless the sequence has at least ``min_len`` entries."""
    if len(x) < min_len:
        raise EmissionsError("%s needs at least %d entries (got %d)."
                             % (name, min_len, len(x)))


def _key(name: str, value: str, allowed: Sequence[str]) -> str:
    """Normalise and validate a registry key, naming the allowed set on failure."""
    v = str(value).strip()
    if v not in allowed:
        raise EmissionsError("unknown %s %r -- expected one of %s."
                             % (name, value, ", ".join(sorted(allowed))))
    return v


#: Beyond this argument ``erf`` is +/-1 to better than 1e-17 -- inside the
#: last double below 1.0 -- so the saturated branch is EXACT, not a tolerance.
#: Delegation to the core implementation stops here on purpose: that routine
#: squares its argument internally, which overflows to ``inf`` somewhere above
#: 1e154 and returns ``nan``, while ``math.erf`` saturates cleanly at any
#: magnitude.  Without this gate the module would answer ``nan`` where the core
#: module happens to be importable and the right answer where it is not.
_ERF_SATURATION = 6.0
#: Same idea for the normal CDF, whose argument is larger by sqrt(2).
_NORM_CDF_SATURATION = _ERF_SATURATION * _SQRT2


def _erf(x: float) -> float:
    """Error function -- the core ``probabilistic`` implementation when present.

    The delegation is gated on magnitude so that both configurations of this
    module (core present, core absent) return bit-identical saturated values
    for extreme arguments; see ``_ERF_SATURATION``.
    """
    if _HAVE_CORE_PROBABILISTIC and -_ERF_SATURATION <= x <= _ERF_SATURATION:
        return float(_core_erf(x))
    return math.erf(x)


def _norm_cdf(x: float) -> float:
    """Standard normal CDF -- the core ``probabilistic`` version when present."""
    if (_HAVE_CORE_PROBABILISTIC
            and -_NORM_CDF_SATURATION <= x <= _NORM_CDF_SATURATION):
        return float(_core_norm_cdf(x))
    return 0.5 * (1.0 + math.erf(x / _SQRT2))


def _solve_symmetric(a: Matrix, b: Vector) -> Vector:
    """Solve a small symmetric system by Cholesky, falling back to pivoted LU."""
    n = len(a)
    if n == 0:
        return []
    for row in a:
        if len(row) != n:
            raise EmissionsError("matrix must be square.")
    if len(b) != n:
        raise EmissionsError("right-hand side length %d != matrix order %d."
                             % (len(b), n))
    # Scale-aware pivot guard: compare against the magnitude of the matrix,
    # never an absolute epsilon -- the normal equations here are sums of
    # squares of urban quantities and can legitimately sit at 1e-8 or 1e12.
    scale = 0.0
    for i in range(n):
        for j in range(n):
            if abs(a[i][j]) > scale:
                scale = abs(a[i][j])
    if scale == 0.0:
        raise EmissionsError("cannot solve an all-zero system.")
    tol = _EPS * scale

    # -- Cholesky attempt ---------------------------------------------------
    chol = [[0.0] * n for _ in range(n)]
    ok = True
    for i in range(n):
        for j in range(i + 1):
            s = math.fsum([chol[i][k] * chol[j][k] for k in range(j)])
            if i == j:
                d = a[i][i] - s
                if d <= tol:
                    ok = False
                    break
                chol[i][j] = math.sqrt(d)
            else:
                chol[i][j] = (a[i][j] - s) / chol[j][j]
        if not ok:
            break
    if ok:
        y = [0.0] * n
        for i in range(n):
            s = math.fsum([chol[i][k] * y[k] for k in range(i)])
            y[i] = (b[i] - s) / chol[i][i]
        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            s = math.fsum([chol[k][i] * x[k] for k in range(i + 1, n)])
            x[i] = (y[i] - s) / chol[i][i]
        return x

    # -- Gaussian elimination with partial pivoting -------------------------
    m = [list(a[i]) + [b[i]] for i in range(n)]
    for col in range(n):
        piv = col
        best = abs(m[col][col])
        for r in range(col + 1, n):
            if abs(m[r][col]) > best:
                best = abs(m[r][col])
                piv = r
        if best <= tol:
            raise EmissionsError("system is singular to working precision.")
        if piv != col:
            m[col], m[piv] = m[piv], m[col]
        pv = m[col][col]
        for r in range(col + 1, n):
            f = m[r][col] / pv
            if f == 0.0:
                continue
            for cc in range(col, n + 1):
                m[r][cc] -= f * m[col][cc]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = math.fsum([m[i][j] * x[j] for j in range(i + 1, n)])
        x[i] = (m[i][n] - s) / m[i][i]
    return x


# ===========================================================================
# 1. UNITS, CONSTANTS AND THE POLLUTANT REGISTRY
# ===========================================================================
#: Pollutants this module knows, in a fixed reporting order.
POLLUTANTS = ("NOx", "PM10", "PM2.5", "CO", "HC", "CO2")

#: Vehicle categories, in a fixed reporting order.
VEHICLE_CATEGORIES = ("passenger_car_petrol", "passenger_car_diesel",
                      "lgv_diesel", "hgv_diesel", "bus_diesel", "motorcycle")

#: Euro emission standards, oldest first.
EURO_CLASSES = ("Euro 3", "Euro 4", "Euro 5", "Euro 6")

#: 100-year global warming potentials (IPCC AR5, no climate-carbon feedback).
GWP100 = {"CO2": 1.0, "CH4": 28.0, "N2O": 265.0, "HFC-134a": 1300.0}

#: Fuel properties: density kg/L, net calorific value MJ/kg and tank-to-wheel
#: CO2 in grams per litre burnt.
FUEL_PROPERTIES = {
    "petrol": {"density_kg_l": 0.745, "lhv_mj_kg": 43.2, "co2_g_per_l": 2310.0},
    "diesel": {"density_kg_l": 0.832, "lhv_mj_kg": 42.7, "co2_g_per_l": 2680.0},
    "cng": {"density_kg_l": 0.180, "lhv_mj_kg": 48.0, "co2_g_per_l": 1630.0},
}

#: Standard gravity, m/s^2 -- the value the MOVES VSP definition uses.
_G = 9.8066

#: Kilometres per hour in one metre per second.
_KMH_PER_MS = 3.6
#: Kilometres per hour in one mile per hour.
_KMH_PER_MPH = 1.609344


def kmh_to_ms(v_kmh: float) -> float:
    """Convert km/h to m/s."""
    return _as_float(v_kmh, "v_kmh") / _KMH_PER_MS


def ms_to_kmh(v_ms: float) -> float:
    """Convert m/s to km/h."""
    return _as_float(v_ms, "v_ms") * _KMH_PER_MS


def mph_to_kmh(v_mph: float) -> float:
    """Convert miles per hour to km/h."""
    return _as_float(v_mph, "v_mph") * _KMH_PER_MPH


def kmh_to_mph(v_kmh: float) -> float:
    """Convert km/h to miles per hour."""
    return _as_float(v_kmh, "v_kmh") / _KMH_PER_MPH


def grams_to_tonnes(grams: float) -> float:
    """Convert grams to metric tonnes."""
    return _as_float(grams, "grams") * 1e-6


def ug_per_m3(c_g_per_m3: float) -> float:
    """Convert a concentration from g/m3 to ug/m3 (the reporting unit)."""
    return _as_float(c_g_per_m3, "c_g_per_m3") * 1e6


def g_per_m3(c_ug_per_m3: float) -> float:
    """Convert a concentration from ug/m3 back to g/m3 (the working unit)."""
    return _as_float(c_ug_per_m3, "c_ug_per_m3") * 1e-6


def fuel_litres_to_co2_g(litres: float, fuel: str = "petrol") -> float:
    """Tank-to-wheel CO2 in grams from litres of fuel burnt."""
    lit = _non_negative(litres, "litres")
    key = _key("fuel", fuel, tuple(sorted(FUEL_PROPERTIES.keys())))
    return lit * FUEL_PROPERTIES[key]["co2_g_per_l"]


# ===========================================================================
# 2. SPEED-DEPENDENT EMISSION FACTORS (COPERT-STYLE)
# ===========================================================================
@dataclass(frozen=True)
class EmissionFactorCurve:
    """A COPERT-5 rational-polynomial hot emission factor curve, in g/km."""

    pollutant: str
    category: str
    a: float
    b: float
    c: float
    d: float
    e: float
    f: float
    g: float
    v_min: float = 10.0
    v_max: float = 130.0
    source: str = "COPERT-5 rational form, representative coefficients"

    @classmethod
    def hyperbolic(cls, pollutant: str, category: str, a_over_v: float,
                   constant: float, linear: float, v_min: float = 10.0,
                   v_max: float = 130.0,
                   source: str = "COPERT-5 rational form (hyperbolic subform)"
                   ) -> "EmissionFactorCurve":
        """Build ``EF(v) = A/v + B + C v`` inside the general rational form."""
        big_a = _positive(a_over_v, "a_over_v")
        big_b = _as_float(constant, "constant")
        big_c = _positive(linear, "linear")
        floor = 2.0 * math.sqrt(big_a * big_c) + big_b
        if floor <= 0.0:
            raise EmissionsError(
                "hyperbolic curve for %s/%s would go non-positive (minimum "
                "%.6g g/km)." % (category, pollutant, floor))
        return cls(pollutant=pollutant, category=category,
                   a=big_c, b=big_b, c=big_a, d=0.0,
                   e=0.0, f=1.0, g=0.0,
                   v_min=v_min, v_max=v_max, source=source)

    def __post_init__(self) -> None:
        if self.v_min <= 0.0 or self.v_max <= self.v_min:
            raise EmissionsError(
                "curve %s/%s needs 0 < v_min < v_max (got %g, %g)."
                % (self.category, self.pollutant, self.v_min, self.v_max))

    def factor(self, speed_kmh: float, clamp: bool = True) -> float:
        """Hot emission factor in g/km at a mean travelling speed in km/h."""
        v = _as_float(speed_kmh, "speed_kmh")
        if clamp:
            if v < self.v_min:
                v = self.v_min
            elif v > self.v_max:
                v = self.v_max
        elif v <= 0.0:
            raise EmissionsError("speed_kmh must be positive for a COPERT "
                                 "curve (got %g)." % v)
        num = self.a * v * v + self.b * v + self.c + self.d / v
        den = self.e * v * v + self.f * v + self.g
        # Relative guard: the denominator is a speed-scaled quantity, so it is
        # compared against the magnitude of its own terms, not an absolute eps.
        den_scale = (abs(self.e) * v * v + abs(self.f) * v + abs(self.g))
        if den_scale == 0.0 or abs(den) <= _EPS * den_scale:
            raise EmissionsError("curve %s/%s has a vanishing denominator at "
                                 "%g km/h." % (self.category, self.pollutant, v))
        return num / den

    def describe(self) -> Dict[str, Any]:
        """Plain-dict description of the curve for reporting."""
        return {"pollutant": self.pollutant, "category": self.category,
                "coefficients": [self.a, self.b, self.c, self.d,
                                 self.e, self.f, self.g],
                "vMinKmh": self.v_min, "vMaxKmh": self.v_max,
                "source": self.source}


def _hyp(pollutant: str, category: str, big_a: float, big_b: float,
         big_c: float, v_min: float, v_max: float) -> EmissionFactorCurve:
    """Shorthand for the hyperbolic subform used by the built-in table."""
    return EmissionFactorCurve.hyperbolic(pollutant, category, big_a, big_b,
                                          big_c, v_min, v_max)


#: Euro class the built-in base curves are quoted for.
_REFERENCE_EURO = "Euro 5"

#: Base hot emission factor curves, keyed ``(category, pollutant)``, quoted at
#: the reference Euro class.  Coefficients are representative of the published
#: COPERT-5 / EMEP-EEA Tier 3 speed dependencies: an inverse-speed congestion
#: term, a constant and a linear high-speed term, so every curve is convex with
#: an interior minimum between roughly 50 and 90 km/h.
BASE_EMISSION_CURVES = {}  # type: Dict[Tuple[str, str], EmissionFactorCurve]

for _cat, _vlo, _vhi, _rows in (
    ("passenger_car_petrol", 10.0, 130.0, (
        ("CO2", 5000.0, -28.0, 1.38),
        ("NOx", 1.5, 0.005, 0.00020),
        ("PM10", 0.060, -0.00050, 0.000012),
        ("PM2.5", 0.057, -0.000475, 0.0000114),
        ("CO", 15.0, -0.20, 0.0045),
        ("HC", 3.0, -0.035, 0.00065),
    )),
    ("passenger_car_diesel", 10.0, 130.0, (
        ("CO2", 4400.0, -25.0, 1.20),
        ("NOx", 12.0, 0.30, 0.0018),
        ("PM10", 0.100, -0.00080, 0.000020),
        ("PM2.5", 0.095, -0.00076, 0.000019),
        ("CO", 3.0, -0.030, 0.00040),
        ("HC", 1.2, -0.012, 0.00012),
    )),
    ("lgv_diesel", 10.0, 120.0, (
        ("CO2", 6200.0, -32.0, 1.75),
        ("NOx", 17.0, 0.42, 0.0026),
        ("PM10", 0.160, -0.00120, 0.000032),
        ("PM2.5", 0.152, -0.00114, 0.0000304),
        ("CO", 4.5, -0.045, 0.00060),
        ("HC", 1.8, -0.018, 0.00018),
    )),
    ("hgv_diesel", 10.0, 90.0, (
        ("CO2", 22000.0, -90.0, 6.50),
        ("NOx", 70.0, 1.60, 0.0140),
        ("PM10", 0.750, -0.00600, 0.00015),
        ("PM2.5", 0.7125, -0.00570, 0.0001425),
        ("CO", 28.0, -0.25, 0.00450),
        ("HC", 9.0, -0.085, 0.00150),
    )),
    ("bus_diesel", 10.0, 80.0, (
        ("CO2", 20000.0, -85.0, 7.00),
        ("NOx", 65.0, 1.50, 0.0160),
        ("PM10", 0.700, -0.00550, 0.00018),
        ("PM2.5", 0.665, -0.005225, 0.000171),
        ("CO", 26.0, -0.24, 0.00600),
        ("HC", 8.0, -0.080, 0.00180),
    )),
    ("motorcycle", 10.0, 110.0, (
        ("CO2", 2200.0, -14.0, 0.75),
        ("NOx", 2.0, 0.010, 0.00025),
        ("PM10", 0.080, -0.00060, 0.000018),
        ("PM2.5", 0.076, -0.00057, 0.0000171),
        ("CO", 60.0, -0.55, 0.01200),
        ("HC", 12.0, -0.11, 0.00220),
    )),
):
    for _pol, _A, _B, _C in _rows:
        BASE_EMISSION_CURVES[(_cat, _pol)] = _hyp(_pol, _cat, _A, _B, _C,
                                                  _vlo, _vhi)
del _cat, _vlo, _vhi, _rows, _pol, _A, _B, _C

#: Multiplicative Euro-class reduction factors relative to the reference class,
#: keyed ``(euro, pollutant)``.  They encode the step changes in the standards
#: plus the usual real-world shortfall against the type-approval limit.
EURO_REDUCTION_FACTORS = {}  # type: Dict[Tuple[str, str], float]
for _pol, _vals in (
    ("NOx", (3.20, 2.20, 1.00, 0.55)),
    ("PM10", (8.00, 4.50, 1.00, 0.85)),
    ("PM2.5", (8.00, 4.50, 1.00, 0.85)),
    ("CO", (2.60, 1.70, 1.00, 0.90)),
    ("HC", (3.00, 1.80, 1.00, 0.85)),
    ("CO2", (1.10, 1.05, 1.00, 0.94)),
):
    for _euro, _v in zip(EURO_CLASSES, _vals):
        EURO_REDUCTION_FACTORS[(_euro, _pol)] = _v
del _pol, _vals, _euro, _v


def emission_factor_curve(category: str, pollutant: str) -> EmissionFactorCurve:
    """Look up the base hot emission factor curve for a category / pollutant."""
    cat = _key("vehicle category", category, VEHICLE_CATEGORIES)
    pol = _key("pollutant", pollutant, POLLUTANTS)
    return BASE_EMISSION_CURVES[(cat, pol)]


def euro_reduction_factor(euro: str, pollutant: str) -> float:
    """Multiplicative factor taking the reference-class curve to a Euro class."""
    eu = _key("Euro class", euro, EURO_CLASSES)
    pol = _key("pollutant", pollutant, POLLUTANTS)
    return EURO_REDUCTION_FACTORS[(eu, pol)]


def emission_factor(pollutant: str, category: str, euro: str,
                    speed_kmh: float) -> float:
    """Hot exhaust emission factor in g/km for one vehicle at a mean speed."""
    curve = emission_factor_curve(category, pollutant)
    return curve.factor(speed_kmh) * euro_reduction_factor(euro, pollutant)


def emission_factor_table(pollutant: str, category: str, euro: str,
                          speeds: Optional[Sequence[float]] = None,
                          n_samples: int = 25) -> List[Dict[str, float]]:
    """Sample the emission factor curve across its documented speed range."""
    curve = emission_factor_curve(category, pollutant)
    if speeds is None:
        n = int(n_samples)
        if n < 2:
            raise EmissionsError("n_samples must be at least 2 (got %d)." % n)
        step = (curve.v_max - curve.v_min) / float(n - 1)
        grid = [curve.v_min + step * i for i in range(n)]
        grid[-1] = curve.v_max
    else:
        grid = _as_floats(speeds, "speeds")
        _check_len(grid, 1, "speeds")
    scale = euro_reduction_factor(euro, pollutant)
    return [{"speedKmh": v, "efGPerKm": curve.factor(v) * scale} for v in grid]


def curve_minimum(curve: EmissionFactorCurve, n_grid: int = 401,
                  n_refine: int = 100) -> Tuple[float, float]:
    """Locate the minimum of a curve over its range by grid scan plus golden section."""
    if not isinstance(curve, EmissionFactorCurve):
        raise EmissionsError("curve_minimum expects an EmissionFactorCurve.")
    n = int(n_grid)
    if n < 3:
        raise EmissionsError("n_grid must be at least 3 (got %d)." % n)
    step = (curve.v_max - curve.v_min) / float(n - 1)
    best_i = 0
    best_v = curve.v_min
    best_f = curve.factor(curve.v_min)
    for i in range(1, n):
        v = curve.v_min + step * i if i < n - 1 else curve.v_max
        fv = curve.factor(v)
        if fv < best_f:                 # strict: the lowest index wins a tie
            best_f = fv
            best_v = v
            best_i = i
    lo = curve.v_min + step * max(0, best_i - 1)
    hi = curve.v_min + step * min(n - 1, best_i + 1)
    if hi <= lo:
        return best_v, best_f
    # Golden-section refinement on the bracketing interval.
    inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
    x1 = hi - inv_phi * (hi - lo)
    x2 = lo + inv_phi * (hi - lo)
    f1 = curve.factor(x1)
    f2 = curve.factor(x2)
    for _ in range(max(1, int(n_refine))):
        if f1 <= f2:
            hi, x2, f2 = x2, x1, f1
            x1 = hi - inv_phi * (hi - lo)
            f1 = curve.factor(x1)
        else:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + inv_phi * (hi - lo)
            f2 = curve.factor(x2)
    v_star = 0.5 * (lo + hi)
    return v_star, curve.factor(v_star)


def hot_emissions(pollutant: str, category: str, euro: str, speed_kmh: float,
                  distance_km: float, vehicles: float = 1.0) -> Grams:
    """Hot exhaust mass in grams: EF(v) x distance x vehicle count, exactly linear."""
    ef = emission_factor(pollutant, category, euro, speed_kmh)
    dist = _non_negative(distance_km, "distance_km")
    veh = _non_negative(vehicles, "vehicles")
    return ef * dist * veh


def _validate_base_curves() -> None:
    """Assert every built-in curve is positive, finite and interior-minimised."""
    for key in sorted(BASE_EMISSION_CURVES.keys()):
        curve = BASE_EMISSION_CURVES[key]
        step = (curve.v_max - curve.v_min) / 50.0
        for i in range(51):
            v = curve.v_min + step * i
            fv = curve.factor(v)
            if not (fv > 0.0) or math.isinf(fv) or math.isnan(fv):
                raise EmissionsError(
                    "built-in curve %s is not positive and finite at %g km/h."
                    % (str(key), v))
        v_star, _ = curve_minimum(curve, 201, 60)
        if not (curve.v_min < v_star < curve.v_max):
            raise EmissionsError(
                "built-in curve %s has its minimum at an endpoint (%g km/h)."
                % (str(key), v_star))


_validate_base_curves()


# ===========================================================================
# 3. FLEET COMPOSITION AND COLD START
# ===========================================================================
@dataclass
class FleetComposition:
    """A fleet as shares over ``(category, Euro class)`` pairs summing to one."""

    shares: Dict[Tuple[str, str], float]
    name: str = "fleet"

    def __post_init__(self) -> None:
        clean = {}  # type: Dict[Tuple[str, str], float]
        for key in sorted(self.shares.keys()):
            if not (isinstance(key, tuple) and len(key) == 2):
                raise EmissionsError("fleet keys must be (category, euro) "
                                     "tuples (got %r)." % (key,))
            cat = _key("vehicle category", key[0], VEHICLE_CATEGORIES)
            eu = _key("Euro class", key[1], EURO_CLASSES)
            share = _non_negative(self.shares[key], "share for %s" % (key,))
            clean[(cat, eu)] = clean.get((cat, eu), 0.0) + share
        if not clean:
            raise EmissionsError("a fleet needs at least one share.")
        self.shares = clean
        total = self.total()
        if total <= 0.0:
            raise EmissionsError("fleet shares sum to zero.")

    def keys_sorted(self) -> List[Tuple[str, str]]:
        """Fleet keys in a fixed, hash-seed independent order."""
        return sorted(self.shares.keys())

    def total(self) -> float:
        """Sum of the shares (one for a normalised fleet)."""
        return math.fsum([self.shares[k] for k in self.keys_sorted()])

    def normalised(self) -> "FleetComposition":
        """Return a copy whose shares sum to exactly one in the fsum sense."""
        total = self.total()
        return FleetComposition({k: self.shares[k] / total
                                 for k in self.keys_sorted()},
                                name=self.name)

    def category_shares(self) -> Dict[str, float]:
        """Shares aggregated over Euro class, keyed by vehicle category."""
        out = {}  # type: Dict[str, float]
        for cat, eu in self.keys_sorted():
            out[cat] = out.get(cat, 0.0) + self.shares[(cat, eu)]
        return out

    def summary(self) -> Dict[str, Any]:
        """JSON-able description of the fleet."""
        cats = self.category_shares()
        return {"name": self.name,
                "total": self.total(),
                "entries": [{"category": c, "euro": e,
                             "share": self.shares[(c, e)]}
                            for c, e in self.keys_sorted()],
                "byCategory": [{"category": c, "share": cats[c]}
                               for c in sorted(cats.keys())]}


def fleet_emission_factor(fleet: FleetComposition, pollutant: str,
                          speed_kmh: float) -> float:
    """Share-weighted fleet hot emission factor in g/km at a mean speed."""
    if not isinstance(fleet, FleetComposition):
        raise EmissionsError("fleet must be a FleetComposition.")
    pol = _key("pollutant", pollutant, POLLUTANTS)
    keys = fleet.keys_sorted()
    total = fleet.total()
    terms = [fleet.shares[k] * emission_factor(pol, k[0], k[1], speed_kmh)
             for k in keys]
    return math.fsum(terms) / total


def fleet_hot_emissions(fleet: FleetComposition, pollutant: str,
                        speed_kmh: float, distance_km: float,
                        vehicles: float = 1.0) -> Grams:
    """Fleet hot exhaust mass in grams over a distance for a vehicle count."""
    ef = fleet_emission_factor(fleet, pollutant, speed_kmh)
    return ef * _non_negative(distance_km, "distance_km") * \
        _non_negative(vehicles, "vehicles")


def cold_start_beta(trip_length_km: float, ambient_c: float) -> float:
    """COPERT beta: the share of mileage driven with a cold engine, in [0, 1]."""
    ltrip = _positive(trip_length_km, "trip_length_km")
    t_amb = _as_float(ambient_c, "ambient_c")
    beta = 0.6474 - 0.02545 * ltrip - (0.00974 - 0.000385 * ltrip) * t_amb
    if beta < 0.0:
        return 0.0
    if beta > 1.0:
        return 1.0
    return beta


#: Cold/hot emission ratio as ``intercept + slope * T`` (T in degrees Celsius).
_COLD_START_RATIO_COEF = {
    "CO": (3.66, -0.09),
    "HC": (2.24, -0.06),
    "NOx": (1.35, -0.013),
    "PM10": (3.10, -0.055),
    "PM2.5": (3.10, -0.055),
    "CO2": (1.30, -0.008),
}

#: Scaling of the cold-start *excess* by vehicle category.
_COLD_START_CATEGORY = {
    "passenger_car_petrol": 1.00,
    "passenger_car_diesel": 0.80,
    "lgv_diesel": 0.85,
    "hgv_diesel": 0.60,
    "bus_diesel": 0.60,
    "motorcycle": 1.20,
}


def cold_start_ratio(pollutant: str, ambient_c: float,
                     category: str = "passenger_car_petrol") -> float:
    """Cold/hot emission ratio at an ambient temperature, never below one."""
    pol = _key("pollutant", pollutant, POLLUTANTS)
    cat = _key("vehicle category", category, VEHICLE_CATEGORIES)
    t_amb = _as_float(ambient_c, "ambient_c")
    intercept, slope = _COLD_START_RATIO_COEF[pol]
    ratio = intercept + slope * t_amb
    excess = (ratio - 1.0) * _COLD_START_CATEGORY[cat]
    if excess < 0.0:
        excess = 0.0
    return 1.0 + excess


@dataclass
class ColdStartResult:
    """Hot, cold-start excess and total mass for a set of trips."""

    pollutant: str
    category: str
    euro: str
    beta: float
    ratio: float
    hot_grams: float
    excess_grams: float

    @property
    def total_grams(self) -> float:
        """Hot plus cold-start excess, in grams."""
        return self.hot_grams + self.excess_grams

    @property
    def excess_share(self) -> float:
        """Cold-start excess as a share of the total (zero when total is zero)."""
        total = self.total_grams
        if total == 0.0:
            return 0.0
        return self.excess_grams / total

    def summary(self) -> Dict[str, Any]:
        """JSON-able description of the cold-start split."""
        return {"pollutant": self.pollutant, "category": self.category,
                "euro": self.euro, "beta": self.beta, "coldHotRatio": self.ratio,
                "hotGrams": self.hot_grams, "excessGrams": self.excess_grams,
                "totalGrams": self.total_grams,
                "excessShare": self.excess_share}


def cold_start_excess(pollutant: str, category: str, euro: str,
                      speed_kmh: float, trip_length_km: float,
                      trips: float = 1.0,
                      ambient_c: float = 20.0) -> ColdStartResult:
    """COPERT cold-start excess: beta x mileage x EF_hot x (cold/hot - 1)."""
    pol = _key("pollutant", pollutant, POLLUTANTS)
    cat = _key("vehicle category", category, VEHICLE_CATEGORIES)
    eu = _key("Euro class", euro, EURO_CLASSES)
    ltrip = _positive(trip_length_km, "trip_length_km")
    n_trips = _non_negative(trips, "trips")
    ef_hot = emission_factor(pol, cat, eu, speed_kmh)
    mileage = ltrip * n_trips
    hot = ef_hot * mileage
    beta = cold_start_beta(ltrip, ambient_c)
    ratio = cold_start_ratio(pol, ambient_c, cat)
    excess = beta * mileage * ef_hot * (ratio - 1.0)
    return ColdStartResult(pollutant=pol, category=cat, euro=eu, beta=beta,
                           ratio=ratio, hot_grams=hot, excess_grams=excess)


# ===========================================================================
# 4. VEHICLE SPECIFIC POWER AND MOVES OPERATING MODES
# ===========================================================================
@dataclass(frozen=True)
class VSPCoefficients:
    """Road-load coefficients and mass for the MOVES VSP definition."""

    rolling_a: float        #: kW*s/m   -- rolling resistance term
    rotating_b: float       #: kW*s^2/m^2 -- rotating / bearing term
    drag_c: float           #: kW*s^3/m^3 -- aerodynamic drag term
    mass_tonnes: float      #: source mass in metric tonnes
    name: str = "light duty"

    def __post_init__(self) -> None:
        if self.mass_tonnes <= 0.0:
            raise EmissionsError("mass_tonnes must be positive (got %g)."
                                 % self.mass_tonnes)


#: MOVES light-duty passenger-car defaults (source type 21).
LIGHT_DUTY_VSP = VSPCoefficients(0.156461, 0.00200193, 0.000492646, 1.4788,
                                 "light duty passenger car")

#: MOVES heavy-duty single-unit-truck-scale defaults.
HEAVY_DUTY_VSP = VSPCoefficients(1.29, 0.0, 0.003509, 17.1, "heavy duty truck")


def vehicle_specific_power(speed_ms: float, accel_ms2: float = 0.0,
                           grade: float = 0.0,
                           coeffs: VSPCoefficients = LIGHT_DUTY_VSP) -> float:
    """Vehicle Specific Power in kW/tonne -- SI in, exactly zero at zero speed."""
    v = _non_negative(speed_ms, "speed_ms")
    a = _as_float(accel_ms2, "accel_ms2")
    theta = _as_float(grade, "grade")
    if not isinstance(coeffs, VSPCoefficients):
        raise EmissionsError("coeffs must be a VSPCoefficients.")
    road_load = (coeffs.rolling_a * v
                 + coeffs.rotating_b * v * v
                 + coeffs.drag_c * v * v * v) / coeffs.mass_tonnes
    inertial = (a + _G * theta) * v
    return road_load + inertial


def road_load_power(speed_ms: float, accel_ms2: float = 0.0,
                    grade: float = 0.0,
                    coeffs: VSPCoefficients = LIGHT_DUTY_VSP) -> float:
    """Total tractive power in kW -- VSP times the source mass in tonnes."""
    return vehicle_specific_power(speed_ms, accel_ms2, grade,
                                  coeffs) * coeffs.mass_tonnes


#: Metres per second in one mile per hour (the MOVES bin boundaries are in mph).
_MPH_MS = 0.44704
#: MOVES braking threshold: -2 mph/s.
_BRAKE_ACCEL_MS2 = -2.0 * _MPH_MS
#: MOVES idle threshold: 1 mph.
_IDLE_SPEED_MS = 1.0 * _MPH_MS
#: MOVES speed-bin boundaries: 25 and 50 mph.
_SPEED_BIN_LOW_MS = 25.0 * _MPH_MS
_SPEED_BIN_HIGH_MS = 50.0 * _MPH_MS

#: The MOVES operating-mode identifiers, ascending.  The gaps (26, 31, 32, 34,
#: 36) are genuine: MOVES does not define those bins.
OPMODE_IDS = (0, 1,
              11, 12, 13, 14, 15, 16,
              21, 22, 23, 24, 25, 27, 28, 29, 30,
              33, 35, 37, 38, 39, 40)

_OPMODE_LABELS = {
    0: "braking",
    1: "idle",
    11: "low speed, VSP < 0", 12: "low speed, VSP 0-3",
    13: "low speed, VSP 3-6", 14: "low speed, VSP 6-9",
    15: "low speed, VSP 9-12", 16: "low speed, VSP >= 12",
    21: "mid speed, VSP < 0", 22: "mid speed, VSP 0-3",
    23: "mid speed, VSP 3-6", 24: "mid speed, VSP 6-9",
    25: "mid speed, VSP 9-12", 27: "mid speed, VSP 12-18",
    28: "mid speed, VSP 18-24", 29: "mid speed, VSP 24-30",
    30: "mid speed, VSP >= 30",
    33: "high speed, VSP < 6", 35: "high speed, VSP 6-12",
    37: "high speed, VSP 12-18", 38: "high speed, VSP 18-24",
    39: "high speed, VSP 24-30", 40: "high speed, VSP >= 30",
}

#: Representative VSP (kW/tonne) at the centre of each operating-mode bin.
_OPMODE_VSP_MID = {
    0: 0.0, 1: 0.0,
    11: 0.0, 12: 1.5, 13: 4.5, 14: 7.5, 15: 10.5, 16: 14.0,
    21: 0.0, 22: 1.5, 23: 4.5, 24: 7.5, 25: 10.5,
    27: 15.0, 28: 21.0, 29: 27.0, 30: 33.0,
    33: 3.0, 35: 9.0, 37: 15.0, 38: 21.0, 39: 27.0, 40: 33.0,
}


def operating_mode_bin(speed_ms: float, accel_ms2: float, vsp: float) -> int:
    """MOVES operating-mode identifier for one second of a speed-time trace."""
    v = _non_negative(speed_ms, "speed_ms")
    a = _as_float(accel_ms2, "accel_ms2")
    p = _as_float(vsp, "vsp")
    if a <= _BRAKE_ACCEL_MS2:
        return 0
    if v < _IDLE_SPEED_MS:
        return 1
    if v < _SPEED_BIN_LOW_MS:
        if p < 0.0:
            return 11
        if p < 3.0:
            return 12
        if p < 6.0:
            return 13
        if p < 9.0:
            return 14
        if p < 12.0:
            return 15
        return 16
    if v < _SPEED_BIN_HIGH_MS:
        if p < 0.0:
            return 21
        if p < 3.0:
            return 22
        if p < 6.0:
            return 23
        if p < 9.0:
            return 24
        if p < 12.0:
            return 25
        if p < 18.0:
            return 27
        if p < 24.0:
            return 28
        if p < 30.0:
            return 29
        return 30
    if p < 6.0:
        return 33
    if p < 12.0:
        return 35
    if p < 18.0:
        return 37
    if p < 24.0:
        return 38
    if p < 30.0:
        return 39
    return 40


def opmode_label(opmode: int) -> str:
    """Human-readable description of a MOVES operating-mode identifier."""
    key = int(opmode)
    if key not in _OPMODE_LABELS:
        raise EmissionsError("unknown MOVES operating mode %d." % key)
    return _OPMODE_LABELS[key]


def opmode_vsp_midpoint(opmode: int) -> float:
    """Representative VSP in kW/tonne at the centre of an operating-mode bin."""
    key = int(opmode)
    if key not in _OPMODE_VSP_MID:
        raise EmissionsError("unknown MOVES operating mode %d." % key)
    return _OPMODE_VSP_MID[key]


#: Representative light-duty bin rates as ``(idle rate g/s, slope g/s per
#: kW/tonne)``.  MOVES ships a measured rate for every (source type, model
#: year, operating mode) triple; this module carries a single VSP-linear
#: surrogate calibrated to reproduce the COPERT curve at motorway cruise.
#: Replace it with a MOVES export for production work.
MOVES_RATE_PARAMS = {
    "CO2": (0.55, 0.42),
    "NOx": (0.00025, 0.00010),
    "PM10": (0.0000030, 0.0000025),
    "PM2.5": (0.00000285, 0.000002375),
    "CO": (0.0020, 0.0011),
    "HC": (0.00030, 0.00013),
}

#: Braking burns slightly more than idle (the engine is still turning over).
_BRAKING_IDLE_MULTIPLIER = 1.15


def moves_opmode_rate(pollutant: str, opmode: int,
                      params: Optional[Dict[str, Tuple[float, float]]] = None
                      ) -> float:
    """Emission rate in g/s for one vehicle in a MOVES operating-mode bin."""
    pol = _key("pollutant", pollutant, POLLUTANTS)
    table = MOVES_RATE_PARAMS if params is None else params
    if pol not in table:
        raise EmissionsError("no MOVES rate parameters for pollutant %r." % pol)
    base, slope = table[pol]
    base = _non_negative(base, "idle rate for %s" % pol)
    slope = _non_negative(slope, "VSP slope for %s" % pol)
    mid = opmode_vsp_midpoint(opmode)
    rate = base + slope * (mid if mid > 0.0 else 0.0)
    if int(opmode) == 0:
        rate = base * _BRAKING_IDLE_MULTIPLIER
    return rate


@dataclass
class OpModeDistribution:
    """Seconds spent in each MOVES operating mode over a trace."""

    seconds: Dict[int, float]
    distance_m: float = 0.0
    name: str = "trace"

    def __post_init__(self) -> None:
        clean = {}  # type: Dict[int, float]
        for key in sorted(self.seconds.keys()):
            k = int(key)
            if k not in _OPMODE_LABELS:
                raise EmissionsError("unknown MOVES operating mode %d." % k)
            clean[k] = clean.get(k, 0.0) + _non_negative(
                self.seconds[key], "seconds in mode %d" % k)
        self.seconds = clean

    def bins(self) -> List[int]:
        """Occupied bins in ascending order -- never a set iteration."""
        return sorted(self.seconds.keys())

    def total_seconds(self) -> float:
        """Total time covered by the distribution."""
        return math.fsum([self.seconds[k] for k in self.bins()])

    def fractions(self) -> Dict[int, float]:
        """Time share per bin; an empty distribution gives an empty dict."""
        total = self.total_seconds()
        if total == 0.0:
            return {}
        return {k: self.seconds[k] / total for k in self.bins()}

    def idle_seconds(self) -> float:
        """Seconds in the idle bin."""
        return self.seconds.get(1, 0.0)

    def braking_seconds(self) -> float:
        """Seconds in the braking bin."""
        return self.seconds.get(0, 0.0)

    def summary(self) -> Dict[str, Any]:
        """JSON-able description of the operating-mode split."""
        frac = self.fractions()
        return {"name": self.name,
                "totalSeconds": self.total_seconds(),
                "distanceM": self.distance_m,
                "idleSeconds": self.idle_seconds(),
                "brakingSeconds": self.braking_seconds(),
                "bins": [{"opMode": k, "label": opmode_label(k),
                          "seconds": self.seconds[k],
                          "fraction": frac.get(k, 0.0),
                          "vspMid": opmode_vsp_midpoint(k)}
                         for k in self.bins()]}


# ===========================================================================
# 5. SPEED TRACES, DRIVE CYCLES AND MODAL EMISSIONS   -- units here are SI
# ===========================================================================
@dataclass
class SpeedTrace:
    """A second-by-second speed-time trace in SI units (seconds, m/s)."""

    times: List[float]
    speeds: List[float]
    grades: Optional[List[float]] = None
    name: str = "trace"

    def __post_init__(self) -> None:
        self.times = _as_floats(self.times, "times")
        self.speeds = _as_floats(self.speeds, "speeds")
        _check_len(self.times, 2, "times")
        if len(self.speeds) != len(self.times):
            raise EmissionsError("times and speeds must be the same length "
                                 "(%d vs %d)." % (len(self.times),
                                                  len(self.speeds)))
        for i in range(1, len(self.times)):
            if self.times[i] <= self.times[i - 1]:
                raise EmissionsError("times must be strictly increasing "
                                     "(index %d)." % i)
        for i, v in enumerate(self.speeds):
            if v < 0.0:
                raise EmissionsError("speeds must be non-negative "
                                     "(index %d, %g m/s)." % (i, v))
        if self.grades is None:
            self.grades = [0.0] * len(self.times)
        else:
            self.grades = _as_floats(self.grades, "grades")
            if len(self.grades) != len(self.times):
                raise EmissionsError("grades must match the trace length "
                                     "(%d vs %d)." % (len(self.grades),
                                                      len(self.times)))

    def __len__(self) -> int:
        return len(self.times)

    def accelerations(self) -> List[float]:
        """Backward-difference accelerations in m/s^2, with a[0] copied from a[1]."""
        n = len(self.times)
        acc = [0.0] * n
        for i in range(1, n):
            acc[i] = (self.speeds[i] - self.speeds[i - 1]) / \
                (self.times[i] - self.times[i - 1])
        acc[0] = acc[1]
        return acc

    def intervals(self) -> List[float]:
        """Duration in seconds attributed to each sample (the last one is zero)."""
        n = len(self.times)
        dt = [0.0] * n
        for i in range(n - 1):
            dt[i] = self.times[i + 1] - self.times[i]
        return dt

    def duration_s(self) -> float:
        """Total trace duration in seconds."""
        return self.times[-1] - self.times[0]

    def distance_m(self) -> float:
        """Distance travelled in metres by trapezoidal integration of speed."""
        terms = []
        for i in range(len(self.times) - 1):
            dt = self.times[i + 1] - self.times[i]
            terms.append(0.5 * (self.speeds[i] + self.speeds[i + 1]) * dt)
        return math.fsum(terms)

    def mean_speed_kmh(self) -> float:
        """Journey mean speed including stopped time, in km/h."""
        dur = self.duration_s()
        if dur == 0.0:
            return 0.0
        return ms_to_kmh(self.distance_m() / dur)

    def idle_time_s(self, threshold_ms: float = _IDLE_SPEED_MS) -> float:
        """Seconds with an interval mean speed below the idle threshold."""
        thr = _non_negative(threshold_ms, "threshold_ms")
        terms = []
        for i in range(len(self.times) - 1):
            mid = 0.5 * (self.speeds[i] + self.speeds[i + 1])
            if mid < thr:
                terms.append(self.times[i + 1] - self.times[i])
        return math.fsum(terms)

    def running_time_s(self, threshold_ms: float = _IDLE_SPEED_MS) -> float:
        """Seconds not counted as idling."""
        return self.duration_s() - self.idle_time_s(threshold_ms)

    def mean_running_speed_kmh(self,
                               threshold_ms: float = _IDLE_SPEED_MS) -> float:
        """Mean speed over the moving portion of the trace only, in km/h."""
        run = self.running_time_s(threshold_ms)
        if run <= 0.0:
            return 0.0
        return ms_to_kmh(self.distance_m() / run)

    def stop_count(self, stop_ms: float = _IDLE_SPEED_MS,
                   restart_ms: float = 5.0 * _MPH_MS) -> int:
        """Count complete stop-and-restart events with a hysteresis band."""
        low = _non_negative(stop_ms, "stop_ms")
        high = _non_negative(restart_ms, "restart_ms")
        if high <= low:
            raise EmissionsError("restart_ms must exceed stop_ms (%g <= %g)."
                                 % (high, low))
        stops = 0
        stopped = self.speeds[0] < low
        for v in self.speeds[1:]:
            if not stopped and v < low:
                stopped = True
                stops += 1
            elif stopped and v > high:
                stopped = False
        return stops

    def stops_per_km(self, stop_ms: float = _IDLE_SPEED_MS,
                     restart_ms: float = 5.0 * _MPH_MS) -> float:
        """Stops per kilometre travelled (zero for a zero-length trace)."""
        dist_km = self.distance_m() / 1000.0
        if dist_km == 0.0:
            return 0.0
        return self.stop_count(stop_ms, restart_ms) / dist_km

    def max_speed_kmh(self) -> float:
        """Peak speed reached, in km/h."""
        return ms_to_kmh(max(self.speeds))

    def positive_kinetic_energy(self) -> float:
        """PKE in m/s^2: the sum of positive v^2 increments over the distance."""
        dist = self.distance_m()
        if dist == 0.0:
            return 0.0
        terms = []
        for i in range(1, len(self.speeds)):
            d = self.speeds[i] ** 2 - self.speeds[i - 1] ** 2
            if d > 0.0:
                terms.append(d)
        return math.fsum(terms) / dist

    def summary(self) -> Dict[str, Any]:
        """JSON-able description of the trace."""
        return {"name": self.name,
                "samples": len(self.times),
                "durationS": self.duration_s(),
                "distanceKm": self.distance_m() / 1000.0,
                "meanSpeedKmh": self.mean_speed_kmh(),
                "meanRunningSpeedKmh": self.mean_running_speed_kmh(),
                "maxSpeedKmh": self.max_speed_kmh(),
                "idleTimeS": self.idle_time_s(),
                "stops": self.stop_count(),
                "stopsPerKm": self.stops_per_km(),
                "pkeMs2": self.positive_kinetic_energy()}


def trace_from_speeds(speeds_ms: Sequence[float], dt: float = 1.0,
                      grades: Optional[Sequence[float]] = None,
                      name: str = "trace") -> SpeedTrace:
    """Build a uniformly sampled ``SpeedTrace`` from a list of speeds in m/s."""
    vals = _as_floats(speeds_ms, "speeds_ms")
    _check_len(vals, 2, "speeds_ms")
    step = _positive(dt, "dt")
    times = [step * i for i in range(len(vals))]
    gr = None if grades is None else _as_floats(grades, "grades")
    return SpeedTrace(times=times, speeds=vals, grades=gr, name=name)


def constant_speed_cycle(speed_kmh: float, duration_s: float, dt: float = 1.0,
                         grade: float = 0.0,
                         name: str = "constant") -> SpeedTrace:
    """A flat cruise at a constant speed -- the closed-form VSP test case."""
    v = _non_negative(kmh_to_ms(speed_kmh), "speed")
    dur = _positive(duration_s, "duration_s")
    step = _positive(dt, "dt")
    n = int(round(dur / step)) + 1
    if n < 2:
        n = 2
    speeds = [v] * n
    grades = [_as_float(grade, "grade")] * n
    return SpeedTrace(times=[step * i for i in range(n)], speeds=speeds,
                      grades=grades, name=name)


def stop_go_cycle(cruise_kmh: float, n_stops: int, cruise_s: float = 40.0,
                  idle_s: float = 20.0, accel_ms2: float = 1.2,
                  decel_ms2: float = 1.5, dt: float = 1.0,
                  name: str = "stop-go") -> SpeedTrace:
    """A deterministic accelerate / cruise / decelerate / idle repeating cycle.

    A trace sampled every ``dt`` seconds cannot represent a dwell shorter than
    one sample, so a POSITIVE ``idle_s`` below ``dt`` is rounded up to a single
    sample rather than vanishing -- the accelerate / cruise / decelerate / idle
    structure has to survive.  ``idle_s=0.0`` is a different request and is
    honoured exactly: no dwell at all, the decelerate ramp running straight
    into the next acceleration.  (That is why ``idle_s`` is validated as
    non-negative while ``cruise_s`` is validated as strictly positive.)
    """
    v_cruise = _non_negative(kmh_to_ms(cruise_kmh), "cruise_kmh")
    stops = int(n_stops)
    if stops < 0:
        raise EmissionsError("n_stops must be non-negative (got %d)." % stops)
    cruise = _positive(cruise_s, "cruise_s")
    idle = _non_negative(idle_s, "idle_s")
    acc = _positive(accel_ms2, "accel_ms2")
    dec = _positive(decel_ms2, "decel_ms2")
    step = _positive(dt, "dt")
    if stops == 0:
        return constant_speed_cycle(cruise_kmh, cruise, step, 0.0, name)

    speeds = [0.0]
    for _ in range(stops):
        v = speeds[-1]
        while v < v_cruise:                         # acceleration ramp
            v = min(v_cruise, v + acc * step)
            speeds.append(v)
        n_cruise = int(round(cruise / step))
        for _i in range(max(1, n_cruise)):          # cruise plateau
            speeds.append(v_cruise)
        v = v_cruise
        while v > 0.0:                              # deceleration ramp
            v = max(0.0, v - dec * step)
            speeds.append(v)
        n_idle = int(round(idle / step))
        if idle > 0.0 and n_idle < 1:               # sub-sample dwell requested
            n_idle = 1
        for _i in range(n_idle):                    # idle dwell (0 is legal)
            speeds.append(0.0)
    return trace_from_speeds(speeds, step, None, name)


def synthetic_drive_cycle(seed: int, duration_s: float = 600.0,
                          dt: float = 1.0, target_mean_kmh: float = 30.0,
                          max_kmh: float = 60.0, idle_fraction: float = 0.15,
                          accel_sigma: float = 0.55,
                          name: str = "synthetic") -> SpeedTrace:
    """A seeded urban drive cycle -- byte-identical for a given seed.

    ``idle_fraction`` is a SHAPING knob on the probability of dropping into a
    dwell, not the achieved idle share: at 0.15 the generator lands near 0.14,
    at 0.5 near 0.27.  It cannot reach zero either, because the cycle opens
    with an unconditional three-sample standstill (3 of 601 samples at the
    defaults, hence the 0.5 % floor).  ``max_kmh`` is a hard clamp, but the
    Ornstein-Uhlenbeck process pulling towards ``target_mean_kmh`` rarely comes
    within 20 km/h of the 60 km/h default, so the clamp only bites when it is
    lowered towards the target.
    """
    rng = random.Random(int(seed))
    dur = _positive(duration_s, "duration_s")
    step = _positive(dt, "dt")
    v_target = _non_negative(kmh_to_ms(target_mean_kmh), "target_mean_kmh")
    v_max = _positive(kmh_to_ms(max_kmh), "max_kmh")
    idle_f = _fraction(idle_fraction, "idle_fraction")
    sigma = _positive(accel_sigma, "accel_sigma")
    n = int(round(dur / step)) + 1
    if n < 2:
        n = 2
    speeds = [0.0]
    v = 0.0
    idling = True
    idle_left = 3
    for _i in range(n - 1):
        if idling:
            idle_left -= 1
            if idle_left <= 0:
                idling = False
            speeds.append(0.0)
            continue
        # Ornstein-Uhlenbeck pull towards the target cruise speed.
        drift = 0.30 * (v_target - v)
        shock = sigma * rng.normalvariate(0.0, 1.0)
        v = v + (drift + shock) * step
        if v < 0.0:
            v = 0.0
        elif v > v_max:
            v = v_max
        if v == 0.0 or rng.random() < idle_f * 0.05:
            idling = True
            idle_left = 5 + int(rng.random() * 15.0)
            v = 0.0
        speeds.append(v)
    return trace_from_speeds(speeds, step, None, name)


def opmode_distribution(trace: SpeedTrace,
                        coeffs: VSPCoefficients = LIGHT_DUTY_VSP
                        ) -> OpModeDistribution:
    """Seconds spent in each MOVES operating mode over a speed-time trace."""
    if not isinstance(trace, SpeedTrace):
        raise EmissionsError("trace must be a SpeedTrace.")
    acc = trace.accelerations()
    dts = trace.intervals()
    buckets = {}  # type: Dict[int, float]
    for i in range(len(trace) - 1):
        dt = dts[i]
        if dt == 0.0:
            continue
        v = trace.speeds[i]
        a = acc[i]
        grade = trace.grades[i] if trace.grades is not None else 0.0
        vsp = vehicle_specific_power(v, a, grade, coeffs)
        b = operating_mode_bin(v, a, vsp)
        buckets[b] = buckets.get(b, 0.0) + dt
    return OpModeDistribution(seconds=buckets, distance_m=trace.distance_m(),
                              name=trace.name)


@dataclass
class ModalResult:
    """Second-by-second modal emission totals for one trace."""

    grams: Dict[str, float]
    distance_km: float
    duration_s: float
    mean_speed_kmh: float
    opmodes: OpModeDistribution
    pollutants: Tuple[str, ...] = POLLUTANTS
    name: str = "trace"

    def grams_per_km(self) -> Dict[str, float]:
        """Modal emission factors in g/km implied by the trace."""
        if self.distance_km == 0.0:
            return {p: 0.0 for p in self.pollutants}
        return {p: self.grams[p] / self.distance_km for p in self.pollutants}

    def grams_per_hour(self) -> Dict[str, float]:
        """Modal emission rates in g/h implied by the trace."""
        if self.duration_s == 0.0:
            return {p: 0.0 for p in self.pollutants}
        return {p: self.grams[p] * 3600.0 / self.duration_s
                for p in self.pollutants}

    def summary(self) -> Dict[str, Any]:
        """JSON-able description of the modal run."""
        per_km = self.grams_per_km()
        return {"name": self.name,
                "distanceKm": self.distance_km,
                "durationS": self.duration_s,
                "meanSpeedKmh": self.mean_speed_kmh,
                "pollutants": [{"pollutant": p, "grams": self.grams[p],
                                "gPerKm": per_km[p]}
                               for p in self.pollutants],
                "opModes": self.opmodes.summary()}


def modal_emissions(trace: SpeedTrace,
                    pollutants: Sequence[str] = POLLUTANTS,
                    coeffs: VSPCoefficients = LIGHT_DUTY_VSP,
                    rate_params: Optional[Dict[str, Tuple[float, float]]] = None,
                    rate_scale: float = 1.0) -> ModalResult:
    """Integrate MOVES operating-mode rates over a trace to modal masses."""
    if not isinstance(trace, SpeedTrace):
        raise EmissionsError("trace must be a SpeedTrace.")
    pols = tuple(_key("pollutant", p, POLLUTANTS) for p in pollutants)
    _check_len(pols, 1, "pollutants")
    scale = _non_negative(rate_scale, "rate_scale")
    dist = opmode_distribution(trace, coeffs)
    grams = {}  # type: Dict[str, float]
    bins = dist.bins()
    for pol in pols:
        terms = [moves_opmode_rate(pol, b, rate_params) * dist.seconds[b]
                 for b in bins]
        grams[pol] = math.fsum(terms) * scale
    return ModalResult(grams=grams, distance_km=trace.distance_m() / 1000.0,
                       duration_s=trace.duration_s(),
                       mean_speed_kmh=trace.mean_speed_kmh(),
                       opmodes=dist, pollutants=pols, name=trace.name)


# ===========================================================================
# 6. FUEL CONSUMPTION -- THE AKCELIK-BIGGS ELEMENTAL MODEL   -- SI units
# ===========================================================================
@dataclass(frozen=True)
class AkcelikBiggsParameters:
    """Akcelik-Biggs four-mode elemental fuel model parameters for a vehicle."""

    alpha_ml_s: float       #: idle fuel rate, mL/s
    beta1_ml_kj: float      #: mL of fuel per kJ of tractive energy
    beta2_ml_kj: float      #: mL per kJ of the extra inertial acceleration term
    mass_kg: float          #: vehicle mass including a nominal load
    rolling_n: float        #: b1, rolling resistance force in newtons
    drag_n_s2_m2: float     #: b2, 0.5 rho Cd A in N s^2/m^2
    fuel: str = "petrol"
    name: str = "light vehicle"

    def __post_init__(self) -> None:
        if self.alpha_ml_s < 0.0:
            raise EmissionsError("alpha_ml_s must be non-negative.")
        if self.mass_kg <= 0.0:
            raise EmissionsError("mass_kg must be positive.")
        if self.fuel not in FUEL_PROPERTIES:
            raise EmissionsError("unknown fuel %r." % self.fuel)


#: A 1.7 tonne petrol car -- the reference light vehicle.
PETROL_CAR = AkcelikBiggsParameters(0.35, 0.0717, 0.0344, 1700.0, 216.8,
                                    0.4224, "petrol", "petrol car 1.7 t")
#: A 1.75 tonne diesel car.
DIESEL_CAR = AkcelikBiggsParameters(0.28, 0.0600, 0.0300, 1750.0, 223.2,
                                    0.4400, "diesel", "diesel car 1.75 t")
#: A 15 tonne urban bus.
URBAN_BUS = AkcelikBiggsParameters(1.60, 0.0790, 0.0380, 15000.0, 1500.0,
                                   3.6000, "diesel", "urban bus 15 t")


def tractive_power(speed_ms: float, accel_ms2: float = 0.0,
                   grade: float = 0.0,
                   params: AkcelikBiggsParameters = PETROL_CAR) -> float:
    """Total tractive power in kW: (rolling + drag + inertia + grade) times speed."""
    v = _non_negative(speed_ms, "speed_ms")
    a = _as_float(accel_ms2, "accel_ms2")
    theta = _as_float(grade, "grade")
    if not isinstance(params, AkcelikBiggsParameters):
        raise EmissionsError("params must be AkcelikBiggsParameters.")
    force = (params.rolling_n
             + params.drag_n_s2_m2 * v * v
             + params.mass_kg * a
             + params.mass_kg * _G * theta)
    return force * v / 1000.0


def instantaneous_fuel_rate(speed_ms: float, accel_ms2: float = 0.0,
                            grade: float = 0.0,
                            params: AkcelikBiggsParameters = PETROL_CAR
                            ) -> float:
    """Akcelik-Biggs instantaneous fuel rate in mL/s from the power demand.

    The two power terms are gated on DIFFERENT quantities, deliberately and
    asymmetrically: ``beta1`` multiplies ``max(0, P_T)``, so it switches off
    whenever total tractive power is negative, while ``beta2`` -- the extra
    inertial-energy penalty -- is gated on the sign of the ACCELERATION alone.
    A vehicle accelerating down a grade steep enough to make ``P_T`` negative
    therefore still pays the ``beta2`` term and burns more than idle (1.84x idle
    at 20 m/s, a = 0.5 m/s2, -20 % grade).  The suite pins that so the
    behaviour is deliberate; change it only against the paper, not by
    inspection.
    """
    v = _non_negative(speed_ms, "speed_ms")
    a = _as_float(accel_ms2, "accel_ms2")
    if v < _IDLE_SPEED_MS:
        return params.alpha_ml_s
    power = tractive_power(v, a, grade, params)
    rate = params.alpha_ml_s + params.beta1_ml_kj * (power if power > 0.0 else 0.0)
    if a > 0.0:
        rate += params.beta2_ml_kj * params.mass_kg * a * a * v / 1000.0
    return rate


#: Acceleration band, in m/s^2, inside which a moving vehicle counts as cruising.
_CRUISE_ACCEL_BAND = 0.1


def four_mode_fuel(speed_ms: float, accel_ms2: float = 0.0,
                   grade: float = 0.0,
                   params: AkcelikBiggsParameters = PETROL_CAR
                   ) -> Tuple[str, float]:
    """Classify an instant as idle / cruise / accelerate / decelerate and rate it."""
    v = _non_negative(speed_ms, "speed_ms")
    a = _as_float(accel_ms2, "accel_ms2")
    rate = instantaneous_fuel_rate(v, a, grade, params)
    if v < _IDLE_SPEED_MS:
        return "idle", rate
    if a > _CRUISE_ACCEL_BAND:
        return "accelerate", rate
    if a < -_CRUISE_ACCEL_BAND:
        return "decelerate", rate
    return "cruise", rate


@dataclass
class FuelResult:
    """Fuel use in litres split across the four elemental modes."""

    idle_litres: float
    cruise_litres: float
    accelerate_litres: float
    decelerate_litres: float
    distance_km: float
    duration_s: float
    fuel: str = "petrol"
    name: str = "trace"

    @property
    def total_litres(self) -> float:
        """Total fuel consumed, in litres."""
        return math.fsum([self.idle_litres, self.cruise_litres,
                          self.accelerate_litres, self.decelerate_litres])

    @property
    def litres_per_100km(self) -> float:
        """Fuel intensity in L/100 km (zero over a zero-length trip)."""
        if self.distance_km == 0.0:
            return 0.0
        return 100.0 * self.total_litres / self.distance_km

    @property
    def co2_grams(self) -> float:
        """Tank-to-wheel CO2 in grams implied by the fuel burnt."""
        return fuel_litres_to_co2_g(self.total_litres, self.fuel)

    def energy_mj(self) -> float:
        """Fuel energy consumed in MJ, from density and calorific value."""
        props = FUEL_PROPERTIES[self.fuel]
        return self.total_litres * props["density_kg_l"] * props["lhv_mj_kg"]

    def summary(self) -> Dict[str, Any]:
        """JSON-able description of the fuel split."""
        return {"name": self.name, "fuel": self.fuel,
                "totalLitres": self.total_litres,
                "idleLitres": self.idle_litres,
                "cruiseLitres": self.cruise_litres,
                "accelerateLitres": self.accelerate_litres,
                "decelerateLitres": self.decelerate_litres,
                "distanceKm": self.distance_km,
                "durationS": self.duration_s,
                "litresPer100Km": self.litres_per_100km,
                "co2Grams": self.co2_grams,
                "energyMJ": self.energy_mj()}


def fuel_from_trace(trace: SpeedTrace,
                    params: AkcelikBiggsParameters = PETROL_CAR) -> FuelResult:
    """Integrate the Akcelik-Biggs rate over a trace, split by elemental mode."""
    if not isinstance(trace, SpeedTrace):
        raise EmissionsError("trace must be a SpeedTrace.")
    acc = trace.accelerations()
    dts = trace.intervals()
    pots = {"idle": [], "cruise": [], "accelerate": [],
            "decelerate": []}  # type: Dict[str, List[float]]
    for i in range(len(trace) - 1):
        dt = dts[i]
        if dt == 0.0:
            continue
        grade = trace.grades[i] if trace.grades is not None else 0.0
        mode, rate = four_mode_fuel(trace.speeds[i], acc[i], grade, params)
        pots[mode].append(rate * dt)
    to_l = 0.001                                  # millilitres -> litres
    return FuelResult(idle_litres=math.fsum(pots["idle"]) * to_l,
                      cruise_litres=math.fsum(pots["cruise"]) * to_l,
                      accelerate_litres=math.fsum(pots["accelerate"]) * to_l,
                      decelerate_litres=math.fsum(pots["decelerate"]) * to_l,
                      distance_km=trace.distance_m() / 1000.0,
                      duration_s=trace.duration_s(),
                      fuel=params.fuel, name=trace.name)


def elemental_fuel_model(distance_km: float, idle_time_s: float,
                         stops: float, cruise_kmh: float = 50.0,
                         accel_ms2: float = 1.2,
                         params: AkcelikBiggsParameters = PETROL_CAR,
                         name: str = "elemental") -> FuelResult:
    """Akcelik's running model F = f1 d + f2 (idle time) + f3 (stops), in litres."""
    dist = _non_negative(distance_km, "distance_km")
    idle_s = _non_negative(idle_time_s, "idle_time_s")
    n_stops = _non_negative(stops, "stops")
    v = _positive(kmh_to_ms(cruise_kmh), "cruise_kmh")
    acc = _positive(accel_ms2, "accel_ms2")
    # f1: millilitres per kilometre at the cruise speed.
    f1 = instantaneous_fuel_rate(v, 0.0, 0.0, params) * 1000.0 / v
    # f2: millilitres per second of idling.
    f2 = params.alpha_ml_s
    # f3: millilitres of excess fuel for one full stop-start cycle -- the
    # kinetic energy thrown away, recovered at the beta1 efficiency, plus the
    # beta2 acceleration penalty accrued over the ramp back up to speed.
    kinetic_kj = 0.5 * params.mass_kg * v * v / 1000.0
    f3 = kinetic_kj * (params.beta1_ml_kj + params.beta2_ml_kj * acc)
    to_l = 0.001
    return FuelResult(idle_litres=f2 * idle_s * to_l,
                      cruise_litres=f1 * dist * to_l,
                      accelerate_litres=f3 * n_stops * to_l,
                      decelerate_litres=0.0,
                      distance_km=dist,
                      duration_s=idle_s + (dist * 1000.0 / v),
                      fuel=params.fuel, name=name)


# ===========================================================================
# 7. THE LINEAR EDGE PROXY AND ITS CALIBRATION
# ===========================================================================
@dataclass(frozen=True)
class EdgeProxyCoefficients:
    """The white paper's edge proxy coefficients for E = a f t + b idle + c stop."""

    a: float                #: grams per vehicle-hour of travel
    b: float                #: grams per vehicle-hour of idling
    c: float                #: grams per stop
    pollutant: str = "CO2"
    source: str = "uncalibrated"

    def as_tuple(self) -> Tuple[float, float, float]:
        """The three coefficients in (a, b, c) order."""
        return (self.a, self.b, self.c)

    def evaluate(self, flow_veh_h: float, travel_time_h: float,
                 idle_veh_h: float, stops: float) -> float:
        """Proxy emissions in grams -- exactly linear in each of its three terms."""
        return edge_emission_proxy(self, flow_veh_h, travel_time_h,
                                   idle_veh_h, stops)

    def summary(self) -> Dict[str, Any]:
        """JSON-able description of the coefficients."""
        return {"pollutant": self.pollutant, "aGPerVehHour": self.a,
                "bGPerIdleVehHour": self.b, "cGPerStop": self.c,
                "source": self.source}


@dataclass
class EdgeObservation:
    """One calibration sample: an edge's traffic state and its modal emissions."""

    edge_id: str
    flow_veh_h: float
    travel_time_h: float
    idle_veh_h: float
    stops: float
    modal_grams: float = 0.0

    def __post_init__(self) -> None:
        self.edge_id = str(self.edge_id)
        self.flow_veh_h = _non_negative(self.flow_veh_h, "flow_veh_h")
        self.travel_time_h = _non_negative(self.travel_time_h, "travel_time_h")
        self.idle_veh_h = _non_negative(self.idle_veh_h, "idle_veh_h")
        self.stops = _non_negative(self.stops, "stops")
        self.modal_grams = _non_negative(self.modal_grams, "modal_grams")

    def design_row(self) -> Tuple[float, float, float]:
        """The regressor row (f t, idle, stops) for the least-squares fit."""
        return (self.flow_veh_h * self.travel_time_h, self.idle_veh_h,
                self.stops)

    def summary(self) -> Dict[str, Any]:
        """JSON-able description of the observation."""
        return {"edgeId": self.edge_id, "flowVehH": self.flow_veh_h,
                "travelTimeH": self.travel_time_h,
                "idleVehH": self.idle_veh_h, "stops": self.stops,
                "modalGrams": self.modal_grams}


def _observation_key(o: "EdgeObservation") -> Tuple[Any, ...]:
    """Total, VALUE-based sort key for an observation.

    The obvious tie-break -- ``(edge_id, id(obs))`` -- is not a tie-break at
    all: ``id`` is a memory address, so two observations that share an edge id
    could be summed in a different order in the next process and shift the
    last bits of an ``fsum``.  Everything here is a stored value.
    """
    return (o.edge_id, o.flow_veh_h, o.travel_time_h, o.idle_veh_h, o.stops,
            o.modal_grams)


def edge_emission_proxy(coeffs: EdgeProxyCoefficients, flow_veh_h: float,
                        travel_time_h: float, idle_veh_h: float,
                        stops: float) -> float:
    """Evaluate E = a f t + b idle + c stop in grams for one edge."""
    if not isinstance(coeffs, EdgeProxyCoefficients):
        raise EmissionsError("coeffs must be EdgeProxyCoefficients.")
    f = _non_negative(flow_veh_h, "flow_veh_h")
    t = _non_negative(travel_time_h, "travel_time_h")
    idle = _non_negative(idle_veh_h, "idle_veh_h")
    st = _non_negative(stops, "stops")
    return math.fsum([coeffs.a * f * t, coeffs.b * idle, coeffs.c * st])


def network_emission_proxy(coeffs: EdgeProxyCoefficients,
                           observations: Sequence[EdgeObservation]) -> float:
    """Total proxy emissions in grams over a network, summed in edge-id order."""
    obs = list(observations)
    _check_len(obs, 1, "observations")
    ordered = sorted(obs, key=_observation_key)
    return math.fsum([edge_emission_proxy(coeffs, o.flow_veh_h,
                                          o.travel_time_h, o.idle_veh_h,
                                          o.stops) for o in ordered])


@dataclass
class ProxyCalibration:
    """Least-squares fit of the edge proxy against modal emission targets."""

    coefficients: EdgeProxyCoefficients
    r_squared: float
    rmse: float
    mean_abs_error: float
    residuals: List[float]
    n_observations: int
    active_terms: Tuple[int, ...]

    def summary(self) -> Dict[str, Any]:
        """JSON-able description of the calibration."""
        return {"coefficients": self.coefficients.summary(),
                "rSquared": self.r_squared, "rmse": self.rmse,
                "meanAbsError": self.mean_abs_error,
                "nObservations": self.n_observations,
                "activeTerms": list(self.active_terms),
                "maxAbsResidual": max([abs(r) for r in self.residuals])
                if self.residuals else 0.0}


def _least_squares_sse(rows: List[Tuple[float, float, float]],
                       targets: List[float],
                       beta: Tuple[float, float, float]) -> float:
    """Sum of squared errors of a candidate coefficient triple."""
    terms = []
    for row, y in zip(rows, targets):
        pred = math.fsum([beta[0] * row[0], beta[1] * row[1], beta[2] * row[2]])
        terms.append((pred - y) ** 2)
    return math.fsum(terms)


def calibrate_edge_proxy(observations: Sequence[EdgeObservation],
                         pollutant: str = "CO2", non_negative: bool = False,
                         ridge: float = 0.0) -> ProxyCalibration:
    """Fit (a, b, c) by least squares against modal emissions, optionally signed.

    ``travel_time_h`` is the *total* time on the edge, idling included, so ``b``
    is the **excess** grams per vehicle-hour of idling over and above the
    running term and is routinely negative (an idling engine burns less than a
    cruising one).  Pass ``non_negative=True`` only when a downstream solver
    insists on a sign-constrained surrogate; the fit is then an exact
    non-negative least squares over the eight supports.
    """
    obs = sorted(list(observations), key=_observation_key)
    _check_len(obs, 3, "observations")
    pol = _key("pollutant", pollutant, POLLUTANTS)
    lam = _non_negative(ridge, "ridge")
    rows = [o.design_row() for o in obs]
    targets = [o.modal_grams for o in obs]

    # -- normal equations ---------------------------------------------------
    xtx = [[0.0] * 3 for _ in range(3)]
    xty = [0.0] * 3
    for i in range(3):
        xty[i] = math.fsum([rows[k][i] * targets[k] for k in range(len(rows))])
        for j in range(3):
            xtx[i][j] = math.fsum([rows[k][i] * rows[k][j]
                                   for k in range(len(rows))])
    if lam > 0.0:
        # Ridge scaled to the mean diagonal so the penalty is dimensionless
        # relative to the data -- never an absolute constant.
        diag_scale = math.fsum([xtx[i][i] for i in range(3)]) / 3.0
        for i in range(3):
            xtx[i][i] += lam * diag_scale

    # -- exhaustive non-negative least squares over the 8 supports ----------
    best_beta = (0.0, 0.0, 0.0)
    best_sse = _least_squares_sse(rows, targets, best_beta)
    best_support = ()  # type: Tuple[int, ...]
    supports = [(), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]
    for support in supports:
        if not support:
            continue
        k = len(support)
        sub_a = [[xtx[i][j] for j in support] for i in support]
        sub_b = [xty[i] for i in support]
        try:
            sol = _solve_symmetric(sub_a, sub_b)
        except EmissionsError:
            continue
        beta = [0.0, 0.0, 0.0]
        feasible = True
        for idx, pos in enumerate(support):
            beta[pos] = sol[idx]
            if non_negative and sol[idx] < 0.0:
                feasible = False
        if not feasible:
            continue
        sse = _least_squares_sse(rows, targets, (beta[0], beta[1], beta[2]))
        # Deterministic tie-break: lower SSE wins, then the smaller support,
        # then the lexicographically smaller support tuple.
        cand = (sse, k, support)
        cur = (best_sse, len(best_support), best_support)
        if cand < cur:
            best_sse = sse
            best_beta = (beta[0], beta[1], beta[2])
            best_support = support
        del k

    residuals = []
    for row, y in zip(rows, targets):
        pred = math.fsum([best_beta[0] * row[0], best_beta[1] * row[1],
                          best_beta[2] * row[2]])
        residuals.append(pred - y)
    n = len(obs)
    sse = math.fsum([r * r for r in residuals])
    ybar = math.fsum(targets) / n
    sst = math.fsum([(y - ybar) ** 2 for y in targets])
    if sst == 0.0:                       # exact-zero test, not an epsilon
        r2 = 1.0 if sse == 0.0 else 0.0
    else:
        r2 = 1.0 - sse / sst
    rmse = math.sqrt(sse / n)
    mae = math.fsum([abs(r) for r in residuals]) / n
    coeffs = EdgeProxyCoefficients(a=best_beta[0], b=best_beta[1],
                                   c=best_beta[2], pollutant=pol,
                                   source="least squares on %d observations" % n)
    return ProxyCalibration(coefficients=coeffs, r_squared=r2, rmse=rmse,
                            mean_abs_error=mae, residuals=residuals,
                            n_observations=n, active_terms=best_support)


@dataclass
class RankConsistency:
    """Kendall rank agreement between the proxy and the modal reference."""

    tau_b: float
    concordant: int
    discordant: int
    ties_proxy: int
    ties_modal: int
    n_pairs: int

    @property
    def monotone(self) -> bool:
        """True when the proxy never reverses a modal ordering."""
        return self.discordant == 0

    def summary(self) -> Dict[str, Any]:
        """JSON-able description of the rank agreement."""
        return {"tauB": self.tau_b, "concordant": self.concordant,
                "discordant": self.discordant, "tiesProxy": self.ties_proxy,
                "tiesModal": self.ties_modal, "nPairs": self.n_pairs,
                "monotone": self.monotone}


def _relative_tie(x: float, y: float) -> bool:
    """True when two values differ by less than a *relative* tolerance."""
    d = x - y
    if d == 0.0:
        return True
    scale = abs(x) + abs(y)
    return abs(d) <= _REL_TOL * scale


def proxy_rank_consistency(coeffs: EdgeProxyCoefficients,
                           observations: Sequence[EdgeObservation]
                           ) -> RankConsistency:
    """Kendall tau-b between the proxy ranking and the modal ranking of edges.

    Ties are decided by ``_relative_tie``, a RELATIVE tolerance, because both
    sequences are floating-point sums whose last bits carry no ranking
    information -- two edges whose proxies differ by an ulp are the same edge as
    far as a ranking is concerned.  ``scipy.stats.kendalltau`` ties on exact
    equality instead, so a differential test has to build its ties from
    genuinely duplicated inputs; otherwise the two implementations disagree BY
    DESIGN on near-ties and the test pins a difference that is not a defect.
    """
    obs = sorted(list(observations), key=_observation_key)
    _check_len(obs, 2, "observations")
    proxy = [edge_emission_proxy(coeffs, o.flow_veh_h, o.travel_time_h,
                                 o.idle_veh_h, o.stops) for o in obs]
    modal = [o.modal_grams for o in obs]
    n = len(obs)
    conc = 0
    disc = 0
    tp = 0
    tm = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            p_tie = _relative_tie(proxy[i], proxy[j])
            m_tie = _relative_tie(modal[i], modal[j])
            if p_tie:
                tp += 1
            if m_tie:
                tm += 1
            if p_tie or m_tie:
                continue
            same = (proxy[i] - proxy[j]) * (modal[i] - modal[j])
            if same > 0.0:
                conc += 1
            else:
                disc += 1
    n0 = n * (n - 1) // 2
    denom = math.sqrt(float(n0 - tp) * float(n0 - tm))
    tau = 0.0 if denom == 0.0 else (conc - disc) / denom
    return RankConsistency(tau_b=tau, concordant=conc, discordant=disc,
                           ties_proxy=tp, ties_modal=tm, n_pairs=n0)


# ===========================================================================
# 8. NON-EXHAUST PARTICULATE MATTER (BRAKE, TYRE AND ROAD WEAR)
# ===========================================================================
#: Total-suspended-particulate wear emission factors in g/km, keyed
#: ``category -> {"brake", "tyre", "road"}``.  Representative EMEP/EEA Tier 2
#: values; the heavy categories already carry a nominal three-axle geometry.
NON_EXHAUST_TSP = {
    "passenger_car_petrol": {"brake": 0.0075, "tyre": 0.0107, "road": 0.0150},
    "passenger_car_diesel": {"brake": 0.0075, "tyre": 0.0107, "road": 0.0150},
    "lgv_diesel": {"brake": 0.0117, "tyre": 0.0169, "road": 0.0150},
    "hgv_diesel": {"brake": 0.0294, "tyre": 0.0318, "road": 0.0760},
    "bus_diesel": {"brake": 0.0294, "tyre": 0.0227, "road": 0.0760},
    "motorcycle": {"brake": 0.0037, "tyre": 0.0046, "road": 0.0060},
}

#: Mass fraction of each wear stream below a given aerodynamic diameter.
NON_EXHAUST_SIZE_FRACTIONS = {
    "brake": {"TSP": 1.00, "PM10": 0.98, "PM2.5": 0.39},
    "tyre": {"TSP": 1.00, "PM10": 0.60, "PM2.5": 0.42},
    "road": {"TSP": 1.00, "PM10": 0.50, "PM2.5": 0.27},
}

#: Load correction factor applied to brake and tyre wear.  Heavy vehicles wear
#: their brakes and tyres roughly in proportion to the load they carry; light
#: vehicles are quoted at their reference load, so their factor is exactly one.
NON_EXHAUST_MASS_FACTOR = {
    "passenger_car_petrol": 1.00,
    "passenger_car_diesel": 1.00,
    "lgv_diesel": 1.00,
    "hgv_diesel": 1.40,
    "bus_diesel": 1.30,
    "motorcycle": 1.00,
}

_WEAR_SOURCES = ("brake", "tyre", "road")
_WEAR_SIZES = ("TSP", "PM10", "PM2.5")


def brake_wear_speed_factor(speed_kmh: float) -> float:
    """EMEP/EEA brake-wear speed correction: 1.67 below 40, 0.185 above 95 km/h."""
    v = _non_negative(speed_kmh, "speed_kmh")
    if v < 40.0:
        return 1.67
    if v > 95.0:
        return 0.185
    return -0.027 * v + 2.75


def tyre_wear_speed_factor(speed_kmh: float) -> float:
    """EMEP/EEA tyre-wear speed correction: 1.39 below 40, 0.902 above 90 km/h."""
    v = _non_negative(speed_kmh, "speed_kmh")
    if v < 40.0:
        return 1.39
    if v > 90.0:
        return 0.902
    return -0.00974 * v + 1.78


@dataclass
class NonExhaustResult:
    """Brake, tyre and road-surface wear emissions for one size fraction."""

    category: str
    size: str
    speed_kmh: float
    brake: float
    tyre: float
    road: float
    unit: str = "g/km"
    distance_km: float = 1.0
    vehicles: float = 1.0

    @property
    def total(self) -> float:
        """Sum of the three wear streams in the result's own unit."""
        return math.fsum([self.brake, self.tyre, self.road])

    def shares(self) -> Dict[str, float]:
        """Share of the total contributed by each wear stream."""
        total = self.total
        if total == 0.0:
            return {s: 0.0 for s in _WEAR_SOURCES}
        return {"brake": self.brake / total, "tyre": self.tyre / total,
                "road": self.road / total}

    def summary(self) -> Dict[str, Any]:
        """JSON-able description of the wear split."""
        sh = self.shares()
        return {"category": self.category, "size": self.size,
                "speedKmh": self.speed_kmh, "unit": self.unit,
                "brake": self.brake, "tyre": self.tyre, "road": self.road,
                "total": self.total,
                "distanceKm": self.distance_km, "vehicles": self.vehicles,
                "shares": [{"source": s, "share": sh[s]}
                           for s in _WEAR_SOURCES]}


def non_exhaust_emission_factor(category: str, speed_kmh: float,
                                size: str = "PM10",
                                load_factor: Optional[float] = None
                                ) -> NonExhaustResult:
    """Speed-corrected non-exhaust wear emission factors in g/km."""
    cat = _key("vehicle category", category, VEHICLE_CATEGORIES)
    sz = _key("size fraction", size, _WEAR_SIZES)
    v = _non_negative(speed_kmh, "speed_kmh")
    lcf = (NON_EXHAUST_MASS_FACTOR[cat] if load_factor is None
           else _positive(load_factor, "load_factor"))
    tsp = NON_EXHAUST_TSP[cat]
    brake = tsp["brake"] * brake_wear_speed_factor(v) * lcf * \
        NON_EXHAUST_SIZE_FRACTIONS["brake"][sz]
    tyre = tsp["tyre"] * tyre_wear_speed_factor(v) * lcf * \
        NON_EXHAUST_SIZE_FRACTIONS["tyre"][sz]
    road = tsp["road"] * NON_EXHAUST_SIZE_FRACTIONS["road"][sz]
    return NonExhaustResult(category=cat, size=sz, speed_kmh=v, brake=brake,
                            tyre=tyre, road=road, unit="g/km")


def non_exhaust_emissions(category: str, speed_kmh: float,
                          distance_km: float, vehicles: float = 1.0,
                          size: str = "PM10",
                          load_factor: Optional[float] = None
                          ) -> NonExhaustResult:
    """Non-exhaust wear mass in grams, exactly linear in distance and count."""
    ef = non_exhaust_emission_factor(category, speed_kmh, size, load_factor)
    dist = _non_negative(distance_km, "distance_km")
    veh = _non_negative(vehicles, "vehicles")
    scale = dist * veh
    return NonExhaustResult(category=ef.category, size=ef.size,
                            speed_kmh=ef.speed_kmh, brake=ef.brake * scale,
                            tyre=ef.tyre * scale, road=ef.road * scale,
                            unit="g", distance_km=dist, vehicles=veh)


# ===========================================================================
# 9. ACOUSTICS FUNDAMENTALS -- DECIBEL ALGEBRA, SPREADING, GROUND, BARRIERS
# ===========================================================================
def db_sum(levels: Sequence[float]) -> Decibels:
    """Incoherent energy sum of decibel levels -- exactly commutative via fsum."""
    vals = _as_floats(levels, "levels")
    _check_len(vals, 1, "levels")
    energy = math.fsum([10.0 ** (0.1 * v) for v in vals])
    if energy <= 0.0:
        return float("-inf")
    return 10.0 * math.log10(energy)


def db_subtract(total_db: float, part_db: float) -> Decibels:
    """Remove one incoherent component from a total level, in decibels."""
    total = _as_float(total_db, "total_db")
    part = _as_float(part_db, "part_db")
    energy = 10.0 ** (0.1 * total) - 10.0 ** (0.1 * part)
    if energy <= 0.0:
        raise EmissionsError("cannot subtract %.3f dB from %.3f dB -- the "
                             "remainder is not positive." % (part, total))
    return 10.0 * math.log10(energy)


def db_energy_mean(levels: Sequence[float]) -> Decibels:
    """Energy-average (not arithmetic mean) of a set of decibel levels."""
    vals = _as_floats(levels, "levels")
    _check_len(vals, 1, "levels")
    energy = math.fsum([10.0 ** (0.1 * v) for v in vals]) / len(vals)
    if energy <= 0.0:
        return float("-inf")
    return 10.0 * math.log10(energy)


def db_weighted_energy_mean(levels: Sequence[float],
                            weights: Sequence[float]) -> Decibels:
    """Duration-weighted energy average of decibel levels (the Leq definition)."""
    vals = _as_floats(levels, "levels")
    wts = _as_floats(weights, "weights")
    _check_len(vals, 1, "levels")
    if len(wts) != len(vals):
        raise EmissionsError("weights must match levels (%d vs %d)."
                             % (len(wts), len(vals)))
    for w in wts:
        if w < 0.0:
            raise EmissionsError("weights must be non-negative.")
    w_total = math.fsum(wts)
    if w_total == 0.0:
        raise EmissionsError("weights sum to zero.")
    energy = math.fsum([wts[i] * 10.0 ** (0.1 * vals[i])
                        for i in range(len(vals))]) / w_total
    if energy <= 0.0:
        return float("-inf")
    return 10.0 * math.log10(energy)


def point_source_attenuation(distance_m: float,
                             ref_distance_m: float = 1.0) -> Decibels:
    """Geometric loss from a POINT source: 20 log10(d/d0), 6.0206 dB per doubling."""
    d = _positive(distance_m, "distance_m")
    d0 = _positive(ref_distance_m, "ref_distance_m")
    return 20.0 * math.log10(d / d0)


def line_source_attenuation(distance_m: float,
                            ref_distance_m: float = 1.0) -> Decibels:
    """Geometric loss from an infinite LINE source: 10 log10(d/d0), 3.0103 per doubling."""
    d = _positive(distance_m, "distance_m")
    d0 = _positive(ref_distance_m, "ref_distance_m")
    return 10.0 * math.log10(d / d0)


def geometric_spreading(distance_m: float, ref_distance_m: float = 1.0,
                        source_type: str = "point") -> Decibels:
    """Dispatch the geometric spreading loss for a point or line source."""
    kind = _key("source type", source_type, ("point", "line"))
    if kind == "point":
        return point_source_attenuation(distance_m, ref_distance_m)
    return line_source_attenuation(distance_m, ref_distance_m)


def atmospheric_absorption(distance_m: float, frequency_hz: float = 500.0,
                           temperature_c: float = 20.0,
                           humidity_pct: float = 70.0,
                           pressure_ratio: float = 1.0) -> Decibels:
    """ISO 9613-1 atmospheric absorption in dB over a propagation distance."""
    d = _non_negative(distance_m, "distance_m")
    f = _positive(frequency_hz, "frequency_hz")
    t_c = _as_float(temperature_c, "temperature_c")
    rh = _non_negative(humidity_pct, "humidity_pct")
    pr = _positive(pressure_ratio, "pressure_ratio")
    t_k = t_c + 273.15
    if t_k <= 0.0:
        raise EmissionsError("temperature must be above absolute zero.")
    t_ref = 293.15
    t_triple = 273.16
    psat_ratio = 10.0 ** (-6.8346 * (t_triple / t_k) ** 1.261 + 4.6151)
    h = rh * psat_ratio / pr                       # molar water concentration, %
    fr_o = pr * (24.0 + 4.04e4 * h * (0.02 + h) / (0.391 + h))
    fr_n = pr * (t_k / t_ref) ** -0.5 * (
        9.0 + 280.0 * h * math.exp(-4.170 * ((t_k / t_ref) ** (-1.0 / 3.0) - 1.0)))
    relaxation = ((t_k / t_ref) ** -2.5
                  * (0.01275 * math.exp(-2239.1 / t_k) / (fr_o + f * f / fr_o)
                     + 0.1068 * math.exp(-3352.0 / t_k) / (fr_n + f * f / fr_n)))
    classical = 1.84e-11 * (1.0 / pr) * (t_k / t_ref) ** 0.5
    alpha_db_per_m = 8.686 * f * f * (classical + relaxation)
    return alpha_db_per_m * d


def ground_attenuation(distance_m: float, mean_height_m: float = 2.0,
                       ground_factor: float = 0.0) -> Decibels:
    """ISO 9613-2 alternative-method ground effect in dB, never negative."""
    d = _positive(distance_m, "distance_m")
    h_m = _non_negative(mean_height_m, "mean_height_m")
    g = _fraction(ground_factor, "ground_factor")
    raw = 4.8 - (2.0 * h_m / d) * (17.0 + 300.0 / d)
    if raw < 0.0:
        raw = 0.0
    return g * raw


def path_difference(source: Tuple[float, float], barrier_top: Tuple[float, float],
                    receiver: Tuple[float, float]) -> float:
    """Diffraction path difference delta = |SB| + |BR| - |SR| in metres."""
    pts = []
    for name, p in (("source", source), ("barrier_top", barrier_top),
                    ("receiver", receiver)):
        if not (isinstance(p, (tuple, list)) and len(p) == 2):
            raise EmissionsError("%s must be an (x, z) pair." % name)
        pts.append((_as_float(p[0], "%s x" % name),
                    _as_float(p[1], "%s z" % name)))
    (sx, sz), (bx, bz), (rx, rz) = pts
    d_sb = math.hypot(bx - sx, bz - sz)
    d_br = math.hypot(rx - bx, rz - bz)
    d_sr = math.hypot(rx - sx, rz - sz)
    return d_sb + d_br - d_sr


def fresnel_number(path_difference_m: float, frequency_hz: float = 500.0,
                   sound_speed_ms: float = 343.0) -> float:
    """Fresnel number N = 2 delta / lambda (negative when the path is clear)."""
    delta = _as_float(path_difference_m, "path_difference_m")
    f = _positive(frequency_hz, "frequency_hz")
    c = _positive(sound_speed_ms, "sound_speed_ms")
    wavelength = c / f
    return 2.0 * delta / wavelength


def maekawa_barrier_attenuation(path_difference_m: float,
                                frequency_hz: float = 500.0,
                                sound_speed_ms: float = 343.0,
                                max_db: float = 24.0) -> Decibels:
    """Maekawa / Kurze-Anderson barrier insertion loss in dB, capped at max_db."""
    cap = _positive(max_db, "max_db")
    n = fresnel_number(path_difference_m, frequency_hz, sound_speed_ms)
    if n > 0.0:
        x = math.sqrt(2.0 * math.pi * n)
        att = 5.0 + 20.0 * math.log10(x / math.tanh(x))
    elif n == 0.0:
        att = 5.0
    elif n > -0.2:
        x = math.sqrt(2.0 * math.pi * (-n))
        tan_x = math.tan(x)
        if tan_x <= 0.0:
            att = 0.0
        else:
            att = 5.0 + 20.0 * math.log10(x / tan_x)
    else:
        att = 0.0
    if att < 0.0:
        att = 0.0
    if att > cap:
        att = cap
    return att


# ===========================================================================
# 10. ROAD-TRAFFIC NOISE -- SOURCE MODELS AND INDICATORS
# ===========================================================================
@dataclass(frozen=True)
class REMELParameters:
    """FHWA reference energy mean emission level: L = a + b log10(speed in mph)."""

    a: float
    b: float
    name: str = "auto"


#: FHWA / TNM-lite REMELs, A-weighted, at the 15 m (50 ft) reference distance.
#: These are the published cruise-throttle coefficients and the suite uses them
#: as hard-coded ground truth: an automobile at 55 mph is 63.9 dB(A) at 15 m.
FHWA_REMELS = {
    "auto": REMELParameters(-2.4, 38.1, "auto"),
    "medium_truck": REMELParameters(16.4, 33.9, "medium truck"),
    "heavy_truck": REMELParameters(38.5, 24.6, "heavy truck"),
    "bus": REMELParameters(41.7, 22.5, "bus"),
    "motorcycle": REMELParameters(7.4, 41.2, "motorcycle"),
}

#: Noise vehicle classes, in a fixed reporting order.  Deliberately distinct
#: from ``VEHICLE_CATEGORIES``: acoustics splits the fleet by mass and axle
#: count, emissions splits it by fuel and Euro class.
NOISE_CATEGORIES = ("auto", "medium_truck", "heavy_truck", "bus", "motorcycle")

#: The FHWA reference distance, in metres (50 feet).
_NOISE_REF_DISTANCE_M = 15.0


def fhwa_remel(vehicle_type: str, speed_kmh: float) -> Decibels:
    """FHWA reference emission level in dB(A) at 15 m for one vehicle."""
    kind = _key("noise vehicle type", vehicle_type, NOISE_CATEGORIES)
    v = _positive(speed_kmh, "speed_kmh")
    p = FHWA_REMELS[kind]
    return p.a + p.b * math.log10(kmh_to_mph(v))


def fhwa_leq(vehicle_type: str, flow_veh_h: float, speed_kmh: float,
             distance_m: float = _NOISE_REF_DISTANCE_M,
             alpha: float = 0.0,
             ref_distance_m: float = _NOISE_REF_DISTANCE_M) -> Decibels:
    """FHWA hourly Leq in dB(A) from one vehicle stream on an infinite line."""
    n = _positive(flow_veh_h, "flow_veh_h")
    v = _positive(speed_kmh, "speed_kmh")
    d = _positive(distance_m, "distance_m")
    d0 = _positive(ref_distance_m, "ref_distance_m")
    a_gr = _non_negative(alpha, "alpha")
    remel = fhwa_remel(vehicle_type, speed_kmh)
    v_m_per_h = v * 1000.0
    line_term = 10.0 * math.log10(n * math.pi * d0 / v_m_per_h)
    spread = (1.0 + a_gr) * line_source_attenuation(d, d0)
    return remel + line_term - spread


@dataclass(frozen=True)
class CnossosParameters:
    """CNOSSOS-EU-lite rolling and propulsion sound-power coefficients."""

    a_rolling: float
    b_rolling: float
    a_propulsion: float
    b_propulsion: float
    v_ref_kmh: float = 70.0
    name: str = "light"


#: Broadband A-weighted aggregations of the CNOSSOS-EU 2012 octave-band
#: coefficient tables, rescaled so that the free-field level they imply at the
#: FHWA 15 m reference distance agrees with the TNM REMELs to within 0.1 dB at
#: 55 mph.  ``_validate_noise_models`` enforces that agreement at import.
CNOSSOS_COEFFICIENTS = {
    "auto": CnossosParameters(92.3, 30.0, 82.3, -3.0, 70.0, "light vehicle"),
    "medium_truck": CnossosParameters(103.2, 30.0, 99.2, -3.0, 70.0,
                                      "medium heavy vehicle"),
    "heavy_truck": CnossosParameters(109.0, 30.0, 106.0, -3.0, 70.0,
                                     "heavy vehicle"),
    "bus": CnossosParameters(108.6, 30.0, 105.6, -3.0, 70.0, "bus"),
    "motorcycle": CnossosParameters(103.1, 57.2, 107.1, -4.0, 70.0,
                                    "powered two-wheeler"),
}


def cnossos_sound_power(vehicle_type: str, speed_kmh: float) -> Decibels:
    """CNOSSOS-EU-lite A-weighted sound power level of one vehicle, dB re 1 pW.

    These are NOT comparable with a published CNOSSOS ``L_W``: the coefficient
    table is rescaled (see ``CNOSSOS_COEFFICIENTS``) so that the two source
    models agree through ``_line_level_from_power``, which uses
    ``L_W' - 10 log10(2 d)`` where the free-field infinite-line law carries
    ``10 log10(2 pi d)``.  The fixed ``10 log10(pi) = 4.97`` dB is absorbed into
    the table, so an automobile at 50 km/h reads 89.17 here against a real
    CNOSSOS 94-100.  The suite pins the module's own number with that
    explanation attached; it does not pin a standards value.
    """
    kind = _key("noise vehicle type", vehicle_type, NOISE_CATEGORIES)
    v = _positive(speed_kmh, "speed_kmh")
    p = CNOSSOS_COEFFICIENTS[kind]
    l_rolling = p.a_rolling + p.b_rolling * math.log10(v / p.v_ref_kmh)
    l_propulsion = p.a_propulsion + p.b_propulsion * \
        (v - p.v_ref_kmh) / p.v_ref_kmh
    return db_sum([l_rolling, l_propulsion])


def cnossos_line_sound_power(vehicle_type: str, flow_veh_h: float,
                             speed_kmh: float) -> Decibels:
    """CNOSSOS-EU-lite sound power per metre of road, dB(A) re 1 pW/m."""
    n = _positive(flow_veh_h, "flow_veh_h")
    v = _positive(speed_kmh, "speed_kmh")
    l_w = cnossos_sound_power(vehicle_type, v)
    veh_per_metre = n / (1000.0 * v)
    return l_w + 10.0 * math.log10(veh_per_metre)


def _line_level_from_power(l_w_per_metre: float, distance_m: float) -> float:
    """Level at a distance from an infinite incoherent line over hard ground."""
    d = _positive(distance_m, "distance_m")
    return _as_float(l_w_per_metre, "l_w_per_metre") - \
        10.0 * math.log10(2.0 * d)


@dataclass
class TrafficStream:
    """One vehicle class on a link: a flow, a speed and an acoustic category."""

    category: str
    flow_veh_h: float
    speed_kmh: float

    def __post_init__(self) -> None:
        self.category = _key("noise vehicle type", self.category,
                             NOISE_CATEGORIES)
        self.flow_veh_h = _positive(self.flow_veh_h, "flow_veh_h")
        self.speed_kmh = _positive(self.speed_kmh, "speed_kmh")

    def summary(self) -> Dict[str, Any]:
        """JSON-able description of the stream."""
        return {"category": self.category, "flowVehH": self.flow_veh_h,
                "speedKmh": self.speed_kmh}


@dataclass
class NoiseResult:
    """A receptor level with the propagation terms that produced it."""

    level_db: float
    source_db: float
    distance_m: float
    model: str
    stream_levels: List[Tuple[str, float]]
    spreading_db: float = 0.0
    ground_db: float = 0.0
    barrier_db: float = 0.0
    absorption_db: float = 0.0

    def total_attenuation_db(self) -> float:
        """Sum of the propagation losses applied to the source level."""
        return math.fsum([self.spreading_db, self.ground_db, self.barrier_db,
                          self.absorption_db])

    def summary(self) -> Dict[str, Any]:
        """JSON-able description of the receptor level."""
        return {"model": self.model, "distanceM": self.distance_m,
                "sourceDb": self.source_db, "levelDb": self.level_db,
                "spreadingDb": self.spreading_db, "groundDb": self.ground_db,
                "barrierDb": self.barrier_db,
                "absorptionDb": self.absorption_db,
                "totalAttenuationDb": self.total_attenuation_db(),
                "streams": [{"category": c, "refLevelDb": v}
                            for c, v in self.stream_levels]}


def road_noise_level(streams: Sequence[TrafficStream], distance_m: float,
                     model: str = "fhwa",
                     ref_distance_m: float = _NOISE_REF_DISTANCE_M,
                     ground_factor: float = 0.0, mean_height_m: float = 2.0,
                     barrier_path_difference_m: Optional[float] = None,
                     frequency_hz: float = 500.0,
                     include_absorption: bool = False,
                     temperature_c: float = 20.0,
                     humidity_pct: float = 70.0) -> NoiseResult:
    """Receptor dB(A) from traffic streams, with spreading, ground and barrier."""
    items = list(streams)
    _check_len(items, 1, "streams")
    kind = _key("noise model", model, ("fhwa", "cnossos"))
    d = _positive(distance_m, "distance_m")
    d0 = _positive(ref_distance_m, "ref_distance_m")
    # Deterministic order: the acoustic category order, then flow, then speed.
    ordered = sorted(items, key=lambda s: (NOISE_CATEGORIES.index(s.category),
                                           s.flow_veh_h, s.speed_kmh))
    levels = []  # type: List[Tuple[str, float]]
    for s in ordered:
        if kind == "fhwa":
            lvl = fhwa_leq(s.category, s.flow_veh_h, s.speed_kmh, d0, 0.0, d0)
        else:
            lvl = _line_level_from_power(
                cnossos_line_sound_power(s.category, s.flow_veh_h,
                                         s.speed_kmh), d0)
        levels.append((s.category, lvl))
    source_db = db_sum([lvl for _c, lvl in levels])
    spreading = line_source_attenuation(d, d0)
    ground = ground_attenuation(d, mean_height_m, ground_factor)
    barrier = 0.0
    if barrier_path_difference_m is not None:
        barrier = maekawa_barrier_attenuation(barrier_path_difference_m,
                                              frequency_hz)
    absorption = 0.0
    if include_absorption:
        absorption = atmospheric_absorption(d - d0 if d > d0 else 0.0,
                                            frequency_hz, temperature_c,
                                            humidity_pct)
    level = source_db - math.fsum([spreading, ground, barrier, absorption])
    return NoiseResult(level_db=level, source_db=source_db, distance_m=d,
                       model=kind, stream_levels=levels,
                       spreading_db=spreading, ground_db=ground,
                       barrier_db=barrier, absorption_db=absorption)


def l_aeq(levels: Sequence[float],
          durations: Optional[Sequence[float]] = None) -> Decibels:
    """Equivalent continuous A-weighted level over a set of periods."""
    if durations is None:
        return db_energy_mean(levels)
    return db_weighted_energy_mean(levels, durations)


def l_den(l_day: float, l_evening: float, l_night_db: float,
          evening_penalty: float = 5.0, night_penalty: float = 10.0,
          day_hours: float = 12.0, evening_hours: float = 4.0,
          night_hours: float = 8.0) -> Decibels:
    """Day-evening-night level: the hour-weighted energy mean with penalties."""
    ld = _as_float(l_day, "l_day")
    le = _as_float(l_evening, "l_evening")
    ln = _as_float(l_night_db, "l_night_db")
    pe = _as_float(evening_penalty, "evening_penalty")
    pn = _as_float(night_penalty, "night_penalty")
    hd = _non_negative(day_hours, "day_hours")
    he = _non_negative(evening_hours, "evening_hours")
    hn = _non_negative(night_hours, "night_hours")
    return db_weighted_energy_mean([ld, le + pe, ln + pn], [hd, he, hn])


def l_night(hourly_levels: Sequence[float],
            night_hours: Sequence[int] = (23, 0, 1, 2, 3, 4, 5, 6)) -> Decibels:
    """Night-time level: the energy mean of the hours in the night window."""
    vals = _as_floats(hourly_levels, "hourly_levels")
    if len(vals) != 24:
        raise EmissionsError("hourly_levels must have 24 entries (got %d)."
                             % len(vals))
    hours = sorted({int(h) for h in night_hours})
    _check_len(hours, 1, "night_hours")
    for h in hours:
        if h < 0 or h > 23:
            raise EmissionsError("night hours must lie in 0..23 (got %d)." % h)
    return db_energy_mean([vals[h] for h in hours])


def facade_level(free_field_db: float, reflection_db: float = 3.0) -> Decibels:
    """Facade level: the free-field level plus the first-reflection allowance."""
    return _as_float(free_field_db, "free_field_db") + \
        _non_negative(reflection_db, "reflection_db")


#: Miedema-Oudshoorn / EU dose-response cubics, keyed source -> coefficients on
#: (L_den - offset) for percent highly annoyed.
_HA_COEFFICIENTS = {
    "road": (9.868e-4, -1.436e-2, 0.5118, 42.0),
    "rail": (7.239e-4, -7.851e-3, 0.1695, 42.0),
    "air": (-9.199e-5, 3.932e-2, 0.2939, 42.0),
}

#: The same family for percent annoyed (a lower reporting threshold).  NOTE
#: the SIGN of the road and rail linear terms.  The transcription that
#: carries them negative puts %A BELOW %HA across the whole reporting range,
#: which is impossible -- "annoyed" is a superset of "highly annoyed", so the
#: curves must nest.  Taken positive they nest and reproduce the published
#: road anchors (about 15 % annoyed at L_den 55, 31 % at 65, 42 % at 70).
#: ``_validate_annoyance_curves`` enforces the nesting at import for all
#: three sources over the 42-85 dB window the cubics are fitted on; below
#: 42 dB every curve in the family is clamped to zero anyway.
_A_COEFFICIENTS = {
    "road": (1.795e-4, 2.110e-2, 0.3946, 37.0),
    "rail": (4.538e-4, 9.482e-3, 0.2129, 37.0),
    "air": (8.588e-6, 1.777e-2, 1.2210, 37.0),
}


def _dose_response(coeffs: Tuple[float, float, float, float],
                   level_db: float) -> float:
    """Evaluate a cubic annoyance dose-response and clamp it into [0, 100]."""
    c3, c2, c1, offset = coeffs
    x = _as_float(level_db, "level_db") - offset
    if x <= 0.0:
        return 0.0
    pct = c3 * x ** 3 + c2 * x * x + c1 * x
    if pct < 0.0:
        return 0.0
    if pct > 100.0:
        return 100.0
    return pct


def percent_highly_annoyed(l_den_db: float, source: str = "road") -> float:
    """Percent of the exposed population highly annoyed at a given L_den."""
    src = _key("noise source", source, tuple(sorted(_HA_COEFFICIENTS.keys())))
    return _dose_response(_HA_COEFFICIENTS[src], l_den_db)


def percent_annoyed(l_den_db: float, source: str = "road") -> float:
    """Percent of the exposed population annoyed at a given L_den."""
    src = _key("noise source", source, tuple(sorted(_A_COEFFICIENTS.keys())))
    return _dose_response(_A_COEFFICIENTS[src], l_den_db)


def percent_sleep_disturbed(l_night_db: float, source: str = "road") -> float:
    """Percent HIGHLY sleep-disturbed at a given L_night (Miedema-Vos quadratic).

    The name reads like the lower "sleep-disturbed" threshold; the curve is the
    HIGHLY-sleep-disturbed one, ``%HSD = 20.8 - 1.05 Ln + 0.01486 Ln^2`` for
    road traffic.  That quadratic does NOT vanish at the 40 dB floor
    conventionally applied to it -- it is worth 2.576 points there -- so the
    function steps discontinuously from 0 to about 2.58 immediately above
    40 dB.  Combining the published curve with the conventional floor is
    standard practice and the step is inherited from the curve, not introduced
    here.  Contrast ``percent_highly_annoyed``, whose cubic is written about its
    own 42 dB offset and is therefore continuous at it.
    """
    src = _key("noise source", source, ("road", "rail", "air"))
    ln = _as_float(l_night_db, "l_night_db")
    if src == "road":
        pct = 20.8 - 1.05 * ln + 0.01486 * ln * ln
    elif src == "rail":
        pct = 11.3 - 0.55 * ln + 0.00759 * ln * ln
    else:
        pct = 18.147 - 0.956 * ln + 0.01482 * ln * ln
    if pct < 0.0 or ln <= 40.0:
        return 0.0
    if pct > 100.0:
        return 100.0
    return pct


def _validate_annoyance_curves() -> None:
    """Assert %A never falls below %HA -- annoyance curves must nest."""
    for src in ("road", "rail", "air"):
        level = 42.0
        while level <= 85.0:
            ha = _dose_response(_HA_COEFFICIENTS[src], level)
            ann = _dose_response(_A_COEFFICIENTS[src], level)
            if ann < ha - 1e-9:
                raise EmissionsError(
                    "annoyance curves do not nest for %s at L_den %.2f: "
                    "%%A = %.3f but %%HA = %.3f." % (src, level, ann, ha))
            level += 0.25


_validate_annoyance_curves()


def _validate_noise_models() -> None:
    """Assert the two source models agree at the 55 mph reference condition."""
    v_ref = mph_to_kmh(55.0)
    for kind in NOISE_CATEGORIES:
        a = fhwa_leq(kind, 1000.0, v_ref, _NOISE_REF_DISTANCE_M)
        b = _line_level_from_power(
            cnossos_line_sound_power(kind, 1000.0, v_ref),
            _NOISE_REF_DISTANCE_M)
        if abs(a - b) > 0.5:
            raise EmissionsError(
                "FHWA and CNOSSOS-lite disagree for %s by %.3f dB at the "
                "reference condition." % (kind, a - b))


_validate_noise_models()


# ===========================================================================
# 11. GAUSSIAN-PLUME DISPERSION
# ===========================================================================
# Everything downstream of a tailpipe is advection plus diffusion.  The
# Gaussian plume is the closed-form steady solution of the advection-diffusion
# equation with a constant wind and Fickian cross-stream mixing:
#
#   C(x, y, z) = Q / (2 pi sigma_y sigma_z u)
#                * exp(-y^2 / (2 sigma_y^2))
#                * [ exp(-(z - H)^2 / (2 sigma_z^2))
#                  + exp(-(z + H)^2 / (2 sigma_z^2)) ]
#
# The second exponential is the image source that turns the ground into a
# perfect reflector.  Two exact consequences fall out of that algebra and the
# suite pins both of them:
#
#   * the ground-level centreline concentration of a ground-level source is
#     EXACTLY  C = Q / (pi sigma_y sigma_z u)  -- the two exponentials are
#     each one at y = z = H = 0, so the 2 in the denominator cancels;
#   * the flux u * integral(C) over any downwind plane is EXACTLY Q, for any
#     release height, because the bracket integrates to sigma_z sqrt(2 pi) on
#     z in [0, inf) no matter where H sits.  ``plume_mass_flux`` performs that
#     quadrature and the suite asserts the answer to a part in 1e-6.
#
# An INVERSION LID adds a second perfect reflector at z = L.  Satisfying both
# boundaries at once needs the doubly-infinite image series
#
#   R(z) = sum over n in Z of [ exp(-(z - H - 2nL)^2 / (2 sz^2))
#                            + exp(-(z + H - 2nL)^2 / (2 sz^2)) ]
#
# and three facts about it govern the implementation:
#
#   * it is PERIODIC with period 2L, so it is only the solution on [0, L];
#     above the lid the concentration is EXACTLY zero, because a perfect
#     reflector confines every gram below itself.  ``_vertical_term`` returns
#     zero there rather than the periodic continuation, and a release ABOVE
#     the lid is refused outright -- that plume does not mix down at all.
#   * truncating it at a fixed small n loses mass as soon as sigma_z
#     approaches L, which is precisely the regime a lid exists to model: four
#     image pairs at sigma_z = 10 L recover only 63 % of the source.  So
#     ``n_reflections`` is a MINIMUM and the series is always extended to the
#     count the supplied sigma_z needs (``_image_pairs_needed``).
#   * its sigma_z -> infinity limit is exact and worth knowing:
#     R -> sqrt(2 pi) sz / L, which turns the crosswind-integrated plume into
#     the uniformly mixed  C = Q / (u L)  -- every gram spread evenly through
#     the mixed layer.  Past ~50 lid depths the series and that closed form
#     agree to a few parts in 1e14 and the closed form is used.  Below about
#     two lid depths the plume is NOT yet mixed and sits legitimately ABOVE
#     Q/(u L) (by 1.4 % at sigma_z = L), so the limit is an asymptote and not
#     a bound.
#
# The sigmas themselves come from ``briggs_sigma_y`` / ``briggs_sigma_z``,
# which carry the published Briggs (1973) rural and urban interpolation
# formulae verbatim.  Keeping the sigmas OUT of the plume function is what
# makes the closed forms above testable without a table lookup in the way.
# ---------------------------------------------------------------------------

#: The six Pasquill-Gifford stability classes, very unstable (A) to stable (F).
STABILITY_CLASSES = ("A", "B", "C", "D", "E", "F")

#: Plain-language description of each stability class, for reporting.
_STABILITY_DESCRIPTIONS = {
    "A": "very unstable -- strong daytime convection, light wind",
    "B": "moderately unstable",
    "C": "slightly unstable",
    "D": "neutral -- overcast, or windy at any hour",
    "E": "slightly stable -- light-wind night with cloud",
    "F": "moderately stable -- calm clear night",
}

#: Daytime insolation classes for the Pasquill table.
_INSOLATION_CLASSES = ("strong", "moderate", "slight")

#: The published Pasquill-Gifford day table, as
#: ``(upper wind-speed bound m/s, strong, moderate, slight)``.  The half
#: classes of the original (A-B, B-C, C-D) are resolved to the MORE STABLE
#: member of the pair: smaller sigmas, higher ground-level concentrations, the
#: conservative choice for a kerbside assessment.
_PASQUILL_DAY = (
    (2.0, "A", "B", "B"),
    (3.0, "B", "B", "C"),
    (5.0, "B", "C", "C"),
    (6.0, "C", "D", "D"),
    (float("inf"), "C", "D", "D"),
)

#: The Pasquill-Gifford night table, as ``(upper wind-speed bound m/s, cloudy,
#: clear)`` where "cloudy" is thin overcast or at least 4/8 low cloud.  The
#: original leaves the sub-2 m/s row blank; the usual convention fills it with
#: the most stable class, and that is what this module does.
_PASQUILL_NIGHT = (
    (2.0, "F", "F"),
    (3.0, "E", "F"),
    (5.0, "D", "E"),
    (6.0, "D", "D"),
    (float("inf"), "D", "D"),
)


def pasquill_stability(wind_speed_ms: float, insolation: str = "moderate",
                       daytime: bool = True,
                       cloud_cover_oktas: float = 0.0) -> str:
    """Pasquill-Gifford stability class from wind speed, insolation and cloud.

    ``cloud_cover_oktas`` is consumed by the NIGHT branch only, where the
    published table splits on thin (< 4/8) against thick (>= 4/8) cover.  In
    DAYLIGHT the published table is indexed by insolation strength alone --
    strength already encodes the cloud -- so the argument is validated and then
    deliberately discarded.  A daytime call with ``insolation='strong'`` and
    8 oktas is physically contradictory but is answered from the strong column
    rather than refused, exactly as the table would answer it.
    """
    u = _non_negative(wind_speed_ms, "wind_speed_ms")
    oktas = _non_negative(cloud_cover_oktas, "cloud_cover_oktas")
    if oktas > 8.0:
        raise EmissionsError("cloud_cover_oktas must lie in [0, 8] (got %g)."
                             % oktas)
    if daytime:
        col = _INSOLATION_CLASSES.index(
            _key("insolation", insolation, _INSOLATION_CLASSES))
        for row in _PASQUILL_DAY:
            if u < row[0]:
                return row[1 + col]
        return _PASQUILL_DAY[-1][1 + col]
    col = 0 if oktas >= 4.0 else 1
    for row in _PASQUILL_NIGHT:
        if u < row[0]:
            return row[1 + col]
    return _PASQUILL_NIGHT[-1][1 + col]


def stability_description(stability: str) -> str:
    """One-line description of a Pasquill-Gifford stability class."""
    return _STABILITY_DESCRIPTIONS[
        _key("stability class", stability, STABILITY_CLASSES)]


#: Briggs (1973) sigma_y parameters as ``(coefficient, inner, exponent)`` for
#: ``sigma = coefficient * x * (1 + inner * x) ** exponent``, keyed
#: ``terrain -> stability``.  Urban lumps A with B and E with F, exactly as
#: the published table does.
_BRIGGS_SIGMA_Y = {
    "rural": {"A": (0.22, 1.0e-4, -0.5), "B": (0.16, 1.0e-4, -0.5),
              "C": (0.11, 1.0e-4, -0.5), "D": (0.08, 1.0e-4, -0.5),
              "E": (0.06, 1.0e-4, -0.5), "F": (0.04, 1.0e-4, -0.5)},
    "urban": {"A": (0.32, 4.0e-4, -0.5), "B": (0.32, 4.0e-4, -0.5),
              "C": (0.22, 4.0e-4, -0.5), "D": (0.16, 4.0e-4, -0.5),
              "E": (0.11, 4.0e-4, -0.5), "F": (0.11, 4.0e-4, -0.5)},
}

#: Briggs (1973) sigma_z parameters in the same form.  Note the POSITIVE
#: exponent on urban A/B: a convective urban plume grows faster than linearly.
_BRIGGS_SIGMA_Z = {
    "rural": {"A": (0.20, 0.0, 0.0), "B": (0.12, 0.0, 0.0),
              "C": (0.08, 2.0e-4, -0.5), "D": (0.06, 1.5e-3, -0.5),
              "E": (0.03, 3.0e-4, -1.0), "F": (0.016, 3.0e-4, -1.0)},
    "urban": {"A": (0.24, 1.0e-3, 0.5), "B": (0.24, 1.0e-3, 0.5),
              "C": (0.20, 0.0, 0.0), "D": (0.14, 3.0e-4, -0.5),
              "E": (0.08, 1.5e-3, -0.5), "F": (0.08, 1.5e-3, -0.5)},
}

#: Terrain types the Briggs curves are published for.
TERRAIN_TYPES = ("rural", "urban")


def _briggs(table: Dict[str, Dict[str, Tuple[float, float, float]]],
            x_m: float, stability: str, terrain: str) -> float:
    """Evaluate a Briggs interpolation formula at a downwind distance."""
    x = _positive(x_m, "x_m")
    cls = _key("stability class", stability, STABILITY_CLASSES)
    ter = _key("terrain", terrain, TERRAIN_TYPES)
    coef, inner, expo = table[ter][cls]
    base = 1.0 + inner * x
    if base <= 0.0:                 # unreachable for x > 0, guarded anyway
        raise EmissionsError("Briggs formula is undefined at x = %g m." % x)
    return coef * x * (base ** expo if expo != 0.0 else 1.0)


def briggs_sigma_y(x_m: float, stability: str = "D",
                   terrain: str = "urban") -> float:
    """Briggs (1973) crosswind dispersion parameter sigma_y in metres."""
    return _briggs(_BRIGGS_SIGMA_Y, x_m, stability, terrain)


def briggs_sigma_z(x_m: float, stability: str = "D",
                   terrain: str = "urban") -> float:
    """Briggs (1973) vertical dispersion parameter sigma_z in metres."""
    return _briggs(_BRIGGS_SIGMA_Z, x_m, stability, terrain)


def plume_rise(stack_diameter_m: float, exit_velocity_ms: float,
               exit_temperature_k: float, ambient_temperature_k: float,
               wind_speed_ms: float, stability: str = "D",
               potential_gradient_k_per_m: Optional[float] = None) -> float:
    """Briggs final plume rise in metres -- buoyancy or momentum, whichever wins."""
    d = _positive(stack_diameter_m, "stack_diameter_m")
    vs = _positive(exit_velocity_ms, "exit_velocity_ms")
    t_s = _positive(exit_temperature_k, "exit_temperature_k")
    t_a = _positive(ambient_temperature_k, "ambient_temperature_k")
    u = _positive(wind_speed_ms, "wind_speed_ms")
    cls = _key("stability class", stability, STABILITY_CLASSES)
    momentum = 3.0 * d * vs / u
    if t_s <= t_a:                  # exact test: no buoyancy at all
        return momentum
    flux = _G * vs * d * d * 0.25 * (t_s - t_a) / t_s
    if cls in ("E", "F"):
        default_grad = 0.020 if cls == "E" else 0.035
        grad = (default_grad if potential_gradient_k_per_m is None
                else _positive(potential_gradient_k_per_m,
                               "potential_gradient_k_per_m"))
        s_param = _G * grad / t_a
        bent = 2.6 * (flux / (u * s_param)) ** (1.0 / 3.0)
        calm = 5.0 * flux ** 0.25 / s_param ** 0.375
        buoyant = bent if bent < calm else calm
    elif flux < 55.0:
        buoyant = 21.425 * flux ** 0.75 / u
    else:
        buoyant = 38.71 * flux ** 0.6 / u
    return buoyant if buoyant > momentum else momentum


#: How far, in sigma_z, the lidded image series must reach before it may be
#: truncated.  Beyond 8 sigma a Gaussian image contributes below 1e-14 of the
#: peak, so a series covering +/-8 sigma either side of the receptor is
#: converged to double precision.
_IMAGE_SPAN_SIGMAS = 8.0
#: Image-pair cap.  Once the series needs more pairs than this, sigma_z exceeds
#: roughly 50 lid depths and the trapped profile is indistinguishable (to a few
#: parts in 1e14) from its exact uniform-mixing limit, which is used instead.
_MAX_IMAGE_PAIRS = 200


def _well_mixed_vertical(sigma_z: float, lid: float, reflect: bool) -> float:
    """Exact ``sigma_z -> inf`` limit of the trapped image series.

    Poisson summation sends ``sum_n exp(-(a - 2 n L)^2 / (2 sz^2))`` to
    ``sqrt(2 pi) sz / (2 L)`` for every offset ``a``, so the two interleaved
    lattices of a reflecting plume sum to ``sqrt(2 pi) sz / L``.  Substituted
    into the plume that is exactly the uniform profile ``C = Q / (u L)`` --
    every gram emitted, spread evenly through the mixed layer.  Without the
    ground image only one lattice survives, so the limit halves.
    """
    limit = _SQRT_2PI * sigma_z / lid
    return limit if reflect else 0.5 * limit


def _image_pairs_needed(sigma_z: float, lid: float) -> int:
    """Image pairs required for the trapped series to converge at this sigma_z.

    The receptor and its ground image both sit inside ``[0, lid]``, so the
    furthest source the receptor can still feel is ``span sigma_z + 2 lid``
    away; each pair steps ``2 lid``.
    """
    reach = _IMAGE_SPAN_SIGMAS * sigma_z + 2.0 * lid
    return int(math.ceil(reach / (2.0 * lid)))


def _vertical_term(z: float, stack_height_m: float, sigma_z: float,
                   reflect: bool, mixing_height_m: Optional[float],
                   n_reflections: int) -> float:
    """The vertical factor of the plume: direct term plus its ground images.

    With no lid this is the classic two-term bracket.  With a lid it is the
    doubly-infinite image series that makes both the ground and the inversion
    perfect reflectors, truncated at whichever is larger of the caller's
    ``n_reflections`` and the count this sigma_z actually NEEDS -- see
    ``_image_pairs_needed``.  Truncating at a fixed small count silently loses
    mass out of the top of the mixed layer once sigma_z approaches the lid
    depth, which is exactly the regime a lid exists to model.

    Above the lid the answer is EXACTLY zero: a perfectly reflecting inversion
    confines every gram below itself.  The untruncated series is periodic with
    period ``2 lid``, so evaluating it at ``z > lid`` would return the profile
    from inside the mixed layer -- mathematically the same function, but
    outside the domain on which it solves the problem.
    """
    inv = 0.5 / (sigma_z * sigma_z)
    if mixing_height_m is None:
        total = math.exp(-(z - stack_height_m) ** 2 * inv)
        if reflect:
            total += math.exp(-(z + stack_height_m) ** 2 * inv)
        return total
    lid = mixing_height_m
    if z > lid:
        return 0.0
    needed = _image_pairs_needed(sigma_z, lid)
    if needed > _MAX_IMAGE_PAIRS:
        return _well_mixed_vertical(sigma_z, lid, reflect)
    n_img = n_reflections if n_reflections > needed else needed
    terms = []
    for n in range(-n_img, n_img + 1):
        shift = 2.0 * n * lid
        terms.append(math.exp(-(z - stack_height_m - shift) ** 2 * inv))
        if reflect:
            terms.append(math.exp(-(z + stack_height_m - shift) ** 2 * inv))
    return math.fsum(sorted(terms))


def _check_release_below_lid(stack_height_m: float,
                             lid: Optional[float], name: str) -> None:
    """Refuse a release above the inversion -- the image method cannot model it."""
    if lid is not None and stack_height_m > lid:
        raise EmissionsError(
            "%s %g m is above mixing_height_m %g m -- a plume released above "
            "the inversion does not mix down and is outside this model."
            % (name, stack_height_m, lid))


def gaussian_plume_concentration(q_g_s: float, wind_speed_ms: float,
                                 sigma_y_m: float, sigma_z_m: float,
                                 y_m: float = 0.0, z_m: float = 0.0,
                                 stack_height_m: float = 0.0,
                                 reflect: bool = True,
                                 mixing_height_m: Optional[float] = None,
                                 n_reflections: int = 4) -> Concentration:
    """Gaussian-plume concentration in g/m3 -- Q/(pi sy sz u) on the centreline.

    ``mixing_height_m`` installs a second perfect reflector at that height.
    ``n_reflections`` is then a MINIMUM image-pair count: the series is always
    extended far enough to converge at the supplied sigma_z (see
    ``_vertical_term``), so a caller cannot silently lose mass by leaving the
    default in place.  Above the lid the concentration is exactly zero, and a
    release above the lid is refused.
    """
    q = _non_negative(q_g_s, "q_g_s")
    u = _positive(wind_speed_ms, "wind_speed_ms")
    sy = _positive(sigma_y_m, "sigma_y_m")
    sz = _positive(sigma_z_m, "sigma_z_m")
    y = _as_float(y_m, "y_m")
    z = _non_negative(z_m, "z_m")
    h = _non_negative(stack_height_m, "stack_height_m")
    lid = None
    if mixing_height_m is not None:
        lid = _positive(mixing_height_m, "mixing_height_m")
    n_img = int(n_reflections)
    if n_img < 0:
        raise EmissionsError("n_reflections must be non-negative (got %d)."
                             % n_img)
    _check_release_below_lid(h, lid, "stack_height_m")
    crosswind = math.exp(-0.5 * (y / sy) ** 2)
    vertical = _vertical_term(z, h, sz, reflect, lid, n_img)
    return q * crosswind * vertical / (2.0 * math.pi * sy * sz * u)


def crosswind_integrated_concentration(q_g_s: float, wind_speed_ms: float,
                                       sigma_z_m: float, z_m: float = 0.0,
                                       stack_height_m: float = 0.0,
                                       reflect: bool = True,
                                       mixing_height_m: Optional[float] = None,
                                       n_reflections: int = 4) -> Concentration:
    """Plume integrated over the crosswind axis -- also the infinite-line answer.

    Feed it a point-source strength in g/s and the result is the crosswind
    integral of the plume in g/m2; feed it a LINE strength in g/(m s) and the
    same expression is the concentration in g/m3 under an infinite crosswind
    line source.  At ground level with a ground-level release it collapses to
    ``2 q / (sqrt(2 pi) sigma_z u)``.

    ``reflect=False`` drops the ground image, which HALVES the ground-level
    answer exactly.  ``mixing_height_m`` traps the plume under an inversion;
    ``n_reflections`` is a minimum image count and the series is extended as
    far as the supplied sigma_z requires, so at ``sigma_z >> lid`` the result
    converges onto the exact well-mixed value ``q / (u lid)`` rather than
    losing mass through the lid.  Above the lid the answer is exactly zero.
    """
    q = _non_negative(q_g_s, "q_g_s")
    u = _positive(wind_speed_ms, "wind_speed_ms")
    sz = _positive(sigma_z_m, "sigma_z_m")
    z = _non_negative(z_m, "z_m")
    h = _non_negative(stack_height_m, "stack_height_m")
    lid = None
    if mixing_height_m is not None:
        lid = _positive(mixing_height_m, "mixing_height_m")
    n_img = int(n_reflections)
    if n_img < 0:
        raise EmissionsError("n_reflections must be non-negative (got %d)."
                             % n_img)
    _check_release_below_lid(h, lid, "stack_height_m")
    vertical = _vertical_term(z, h, sz, reflect, lid, n_img)
    return q * vertical / (_SQRT_2PI * sz * u)


def line_source_concentration(q_g_m_s: float, wind_speed_ms: float,
                              sigma_y_m: float, sigma_z_m: float,
                              y_start_m: float, y_end_m: float,
                              y_m: float = 0.0, z_m: float = 0.0,
                              source_height_m: float = 0.0,
                              reflect: bool = True) -> Concentration:
    """Closed-form finite line source: the erf integral of the point plume.

    Integrating the point plume over a segment ``[y_start, y_end]`` of the
    crosswind axis gives exactly

        C = q / (sqrt(2 pi) sigma_z u) * R(z)
            * [ Phi((y - y_start)/sigma_y) - Phi((y - y_end)/sigma_y) ]

    with ``R(z)`` the vertical bracket.  As the segment grows without bound
    the bracket tends to one and the expression becomes
    ``crosswind_integrated_concentration`` -- the identity the suite uses to
    tie the two together.
    """
    q = _non_negative(q_g_m_s, "q_g_m_s")
    u = _positive(wind_speed_ms, "wind_speed_ms")
    sy = _positive(sigma_y_m, "sigma_y_m")
    sz = _positive(sigma_z_m, "sigma_z_m")
    y0 = _as_float(y_start_m, "y_start_m")
    y1 = _as_float(y_end_m, "y_end_m")
    if y1 <= y0:
        raise EmissionsError("y_end_m must exceed y_start_m (%g <= %g)."
                             % (y1, y0))
    y = _as_float(y_m, "y_m")
    z = _non_negative(z_m, "z_m")
    h = _non_negative(source_height_m, "source_height_m")
    vertical = _vertical_term(z, h, sz, reflect, None, 0)
    span = _norm_cdf((y - y0) / sy) - _norm_cdf((y - y1) / sy)
    if span < 0.0:                  # only reachable through rounding
        span = 0.0
    return q * vertical * span / (_SQRT_2PI * sz * u)


def road_link_concentration(emission_g_m_s: float, wind_speed_ms: float,
                            distance_m: float, road_length_m: float = 2000.0,
                            stability: str = "D", terrain: str = "urban",
                            wind_angle_deg: float = 90.0,
                            receptor_height_m: float = 1.5,
                            source_height_m: float = 0.0,
                            initial_sigma_z_m: float = 0.0,
                            min_wind_ms: float = 0.5,
                            background_ug_m3: float = 0.0) -> float:
    """Kerbside ug/m3 from a straight road treated as a finite line source.

    ``wind_angle_deg`` is the angle between the wind and the road axis, so 90
    degrees is the cross-road worst case and 0 degrees is a wind blowing along
    the carriageway.  Only the component normal to the road ventilates the
    receptor, floored at ``min_wind_ms`` the way CALINE floors it, because a
    dead-calm denominator is not a physical answer.  ``initial_sigma_z_m`` is
    the vehicle-wake mixing depth (1-2 m on a busy road); it defaults to zero
    so that the closed forms above stay exactly recoverable.
    """
    q = _non_negative(emission_g_m_s, "emission_g_m_s")
    u = _positive(wind_speed_ms, "wind_speed_ms")
    x = _positive(distance_m, "distance_m")
    length = _positive(road_length_m, "road_length_m")
    angle = _as_float(wind_angle_deg, "wind_angle_deg")
    u_min = _positive(min_wind_ms, "min_wind_ms")
    sz0 = _non_negative(initial_sigma_z_m, "initial_sigma_z_m")
    bg = _non_negative(background_ug_m3, "background_ug_m3")
    normal = u * abs(math.sin(math.radians(angle)))
    u_eff = normal if normal > u_min else u_min
    sy = briggs_sigma_y(x, stability, terrain)
    sz = briggs_sigma_z(x, stability, terrain)
    if sz0 > 0.0:
        sz = math.sqrt(sz * sz + sz0 * sz0)
    conc = line_source_concentration(q, u_eff, sy, sz, -0.5 * length,
                                     0.5 * length, 0.0, receptor_height_m,
                                     source_height_m)
    return ug_per_m3(conc) + bg


def _simpson_weights(n: int) -> List[float]:
    """Composite Simpson weights over ``n`` intervals (``n`` even)."""
    if n < 2 or n % 2 != 0:
        raise EmissionsError("Simpson needs an even interval count >= 2 "
                             "(got %d)." % n)
    w = [1.0] * (n + 1)
    for i in range(1, n):
        w[i] = 4.0 if i % 2 == 1 else 2.0
    return w


def plume_mass_flux(q_g_s: float, wind_speed_ms: float, sigma_y_m: float,
                    sigma_z_m: float, stack_height_m: float = 0.0,
                    n_y: int = 400, n_z: int = 400,
                    span_sigmas: float = 8.0,
                    mixing_height_m: Optional[float] = None,
                    n_reflections: int = 4) -> float:
    """Mass flux u * integral(C dy dz) through a downwind plane -- must equal Q.

    This is the conservation oracle made executable: a plume that does not
    integrate back to its own source strength is not a solution of anything.
    The quadrature is a tensor-product composite Simpson rule over
    ``[-span sy, span sy]`` and ``[0, H + span sz]``; the vertical bracket is
    even about the ground for every release height, so the half-range integral
    converges just as fast as the crosswind one.  The integrand is a product
    of a function of ``y`` and a function of ``z``, and a tensor Simpson rule
    on a separable integrand factorises EXACTLY into the product of its two
    one-dimensional rules -- so the loops below are the full two-dimensional
    rule, evaluated in ``n_y + n_z`` work instead of ``n_y * n_z``.

    Passing ``mixing_height_m`` integrates the TRAPPED plume, over ``[0, lid]``
    rather than ``[0, H + span sz]``, and the answer must still be Q: a perfect
    inversion moves mass around inside the mixed layer but destroys none of it.
    Leaving the lid out of this oracle is what let a truncated image series lose
    a third of the plume unnoticed, so the lidded path is wired in on purpose.

    ``span_sigmas``, ``n_y`` and ``n_z`` are the quadrature's own knobs and it
    is a quadrature: at the defaults the identity holds to ~2e-15, but a
    three-point rule across +/-8 sigma, or a +/-1 sigma window that discards
    half the mass by construction, will return whatever such a rule returns.
    """
    q = _non_negative(q_g_s, "q_g_s")
    u = _positive(wind_speed_ms, "wind_speed_ms")
    sy = _positive(sigma_y_m, "sigma_y_m")
    sz = _positive(sigma_z_m, "sigma_z_m")
    h = _non_negative(stack_height_m, "stack_height_m")
    span = _positive(span_sigmas, "span_sigmas")
    lid = None
    if mixing_height_m is not None:
        lid = _positive(mixing_height_m, "mixing_height_m")
    n_ref = int(n_reflections)
    if n_ref < 0:
        raise EmissionsError("n_reflections must be non-negative (got %d)."
                             % n_ref)
    _check_release_below_lid(h, lid, "stack_height_m")
    wy = _simpson_weights(int(n_y))
    wz = _simpson_weights(int(n_z))
    y_hi = span * sy
    z_hi = h + span * sz if lid is None else lid
    hy = 2.0 * y_hi / int(n_y)
    hz = z_hi / int(n_z)
    vertical = math.fsum([wz[j] * _vertical_term(hz * j, h, sz, True, lid,
                                                 n_ref)
                          for j in range(len(wz))]) * hz / 3.0
    crosswind = math.fsum([wy[i] * math.exp(-0.5 * ((-y_hi + hy * i) / sy) ** 2)
                           for i in range(len(wy))]) * hy / 3.0
    amplitude = q / (2.0 * math.pi * sy * sz * u)
    return u * amplitude * crosswind * vertical


# ===========================================================================
# 12. BOX AND STREET-CANYON MODELS
# ===========================================================================
# The plume assumes the sky is open.  Between two rows of buildings it is not:
# the wind skims the roofs and drives a recirculating vortex in the canyon, so
# the leeward facade breathes the same air twice.  Two models cover that:
#
#   * the BOX model -- one well-mixed cell of height H and crosswind width W
#     flushed at the wind speed u.  At steady state the mass balance is
#     Q = C u H W, so C = Q / (u H W) EXACTLY, and the approach to it is the
#     first-order relaxation with time constant L/u.
#   * an OSPM-LITE canyon -- two coupled boxes.  The upwind building throws a
#     vortex roughly two building heights long; whatever is emitted under it
#     is trapped, well mixed, and leaves only through the vortex roof at the
#     turbulent exchange velocity w_t.  The rest of the carriageway is
#     ventilated by the through-flow.  The leeward facade sees vortex air plus
#     the direct contribution; the windward facade sees the direct
#     contribution plus a documented re-entrainment fraction of the vortex.
#
# Both are exactly linear in the emission rate, both collapse to the
# background as the wind speed grows without bound, and the vortex obeys a
# closed mass balance -- ``C_rec * w_t * L_rec == q * L_rec / W`` -- which is
# what the suite checks rather than any tuned number.
# ---------------------------------------------------------------------------
def box_model_concentration(q_g_s: float, wind_speed_ms: float,
                            mixing_height_m: float, width_m: float,
                            background_g_m3: float = 0.0) -> Concentration:
    """Steady-state box concentration in g/m3 -- exactly Q / (u H W) plus background."""
    q = _non_negative(q_g_s, "q_g_s")
    u = _positive(wind_speed_ms, "wind_speed_ms")
    h = _positive(mixing_height_m, "mixing_height_m")
    w = _positive(width_m, "width_m")
    bg = _non_negative(background_g_m3, "background_g_m3")
    return q / (u * h * w) + bg


def box_model_buildup(q_g_s: float, wind_speed_ms: float,
                      mixing_height_m: float, width_m: float,
                      length_m: float, duration_s: float, dt_s: float = 60.0,
                      initial_g_m3: float = 0.0,
                      background_g_m3: float = 0.0) -> List[Tuple[float, float]]:
    """Transient box model: ``(time, concentration)`` on the exact exponential.

    The cell has volume ``L W H`` and is flushed at ``u W H`` cubic metres per
    second, so the residence time is exactly ``L / u`` and the solution of
    ``V dC/dt = Q - u W H C`` is a pure exponential relaxation onto the
    steady state.  The samples below are evaluations of that closed form, not
    a numerical integration, so the series is exact at every point.
    """
    q = _non_negative(q_g_s, "q_g_s")
    u = _positive(wind_speed_ms, "wind_speed_ms")
    h = _positive(mixing_height_m, "mixing_height_m")
    w = _positive(width_m, "width_m")
    length = _positive(length_m, "length_m")
    dur = _non_negative(duration_s, "duration_s")
    step = _positive(dt_s, "dt_s")
    c0 = _non_negative(initial_g_m3, "initial_g_m3")
    bg = _non_negative(background_g_m3, "background_g_m3")
    steady = q / (u * h * w)
    tau = length / u
    n = int(math.floor(dur / step + 1e-9))
    series = []
    for i in range(n + 1):
        t = step * i
        decay = math.exp(-t / tau)
        series.append((t, steady + (c0 - steady) * decay + bg))
    if series[-1][0] < dur:
        decay = math.exp(-dur / tau)
        series.append((dur, steady + (c0 - steady) * decay + bg))
    return series


@dataclass
class StreetCanyon:
    """A street canyon: two facades of a given height a given width apart."""

    height_m: float
    width_m: float
    name: str = "canyon"

    def __post_init__(self) -> None:
        self.height_m = _positive(self.height_m, "height_m")
        self.width_m = _positive(self.width_m, "width_m")
        self.name = str(self.name)

    def aspect_ratio(self) -> float:
        """Height-to-width ratio H/W -- above 0.5 a single vortex fills the canyon."""
        return self.height_m / self.width_m

    def regime(self) -> str:
        """Oke's canyon flow regime from the aspect ratio."""
        ar = self.aspect_ratio()
        if ar < 0.3:
            return "isolated roughness"
        if ar < 0.5:
            return "wake interference"
        return "skimming"

    def summary(self) -> Dict[str, Any]:
        """JSON-able description of the canyon geometry."""
        return {"name": self.name, "heightM": self.height_m,
                "widthM": self.width_m, "aspectRatio": self.aspect_ratio(),
                "regime": self.regime()}


@dataclass
class CanyonResult:
    """Leeward and windward facade concentrations with the terms behind them."""

    leeward_ug_m3: float
    windward_ug_m3: float
    recirculation_ug_m3: float
    direct_ug_m3: float
    background_ug_m3: float
    exchange_velocity_ms: float
    recirculation_length_m: float
    trapped_fraction: float
    aspect_ratio: float
    name: str = "canyon"

    @property
    def max_ug_m3(self) -> float:
        """The worse of the two facades -- the one that fails a limit value first."""
        return max(self.leeward_ug_m3, self.windward_ug_m3)

    @property
    def leeward_increment_ug_m3(self) -> float:
        """Leeward level above the urban background."""
        return self.leeward_ug_m3 - self.background_ug_m3

    def summary(self) -> Dict[str, Any]:
        """JSON-able description of the canyon result."""
        return {"name": self.name,
                "leewardUgM3": self.leeward_ug_m3,
                "windwardUgM3": self.windward_ug_m3,
                "maxUgM3": self.max_ug_m3,
                "recirculationUgM3": self.recirculation_ug_m3,
                "directUgM3": self.direct_ug_m3,
                "backgroundUgM3": self.background_ug_m3,
                "exchangeVelocityMs": self.exchange_velocity_ms,
                "recirculationLengthM": self.recirculation_length_m,
                "trappedFraction": self.trapped_fraction,
                "aspectRatio": self.aspect_ratio}


def canyon_concentration(canyon: StreetCanyon, emission_g_m_s: float,
                         roof_wind_ms: float, wind_angle_deg: float = 90.0,
                         background_ug_m3: float = 0.0,
                         traffic_turbulence_ms: float = 0.0,
                         turbulence_coefficient: float = 0.1,
                         exchange_floor_ms: float = 0.05,
                         through_flow_factor: float = 0.4,
                         recirculation_leak: float = 0.2) -> CanyonResult:
    """OSPM-lite street canyon: a mass-balanced vortex plus a through-flow box.

    ``emission_g_m_s`` is the whole carriageway's emission per metre of street
    length.  ``wind_angle_deg`` is the angle between the roof-level wind and
    the street axis: the vortex is driven by the cross-street component, so a
    wind blowing straight along the canyon produces no recirculation at all
    and the answer degenerates to the through-flow box.

    The vortex extends ``2 H`` from the upwind facade, capped by the canyon
    width, and the fraction of the carriageway under it is trapped.  Trapped
    mass leaves only through the vortex roof at ``w_t = sqrt(sigma_w^2 +
    sigma_traffic^2)`` with ``sigma_w = turbulence_coefficient * u_roof``, so
    the vortex mass balance closes exactly.  ``exchange_floor_ms`` is the
    still-air turbulence floor -- a physical constant of the parameterisation,
    not a numerical epsilon.

    Two consequences of that closure are exact and deliberate, and the suite
    pins both:

    * the recirculation concentration is INDEPENDENT of the vortex length.
      Substituting ``trapped = l_rec / W`` into ``q trapped / (w_t l_rec)``
      cancels ``l_rec`` identically, leaving ``c_rec = q / (w_t W)`` -- a
      shorter vortex traps proportionally less emission but has proportionally
      less roof to vent it through.  So ``c_rec`` does not depend on the canyon
      HEIGHT at all, nor, above the saturation angle where the vortex fills the
      canyon, on the wind angle.
    * the limit as the vortex vanishes is NOT zero, so the exact-zero branch at
      an along-street wind is a genuine cliff: in a square 20 m canyon the
      leeward level falls by a factor of ~5 between a wind 1e-9 degrees off the
      axis and one exactly along it.  That is a property of a mass balance
      whose vent area shrinks with the vortex, not an implementation error --
      removing it means choosing a different recirculation model.
      ``through_flow_factor`` is correspondingly DEAD above the saturation
      angle, where no through-flow region is left for it to scale.
    """
    if not isinstance(canyon, StreetCanyon):
        raise EmissionsError("canyon must be a StreetCanyon.")
    q = _non_negative(emission_g_m_s, "emission_g_m_s")
    u_roof = _positive(roof_wind_ms, "roof_wind_ms")
    angle = _as_float(wind_angle_deg, "wind_angle_deg")
    bg = _non_negative(background_ug_m3, "background_ug_m3")
    sig_traffic = _non_negative(traffic_turbulence_ms, "traffic_turbulence_ms")
    k_turb = _positive(turbulence_coefficient, "turbulence_coefficient")
    w_floor = _positive(exchange_floor_ms, "exchange_floor_ms")
    alpha = _positive(through_flow_factor, "through_flow_factor")
    leak = _fraction(recirculation_leak, "recirculation_leak")

    h = canyon.height_m
    w = canyon.width_m
    cross = abs(math.sin(math.radians(angle)))
    l_rec = 2.0 * h * cross
    if l_rec > w:
        l_rec = w
    trapped = l_rec / w
    sigma_w = k_turb * u_roof
    w_t = math.sqrt(sigma_w * sigma_w + sig_traffic * sig_traffic)
    if w_t < w_floor:
        w_t = w_floor
    if l_rec == 0.0:                    # exact-zero test: no vortex at all
        c_rec = 0.0
    else:
        c_rec = q * trapped / (w_t * l_rec)
    u_through = alpha * u_roof
    c_dir = q * (1.0 - trapped) / (u_through * h)
    leeward = ug_per_m3(c_rec + c_dir) + bg
    windward = ug_per_m3(c_dir + leak * c_rec) + bg
    return CanyonResult(leeward_ug_m3=leeward, windward_ug_m3=windward,
                        recirculation_ug_m3=ug_per_m3(c_rec),
                        direct_ug_m3=ug_per_m3(c_dir),
                        background_ug_m3=bg, exchange_velocity_ms=w_t,
                        recirculation_length_m=l_rec,
                        trapped_fraction=trapped,
                        aspect_ratio=canyon.aspect_ratio(), name=canyon.name)


# ===========================================================================
# 13. BACKGROUND AND POPULATION-WEIGHTED EXPOSURE
# ===========================================================================
def add_background(increment: float, background: float) -> float:
    """Add an urban background to a local increment -- same unit, both non-negative.

    A one-line function that exists so the unit discipline lives in exactly
    one place: the two arguments must already be in the same unit, and a
    negative "concentration" on either side is a bug rather than a number.
    """
    inc = _non_negative(increment, "increment")
    bg = _non_negative(background, "background")
    return inc + bg


@dataclass
class ExposureZone:
    """A population living at a modelled concentration -- one exposure cell."""

    name: str
    population: float
    concentration_ug_m3: float
    area_km2: float = 0.0

    def __post_init__(self) -> None:
        self.name = str(self.name)
        self.population = _non_negative(self.population, "population")
        self.concentration_ug_m3 = _non_negative(self.concentration_ug_m3,
                                                 "concentration_ug_m3")
        self.area_km2 = _non_negative(self.area_km2, "area_km2")

    def person_ug(self) -> float:
        """Population times concentration -- the numerator of the weighted mean."""
        return self.population * self.concentration_ug_m3

    def density_per_km2(self) -> float:
        """People per square kilometre (zero for a zero-area zone)."""
        if self.area_km2 == 0.0:
            return 0.0
        return self.population / self.area_km2

    def summary(self) -> Dict[str, Any]:
        """JSON-able description of the zone."""
        return {"name": self.name, "population": self.population,
                "concentrationUgM3": self.concentration_ug_m3,
                "areaKm2": self.area_km2,
                "densityPerKm2": self.density_per_km2()}


def _ordered_zones(zones: Sequence[ExposureZone]) -> List[ExposureZone]:
    """Zones in a fixed, hash-seed independent order."""
    items = list(zones)
    _check_len(items, 1, "zones")
    for z in items:
        if not isinstance(z, ExposureZone):
            raise EmissionsError("every zone must be an ExposureZone.")
    return sorted(items, key=lambda z: (z.name, z.concentration_ug_m3,
                                        z.population))


def population_weighted_exposure(zones: Sequence[ExposureZone]) -> float:
    """Population-weighted mean concentration in ug/m3 over a set of zones."""
    items = _ordered_zones(zones)
    total_pop = math.fsum([z.population for z in items])
    if total_pop == 0.0:                # exact-zero test, never an epsilon
        return 0.0
    return math.fsum([z.person_ug() for z in items]) / total_pop


def exceedance_population(zones: Sequence[ExposureZone],
                          limit_ug_m3: float) -> Dict[str, Any]:
    """Population living above a limit value, with the offending zones named."""
    items = _ordered_zones(zones)
    limit = _non_negative(limit_ug_m3, "limit_ug_m3")
    over = [z for z in items if z.concentration_ug_m3 > limit]
    pop_over = math.fsum([z.population for z in over])
    total_pop = math.fsum([z.population for z in items])
    share = 0.0 if total_pop == 0.0 else pop_over / total_pop
    return {"limitUgM3": limit, "population": pop_over,
            "totalPopulation": total_pop, "share": share,
            "zones": [z.name for z in over],
            "worstZone": max(items, key=lambda z: (z.concentration_ug_m3,
                                                   z.name)).name}


def annual_mean_from_hourly(hourly_ug_m3: Sequence[Optional[float]],
                            min_capture: float = 0.75) -> float:
    """Annual mean from an hourly series, refusing to report below a data capture.

    The series IS the averaging period -- one entry per hour of it -- and
    ``None`` marks a missing hour.  European reporting requires 75 % valid
    data before an annual mean may be published at all, so the default refuses
    rather than quietly averaging a fortnight of readings into a year.

    ``min_capture=0.0`` means "publish whatever there is", but there is still no
    mean of NOTHING: an averaging period with no valid hour at all raises
    ``EmissionsError`` rather than dividing by zero.  An instrument offline for
    the whole period is precisely the case a capture guard exists to catch, so
    it has to surface as this module's own error type -- a caller wrapping the
    call in ``except EmissionsError`` would not catch a ``ZeroDivisionError``.
    """
    vals = list(hourly_ug_m3)
    _check_len(vals, 1, "hourly_ug_m3")
    capture_floor = _fraction(min_capture, "min_capture")
    valid = []
    for i, v in enumerate(vals):
        if v is None:
            continue
        valid.append(_non_negative(v, "hourly_ug_m3[%d]" % i))
    if not valid:
        raise EmissionsError("no valid hours in hourly_ug_m3 (%d entries, all "
                             "missing) -- there is no mean to report."
                             % len(vals))
    capture = len(valid) / float(len(vals))
    if capture < capture_floor:
        raise EmissionsError("data capture %.1f %% is below the required "
                             "%.1f %% (%d of %d hours valid)."
                             % (100.0 * capture, 100.0 * capture_floor,
                                len(valid), len(vals)))
    return math.fsum(valid) / len(valid)


# ===========================================================================
# 14. HEALTH IMPACT -- RELATIVE RISK, ATTRIBUTABLE FRACTION, YLL
# ===========================================================================
# The chain is short and every link is a closed form.  A concentration-response
# function turns ug/m3 into a relative risk,
#
#     RR(C) = exp( beta * max(0, C - C_counterfactual) ),
#     beta  = ln(RR_per_10) / 10,
#
# the relative risk turns into the population attributable fraction,
#
#     AF = (RR - 1) / RR  in [0, 1),  exactly 0 when RR == 1,
#
# and the attributable fraction multiplies the baseline death count.  Nothing
# here is fitted or iterated; the suite pins AF against its own algebra and
# against the identity ``AF(RR) == 1 - 1/RR``.
# ---------------------------------------------------------------------------

#: Concentration-response functions for long-term exposure, as relative risk
#: per 10 ug/m3 above a counterfactual.  Values follow WHO HRAPIE / AirQ+ for
#: all-cause adult mortality.  ``NOx`` carries the NO2 function, because NO2
#: is the marker the epidemiology is estimated on.
CONCENTRATION_RESPONSE = {
    "PM2.5": {"rrPer10": 1.062, "counterfactualUgM3": 0.0,
              "endpoint": "all-cause mortality, adults 30+",
              "source": "WHO HRAPIE (2013)"},
    "PM10": {"rrPer10": 1.041, "counterfactualUgM3": 0.0,
             "endpoint": "all-cause mortality, adults 30+",
             "source": "WHO HRAPIE (2013)"},
    "NOx": {"rrPer10": 1.055, "counterfactualUgM3": 20.0,
            "endpoint": "all-cause mortality, adults 30+ (NO2 as marker)",
            "source": "WHO HRAPIE (2013), 20 ug/m3 cut-off"},
}

#: Pollutants a health calculation can be run for, in reporting order.
HEALTH_POLLUTANTS = ("PM2.5", "PM10", "NOx")


def relative_risk(pollutant: str, concentration_ug_m3: float,
                  rr_per_10: Optional[float] = None,
                  counterfactual_ug_m3: Optional[float] = None) -> float:
    """Relative risk from a log-linear concentration-response function."""
    pol = _key("health pollutant", pollutant, HEALTH_POLLUTANTS)
    crf = CONCENTRATION_RESPONSE[pol]
    conc = _non_negative(concentration_ug_m3, "concentration_ug_m3")
    rr10 = (crf["rrPer10"] if rr_per_10 is None
            else _positive(rr_per_10, "rr_per_10"))
    if rr10 < 1.0:
        raise EmissionsError("rr_per_10 must be at least 1 (got %g)." % rr10)
    cut = (crf["counterfactualUgM3"] if counterfactual_ug_m3 is None
           else _non_negative(counterfactual_ug_m3, "counterfactual_ug_m3"))
    excess = conc - cut
    if excess <= 0.0:                   # exact test: no exposure above the cut
        return 1.0
    beta = math.log(rr10) / 10.0
    return math.exp(beta * excess)


def attributable_fraction(rr: float) -> float:
    """Population attributable fraction AF = (RR - 1)/RR, exactly 0 at RR == 1."""
    r = _as_float(rr, "rr")
    if r < 1.0:
        raise EmissionsError("relative risk must be at least 1 (got %g)." % r)
    if r == 1.0:                        # exact-zero branch, never an epsilon
        return 0.0
    return (r - 1.0) / r


def premature_deaths(population: float, baseline_rate_per_100k: float,
                     attributable: float) -> float:
    """Attributable deaths: population x baseline rate x attributable fraction."""
    pop = _non_negative(population, "population")
    rate = _non_negative(baseline_rate_per_100k, "baseline_rate_per_100k")
    af = _fraction(attributable, "attributable")
    return pop * rate / 1.0e5 * af


def years_of_life_lost(deaths: float, years_per_death: float = 10.6) -> float:
    """YLL proxy: attributable deaths times the mean years lost per death."""
    d = _non_negative(deaths, "deaths")
    years = _non_negative(years_per_death, "years_per_death")
    return d * years


@dataclass
class HealthImpact:
    """One pollutant's long-term health burden over an exposed population."""

    pollutant: str
    concentration_ug_m3: float
    counterfactual_ug_m3: float
    population: float
    baseline_rate_per_100k: float
    relative_risk: float
    attributable_fraction: float
    deaths: float
    years_of_life_lost: float
    monetised: float = 0.0
    endpoint: str = ""

    @property
    def deaths_per_100k(self) -> float:
        """Attributable deaths per 100 000 people exposed."""
        if self.population == 0.0:
            return 0.0
        return 1.0e5 * self.deaths / self.population

    def summary(self) -> Dict[str, Any]:
        """JSON-able description of the health burden."""
        return {"pollutant": self.pollutant,
                "concentrationUgM3": self.concentration_ug_m3,
                "counterfactualUgM3": self.counterfactual_ug_m3,
                "population": self.population,
                "baselineRatePer100k": self.baseline_rate_per_100k,
                "relativeRisk": self.relative_risk,
                "attributableFraction": self.attributable_fraction,
                "deaths": self.deaths,
                "deathsPer100k": self.deaths_per_100k,
                "yearsOfLifeLost": self.years_of_life_lost,
                "monetised": self.monetised,
                "endpoint": self.endpoint}


def health_impact_assessment(pollutant: str, concentration_ug_m3: float,
                             population: float,
                             baseline_rate_per_100k: float = 1000.0,
                             rr_per_10: Optional[float] = None,
                             counterfactual_ug_m3: Optional[float] = None,
                             years_per_death: float = 10.6,
                             value_per_life_year: float = 0.0) -> HealthImpact:
    """Full chain: concentration -> RR -> AF -> deaths -> YLL -> monetised value."""
    pol = _key("health pollutant", pollutant, HEALTH_POLLUTANTS)
    crf = CONCENTRATION_RESPONSE[pol]
    cut = (crf["counterfactualUgM3"] if counterfactual_ug_m3 is None
           else _non_negative(counterfactual_ug_m3, "counterfactual_ug_m3"))
    rr = relative_risk(pol, concentration_ug_m3, rr_per_10, cut)
    af = attributable_fraction(rr)
    deaths = premature_deaths(population, baseline_rate_per_100k, af)
    yll = years_of_life_lost(deaths, years_per_death)
    vly = _non_negative(value_per_life_year, "value_per_life_year")
    return HealthImpact(pollutant=pol,
                        concentration_ug_m3=_non_negative(
                            concentration_ug_m3, "concentration_ug_m3"),
                        counterfactual_ug_m3=cut,
                        population=_non_negative(population, "population"),
                        baseline_rate_per_100k=_non_negative(
                            baseline_rate_per_100k, "baseline_rate_per_100k"),
                        relative_risk=rr, attributable_fraction=af,
                        deaths=deaths, years_of_life_lost=yll,
                        monetised=yll * vly, endpoint=crf["endpoint"])


# ===========================================================================
# 15. CARBON ACCOUNTING AND THE SOCIAL COST OF CARBON
# ===========================================================================
def co2_equivalent(masses_g: Dict[str, float],
                   gwp: Optional[Dict[str, float]] = None) -> float:
    """CO2-equivalent mass in grams: sum of mass x GWP100, summed in key order."""
    if not isinstance(masses_g, dict):
        raise EmissionsError("masses_g must be a dict of gas -> grams.")
    table = GWP100 if gwp is None else gwp
    if not isinstance(table, dict):
        raise EmissionsError("gwp must be a dict of gas -> potential.")
    keys = sorted(masses_g.keys())      # never a dict iteration order
    terms = []
    for gas in keys:
        if gas not in table:
            raise EmissionsError("no global warming potential for %r "
                                 "(known: %s)."
                                 % (gas, ", ".join(sorted(table.keys()))))
        mass = _non_negative(masses_g[gas], "mass of %s" % gas)
        terms.append(mass * _non_negative(table[gas], "GWP of %s" % gas))
    return math.fsum(terms)


def discount_factor(discount_rate: float, years: float) -> float:
    """Present-value factor 1 / (1 + r)^t -- exactly 1 at t = 0 or r = 0."""
    r = _as_float(discount_rate, "discount_rate")
    t = _as_float(years, "years")
    if r <= -1.0:
        raise EmissionsError("discount_rate must exceed -1 (got %g)." % r)
    if r == 0.0 or t == 0.0:            # exact branches, never an epsilon
        return 1.0
    return (1.0 + r) ** (-t)


def social_cost_of_carbon(tonnes_co2e: float, price_per_tonne: float = 120.0,
                          years: float = 0.0,
                          discount_rate: float = 0.035,
                          price_growth: float = 0.0) -> float:
    """Discounted social cost of a tonnage of CO2e emitted ``years`` from now."""
    tonnes = _non_negative(tonnes_co2e, "tonnes_co2e")
    price = _non_negative(price_per_tonne, "price_per_tonne")
    t = _non_negative(years, "years")
    growth = _as_float(price_growth, "price_growth")
    if growth <= -1.0:
        raise EmissionsError("price_growth must exceed -1 (got %g)." % growth)
    escalated = price * ((1.0 + growth) ** t if growth != 0.0 else 1.0)
    return tonnes * escalated * discount_factor(discount_rate, t)


@dataclass
class CarbonAccount:
    """A stream of annual CO2e tonnages valued at a rising, discounted price."""

    years: List[float]
    tonnes: List[float]
    prices: List[float]
    undiscounted: List[float]
    discounted: List[float]
    discount_rate: float
    name: str = "carbon account"

    @property
    def total_tonnes(self) -> float:
        """Total CO2e over the appraisal period, in tonnes."""
        return math.fsum(self.tonnes)

    @property
    def net_present_value(self) -> float:
        """Present value of the whole stream."""
        return math.fsum(self.discounted)

    @property
    def undiscounted_total(self) -> float:
        """Nominal value of the whole stream."""
        return math.fsum(self.undiscounted)

    def effective_price(self) -> float:
        """NPV per tonne emitted (zero for an empty tonnage)."""
        total = self.total_tonnes
        if total == 0.0:
            return 0.0
        return self.net_present_value / total

    def summary(self) -> Dict[str, Any]:
        """JSON-able description of the carbon account."""
        return {"name": self.name,
                "discountRate": self.discount_rate,
                "totalTonnes": self.total_tonnes,
                "netPresentValue": self.net_present_value,
                "undiscountedTotal": self.undiscounted_total,
                "effectivePrice": self.effective_price(),
                "years": [{"year": self.years[i], "tonnes": self.tonnes[i],
                           "price": self.prices[i],
                           "undiscounted": self.undiscounted[i],
                           "discounted": self.discounted[i]}
                          for i in range(len(self.years))]}


def carbon_account(annual_tonnes: Sequence[float],
                   price_per_tonne: float = 120.0,
                   discount_rate: float = 0.035,
                   price_growth: float = 0.02,
                   base_year: float = 0.0,
                   name: str = "carbon account") -> CarbonAccount:
    """Value a stream of annual CO2e tonnages at a rising, discounted price."""
    tonnes = _as_floats(annual_tonnes, "annual_tonnes")
    _check_len(tonnes, 1, "annual_tonnes")
    for i, v in enumerate(tonnes):
        if v < 0.0:
            raise EmissionsError("annual_tonnes[%d] must be non-negative "
                                 "(got %g)." % (i, v))
    price = _non_negative(price_per_tonne, "price_per_tonne")
    rate = _as_float(discount_rate, "discount_rate")
    growth = _as_float(price_growth, "price_growth")
    start = _as_float(base_year, "base_year")
    years, prices, gross, net = [], [], [], []
    for i, tn in enumerate(tonnes):
        t = float(i)
        escalated = price * ((1.0 + growth) ** t if growth != 0.0 else 1.0)
        years.append(start + t)
        prices.append(escalated)
        gross.append(tn * escalated)
        net.append(tn * escalated * discount_factor(rate, t))
    return CarbonAccount(years=years, tonnes=tonnes, prices=prices,
                         undiscounted=gross, discounted=net,
                         discount_rate=rate, name=name)


# ===========================================================================
# 16. URBAN HEAT AND THERMAL COMFORT
# ===========================================================================
# A street is also a radiation trap.  How much sky a pedestrian can see sets
# how much long-wave the pavement can lose, and for an infinitely long canyon
# that view factor has an exact closed form: the sky subtends a strip between
# the two roof edges, and the two-dimensional view factor to a strip is the
# sine of its half-angle from the vertical,
#
#     SVF = (W/2) / sqrt(H^2 + (W/2)^2) = 1 / sqrt(1 + (2H/W)^2).
#
# It is exactly 1 for an open field and tends to 0 for a slot canyon; the
# suite checks both limits and the asymmetric generalisation against it.
# ---------------------------------------------------------------------------

#: Stefan-Boltzmann constant, W/(m^2 K^4).
_STEFAN_BOLTZMANN = 5.670374419e-8

#: The UTCI reference condition: 0.5 m/s at 10 m, and a vapour pressure equal
#: to 50 % relative humidity at the air temperature.
_UTCI_REFERENCE_WIND_MS = 0.5
#: The wind range the UTCI is defined over.
_UTCI_WIND_MIN_MS = 0.5
_UTCI_WIND_MAX_MS = 17.0
#: The PET reference wind speed, m/s.
_PET_REFERENCE_WIND_MS = 0.1


def sky_view_factor(height_m: float, width_m: float) -> float:
    """Sky view factor at the centre of a symmetric infinite street canyon."""
    h = _non_negative(height_m, "height_m")
    w = _positive(width_m, "width_m")
    half = 0.5 * w
    return half / math.hypot(h, half)


def sky_view_factor_asymmetric(left_height_m: float, right_height_m: float,
                               width_m: float) -> float:
    """Sky view factor at the centre of a canyon with unequal facade heights."""
    hl = _non_negative(left_height_m, "left_height_m")
    hr = _non_negative(right_height_m, "right_height_m")
    w = _positive(width_m, "width_m")
    half = 0.5 * w
    return 0.5 * (half / math.hypot(hl, half) + half / math.hypot(hr, half))


def saturation_vapour_pressure(temperature_c: float) -> float:
    """Saturation vapour pressure in hPa (WMO / Magnus-Tetens over water).

    ``6.112 exp(17.62 t / (243.12 + t))`` is exact at 0 C by construction and
    runs 0.17-0.29 % LOW against the accepted table between 10 and 30 C
    (23.3260 against 23.388 hPa at 20 C).  That is the accuracy of this
    two-parameter fit, not a transcription error: Buck's three-parameter form
    lands within 0.03 % over the same range.  A test against published values
    needs at least 3e-3 relative room.
    """
    t = _as_float(temperature_c, "temperature_c")
    if t <= -243.12:
        raise EmissionsError("temperature_c is below the Magnus pole "
                             "(got %g)." % t)
    return 6.112 * math.exp(17.62 * t / (243.12 + t))


def vapour_pressure(temperature_c: float,
                    relative_humidity_pct: float = 50.0) -> float:
    """Actual vapour pressure in hPa from temperature and relative humidity."""
    rh = _non_negative(relative_humidity_pct, "relative_humidity_pct")
    if rh > 100.0:
        raise EmissionsError("relative_humidity_pct must lie in [0, 100] "
                             "(got %g)." % rh)
    return 0.01 * rh * saturation_vapour_pressure(temperature_c)


def wind_speed_at_height(reference_speed_ms: float, height_m: float,
                         reference_height_m: float = 10.0,
                         roughness_length_m: float = 0.5) -> float:
    """Logarithmic wind profile -- exactly the reference speed at the reference height."""
    v_ref = _non_negative(reference_speed_ms, "reference_speed_ms")
    z = _positive(height_m, "height_m")
    z_ref = _positive(reference_height_m, "reference_height_m")
    z0 = _positive(roughness_length_m, "roughness_length_m")
    if z <= z0 or z_ref <= z0:
        raise EmissionsError("both heights must exceed the roughness length "
                             "%g m (got %g and %g)." % (z0, z, z_ref))
    return v_ref * math.log(z / z0) / math.log(z_ref / z0)


def mean_radiant_temperature(air_temperature_c: float,
                             global_radiation_w_m2: float = 0.0,
                             sky_view: float = 1.0,
                             relative_humidity_pct: float = 50.0,
                             albedo: float = 0.20,
                             surface_excess_c: float = 5.0,
                             diffuse_fraction: float = 0.25,
                             absorptivity: float = 0.70,
                             emissivity: float = 0.97,
                             surface_emissivity: float = 0.95,
                             projected_area_factor: float = 0.22,
                             sky_emissivity: Optional[float] = None) -> float:
    """Mean radiant temperature in degrees C from the radiation budget on a body.

    Long-wave comes from the sky through the fraction ``sky_view`` of the
    hemisphere and from the surrounding surfaces through the rest; the sky's
    emissivity follows Brutsaert unless one is supplied.  Short-wave splits
    the quoted global irradiance into a beam (weighted by the projected area
    factor of a standing person), an isotropic diffuse half and a ground
    reflection.  Forcing ``sky_emissivity = 1``, ``sky_view = 1`` and zero
    radiation returns the air temperature EXACTLY, which is the identity the
    suite uses to pin the whole budget.
    """
    t_c = _as_float(air_temperature_c, "air_temperature_c")
    t_k = t_c + 273.15
    if t_k <= 0.0:
        raise EmissionsError("air_temperature_c must be above absolute zero.")
    glob = _non_negative(global_radiation_w_m2, "global_radiation_w_m2")
    svf = _fraction(sky_view, "sky_view")
    alb = _fraction(albedo, "albedo")
    excess = _as_float(surface_excess_c, "surface_excess_c")
    f_diff = _fraction(diffuse_fraction, "diffuse_fraction")
    a_k = _fraction(absorptivity, "absorptivity")
    eps_p = _positive(emissivity, "emissivity")
    eps_s = _positive(surface_emissivity, "surface_emissivity")
    f_p = _fraction(projected_area_factor, "projected_area_factor")
    e_hpa = vapour_pressure(t_c, relative_humidity_pct)
    if sky_emissivity is None:
        eps_sky = 1.24 * (e_hpa / t_k) ** (1.0 / 7.0)
        if eps_sky > 1.0:
            eps_sky = 1.0
    else:
        eps_sky = _positive(sky_emissivity, "sky_emissivity")
    t_surf_k = t_k + excess
    if t_surf_k <= 0.0:
        raise EmissionsError("surface temperature falls below absolute zero.")
    l_sky = eps_sky * _STEFAN_BOLTZMANN * t_k ** 4
    l_surf = eps_s * _STEFAN_BOLTZMANN * t_surf_k ** 4
    l_in = svf * l_sky + (1.0 - svf) * l_surf
    diffuse = f_diff * glob
    beam = glob - diffuse
    s_str = f_p * beam + 0.5 * diffuse + 0.5 * alb * glob
    total = l_in + (a_k / eps_p) * s_str
    return (total / _STEFAN_BOLTZMANN) ** 0.25 - 273.15


def utci_lite(air_temperature_c: float, mean_radiant_c: float,
              wind_speed_ms: float = _UTCI_REFERENCE_WIND_MS,
              relative_humidity_pct: float = 50.0) -> float:
    """Reduced UTCI surrogate in degrees C -- exact at the reference condition.

    The published Universal Thermal Climate Index is a sixth-order polynomial
    in four variables with more than two hundred terms.  This is a deliberate
    three-term surrogate: a wind-damped radiant loading, a wind-chill offset
    that vanishes at the 0.5 m/s reference wind, and a humidity offset that
    vanishes at 50 % relative humidity and only bites above 20 C.  It
    reproduces the reference condition EXACTLY (``Tmrt == Ta``, 0.5 m/s,
    50 % RH gives ``UTCI == Ta``) and the published sign and rough magnitude
    of each offset -- not the official polynomial, and it does not pretend to.

    The magnitude gap is real and is the price of three terms: at the published
    Ta = 30 C, Tmrt = 50 C, v = 1 m/s, RH = 50 % test point the official index
    is near 38 C and this surrogate returns 35.97.  Any value test on this
    function must leave at least 3 C of room and must say in its own text that
    it is bounding a surrogate, not pinning UTCI.  The one thing that IS exact
    is the reference identity.
    """
    t_a = _as_float(air_temperature_c, "air_temperature_c")
    t_mrt = _as_float(mean_radiant_c, "mean_radiant_c")
    va_raw = _non_negative(wind_speed_ms, "wind_speed_ms")
    e_hpa = vapour_pressure(t_a, relative_humidity_pct)
    e_ref = vapour_pressure(t_a, 50.0)
    va = va_raw
    if va < _UTCI_WIND_MIN_MS:
        va = _UTCI_WIND_MIN_MS
    elif va > _UTCI_WIND_MAX_MS:
        va = _UTCI_WIND_MAX_MS
    radiant = 0.35 * math.exp(-0.12 * va) * (t_mrt - t_a)
    chill = -(0.80 + 0.04 * max(0.0, 25.0 - t_a)) * \
        (math.sqrt(va) - math.sqrt(_UTCI_REFERENCE_WIND_MS))
    humid = 0.0015 * (e_hpa - e_ref) * max(0.0, t_a - 20.0)
    return t_a + radiant + chill + humid


def pet_lite(air_temperature_c: float, mean_radiant_c: float,
             wind_speed_ms: float = _PET_REFERENCE_WIND_MS,
             relative_humidity_pct: float = 50.0,
             emissivity: float = 0.97) -> float:
    """Physiologically Equivalent Temperature surrogate built on operative temperature.

    The core is the genuine operative temperature -- the convection- and
    radiation-weighted blend ``(h_c Ta + h_r Tmrt) / (h_c + h_r)`` with the
    linearised radiative coefficient ``h_r = 4 eps sigma T^3`` and the
    standard forced-convection coefficient ``h_c = 12.1 sqrt(v)``.  A humidity
    offset that vanishes at 50 % relative humidity is added on top.  Because
    the two weights sum to one, ``Tmrt == Ta`` at 50 % RH returns ``Ta``
    EXACTLY, whatever the wind.
    """
    t_a = _as_float(air_temperature_c, "air_temperature_c")
    t_mrt = _as_float(mean_radiant_c, "mean_radiant_c")
    va = _non_negative(wind_speed_ms, "wind_speed_ms")
    eps = _positive(emissivity, "emissivity")
    t_k = t_a + 273.15
    if t_k <= 0.0:
        raise EmissionsError("air_temperature_c must be above absolute zero.")
    if va < _PET_REFERENCE_WIND_MS:
        va = _PET_REFERENCE_WIND_MS
    h_r = 4.0 * eps * _STEFAN_BOLTZMANN * t_k ** 3
    h_c = 12.1 * math.sqrt(va)
    operative = (h_c * t_a + h_r * t_mrt) / (h_c + h_r)
    e_hpa = vapour_pressure(t_a, relative_humidity_pct)
    e_ref = vapour_pressure(t_a, 50.0)
    humid = 0.0022 * (e_hpa - e_ref) * max(0.0, operative - 18.0)
    return operative + humid


#: The published UTCI thermal-stress bands, as ``(lower bound, label)`` with
#: the bounds ascending.  The first entry is the open-ended cold tail.
_UTCI_BANDS = (
    (float("-inf"), "extreme cold stress"),
    (-40.0, "very strong cold stress"),
    (-27.0, "strong cold stress"),
    (-13.0, "moderate cold stress"),
    (0.0, "slight cold stress"),
    (9.0, "no thermal stress"),
    (26.0, "moderate heat stress"),
    (32.0, "strong heat stress"),
    (38.0, "very strong heat stress"),
    (46.0, "extreme heat stress"),
)


def thermal_stress_category(utci_c: float) -> str:
    """UTCI thermal-stress band for an index value in degrees C."""
    value = _as_float(utci_c, "utci_c")
    label = _UTCI_BANDS[0][1]
    for lower, name in _UTCI_BANDS:
        if value >= lower:
            label = name
        else:
            break
    return label


# ===========================================================================
# 17. FRONTEND PAYLOADS
# ===========================================================================
def _json_safe(x: Any) -> Any:
    """Replace non-finite floats with None so the dict survives json.dumps."""
    if isinstance(x, float):
        if math.isinf(x) or math.isnan(x):
            return None
        return round(x, 10)
    return x


def _series(pairs: Sequence[Tuple[float, float]], x_name: str, y_name: str,
            max_points: int = 240) -> List[Dict[str, Any]]:
    """Thin a series of pairs to a JSON-able list of labelled points."""
    items = list(pairs)
    stride = max(1, len(items) // max_points)
    return [{x_name: _json_safe(float(items[i][0])),
             y_name: _json_safe(float(items[i][1]))}
            for i in range(0, len(items), stride)]


def build_emissions_environment_payload(
        fleet: FleetComposition, *,
        pollutant: str = "NOx",
        speed_range_kmh: Tuple[float, float] = (5.0, 120.0),
        curve_points: int = 40,
        modal: Optional[ModalResult] = None,
        fuel: Optional[FuelResult] = None,
        cold_start: Optional[ColdStartResult] = None,
        calibration: Optional[ProxyCalibration] = None,
        rank: Optional[RankConsistency] = None,
        non_exhaust: Optional[NonExhaustResult] = None,
        noise: Optional[NoiseResult] = None,
        noise_profile: Optional[Sequence[Tuple[float, float]]] = None,
        dispersion_profile: Optional[Sequence[Tuple[float, float]]] = None,
        canyon: Optional[CanyonResult] = None,
        zones: Optional[Sequence[ExposureZone]] = None,
        health: Optional[HealthImpact] = None,
        carbon: Optional[CarbonAccount] = None) -> Dict[str, Any]:
    """Frontend-ready JSON dict: emission curves, noise, dispersion, exposure, carbon."""
    if not isinstance(fleet, FleetComposition):
        raise EmissionsError("fleet must be a FleetComposition.")
    pol = _key("pollutant", pollutant, POLLUTANTS)
    lo = _positive(speed_range_kmh[0], "speed_range_kmh[0]")
    hi = _positive(speed_range_kmh[1], "speed_range_kmh[1]")
    if hi <= lo:
        raise EmissionsError("speed_range_kmh must be increasing (%g, %g)."
                             % (lo, hi))
    n = int(curve_points)
    if n < 2:
        raise EmissionsError("curve_points must be at least 2 (got %d)." % n)
    step = (hi - lo) / float(n - 1)
    curve = []
    for i in range(n):
        v = lo + step * i if i < n - 1 else hi
        curve.append({"speedKmh": _json_safe(v),
                      "efGPerKm": _json_safe(
                          fleet_emission_factor(fleet, pol, v))})
    payload = {
        "pollutant": pol,
        "pollutants": list(POLLUTANTS),
        "vehicleCategories": list(VEHICLE_CATEGORIES),
        "euroClasses": list(EURO_CLASSES),
        "fleet": fleet.summary(),
        "emissionCurve": curve,
        "stabilityClasses": [{"class": c, "description": stability_description(c)}
                             for c in STABILITY_CLASSES],
    }  # type: Dict[str, Any]
    if modal is not None:
        payload["modal"] = modal.summary()
    if fuel is not None:
        payload["fuel"] = fuel.summary()
    if cold_start is not None:
        payload["coldStart"] = cold_start.summary()
    if calibration is not None:
        payload["proxyCalibration"] = calibration.summary()
    if rank is not None:
        payload["rankConsistency"] = rank.summary()
    if non_exhaust is not None:
        payload["nonExhaust"] = non_exhaust.summary()
    if noise is not None:
        payload["noise"] = noise.summary()
    if noise_profile is not None:
        payload["noiseProfile"] = _series(noise_profile, "distanceM", "levelDb")
    if dispersion_profile is not None:
        payload["dispersionProfile"] = _series(dispersion_profile,
                                               "distanceM", "ugPerM3")
    if canyon is not None:
        payload["canyon"] = canyon.summary()
    if zones is not None:
        ordered = _ordered_zones(zones)
        payload["exposure"] = {
            "populationWeightedMean": _json_safe(
                population_weighted_exposure(ordered)),
            "zones": [z.summary() for z in ordered],
        }
    if health is not None:
        payload["health"] = health.summary()
    if carbon is not None:
        payload["carbon"] = carbon.summary()
    return payload


def build_a2ui_payload(fleet: FleetComposition, *,
                       pollutant: str = "NOx",
                       modal: Optional[ModalResult] = None,
                       fuel: Optional[FuelResult] = None,
                       noise: Optional[NoiseResult] = None,
                       noise_profile: Optional[Sequence[Tuple[float, float]]] = None,
                       dispersion_profile: Optional[Sequence[Tuple[float, float]]] = None,
                       canyon: Optional[CanyonResult] = None,
                       health: Optional[HealthImpact] = None,
                       carbon: Optional[CarbonAccount] = None,
                       l_den_db: Optional[float] = None,
                       utci_c: Optional[float] = None) -> Dict[str, Any]:
    """Visualisation-ready dict: stat cards, chart series and colour-coded bands."""
    if not isinstance(fleet, FleetComposition):
        raise EmissionsError("fleet must be a FleetComposition.")
    pol = _key("pollutant", pollutant, POLLUTANTS)
    urban_ef = fleet_emission_factor(fleet, pol, 30.0)
    cards = [
        {"label": "Fleet EF at 30 km/h", "value": round(urban_ef, 4),
         "unit": "g/km %s" % pol},
        {"label": "Fleet entries", "value": len(fleet.keys_sorted()),
         "unit": "category x Euro"},
    ]  # type: List[Dict[str, Any]]
    if modal is not None:
        cards.append({"label": "Modal %s" % pol,
                      "value": round(modal.grams_per_km().get(pol, 0.0), 4),
                      "unit": "g/km"})
    if fuel is not None:
        cards.append({"label": "Fuel intensity",
                      "value": round(fuel.litres_per_100km, 2),
                      "unit": "L/100 km"})
    if noise is not None:
        cards.append({"label": "Facade noise",
                      "value": round(facade_level(noise.level_db), 1),
                      "unit": "dB(A)"})
    if l_den_db is not None:
        cards.append({"label": "Highly annoyed",
                      "value": round(percent_highly_annoyed(l_den_db), 1),
                      "unit": "%% at L_den %.1f" % l_den_db})
    if canyon is not None:
        cards.append({"label": "Leeward facade",
                      "value": round(canyon.leeward_ug_m3, 1),
                      "unit": "ug/m3"})
    if health is not None:
        cards.append({"label": "Attributable deaths",
                      "value": round(health.deaths, 1), "unit": "per year"})
    if carbon is not None:
        cards.append({"label": "Carbon NPV",
                      "value": round(carbon.net_present_value, 0),
                      "unit": "currency"})
    if utci_c is not None:
        cards.append({"label": "UTCI", "value": round(utci_c, 1),
                      "unit": thermal_stress_category(utci_c)})
    charts = {
        "emissionCurve": [
            {"speedKmh": _json_safe(5.0 + 2.5 * i),
             "efGPerKm": _json_safe(
                 fleet_emission_factor(fleet, pol, 5.0 + 2.5 * i))}
            for i in range(47)],
    }  # type: Dict[str, Any]
    if modal is not None:
        frac = modal.opmodes.fractions()
        charts["operatingModes"] = [
            {"opMode": b, "label": opmode_label(b),
             "fraction": _json_safe(frac.get(b, 0.0))}
            for b in modal.opmodes.bins()]
        charts["modalPollutants"] = [
            {"pollutant": p, "grams": _json_safe(modal.grams[p])}
            for p in modal.pollutants]
    if fuel is not None:
        charts["fuelModes"] = [
            {"mode": "idle", "litres": _json_safe(fuel.idle_litres)},
            {"mode": "cruise", "litres": _json_safe(fuel.cruise_litres)},
            {"mode": "accelerate", "litres": _json_safe(fuel.accelerate_litres)},
            {"mode": "decelerate", "litres": _json_safe(fuel.decelerate_litres)},
        ]
    if noise_profile is not None:
        charts["noiseProfile"] = _series(noise_profile, "distanceM", "levelDb")
    if dispersion_profile is not None:
        charts["dispersionProfile"] = _series(dispersion_profile,
                                              "distanceM", "ugPerM3")
    if canyon is not None:
        charts["canyonFacades"] = [
            {"facade": "leeward", "ugPerM3": _json_safe(canyon.leeward_ug_m3)},
            {"facade": "windward", "ugPerM3": _json_safe(canyon.windward_ug_m3)},
            {"facade": "background",
             "ugPerM3": _json_safe(canyon.background_ug_m3)},
        ]
    if carbon is not None:
        charts["carbon"] = [
            {"year": _json_safe(carbon.years[i]),
             "discounted": _json_safe(carbon.discounted[i])}
            for i in range(len(carbon.years))]
    return {
        "title": "Emissions and environment -- %s" % fleet.name,
        "statCards": cards,
        "charts": charts,
        "thermalBands": [{"lower": (None if math.isinf(lo) else lo),
                          "label": name}
                         for lo, name in _UTCI_BANDS],
        "healthPollutants": list(HEALTH_POLLUTANTS),
    }


# ===========================================================================
# 18. DEMO
# ===========================================================================
def _demo() -> None:
    import json                        # demo-only, keeps the module lean

    print("=" * 74)
    print("  EMISSIONS AND ENVIRONMENT (pure Python) - demo")
    print("=" * 74)

    # -- 1. units, constants and registries --------------------------------
    print("\n-- 1. units, constants and registries --")
    print("   pollutants %s" % (POLLUTANTS,))
    print("   categories %s" % (VEHICLE_CATEGORIES,))
    print("   Euro classes %s" % (EURO_CLASSES,))
    print("   GWP100: %s" % {k: GWP100[k] for k in sorted(GWP100)})
    print("   fuels: %s" % ", ".join(
        "%s %.3f kg/L, %.1f MJ/kg, %.0f gCO2/L"
        % (f, FUEL_PROPERTIES[f]["density_kg_l"],
           FUEL_PROPERTIES[f]["lhv_mj_kg"], FUEL_PROPERTIES[f]["co2_g_per_l"])
        for f in sorted(FUEL_PROPERTIES)))
    print("   50 km/h = %.4f m/s = %.4f mph; 30 mph = %.4f km/h (back %.4f)"
          % (kmh_to_ms(50.0), kmh_to_mph(50.0), mph_to_kmh(30.0),
             ms_to_kmh(kmh_to_ms(50.0))))
    print("   1.25e6 g = %.4f t | 4.2e-5 g/m3 = %.2f ug/m3 (back %.3e g/m3)"
          % (grams_to_tonnes(1.25e6), ug_per_m3(4.2e-5),
             g_per_m3(ug_per_m3(4.2e-5))))
    print("   6.4 L of diesel -> %.0f g CO2" % fuel_litres_to_co2_g(6.4,
                                                                   "diesel"))

    # -- 2. speed-dependent emission factors -------------------------------
    print("\n-- 2. COPERT-style speed curves: the U shape is the whole point --")
    print("   %d base curves, %d Euro reduction factors"
          % (len(BASE_EMISSION_CURVES), len(EURO_REDUCTION_FACTORS)))
    print("   %-26s %9s %9s %9s %9s" % ("curve (Euro 5 base)", "v*",
                                          "EF(v*)", "EF(v_min)", "EF(v_max)"))
    for cat in ("passenger_car_petrol", "passenger_car_diesel", "hgv_diesel"):
        for pol in ("NOx", "CO2"):
            curve = emission_factor_curve(cat, pol)
            v_star, ef_star = curve_minimum(curve, 401, 80)
            print("   %-26s %9.3f %9.4f %9.4f %9.4f"
                  % ("%s/%s" % (cat, pol), v_star, ef_star,
                     curve.factor(curve.v_min), curve.factor(curve.v_max)))
    print("   every minimum above is INTERIOR -- that U shape is why "
          "congestion costs air quality twice")
    petrol_nox = emission_factor_curve("passenger_car_petrol", "NOx")
    print("   curve description: %s" % petrol_nox.describe()["coefficients"])
    print("   Euro reduction on NOx: %s"
          % ["%s %.2f" % (e, euro_reduction_factor(e, "NOx"))
             for e in EURO_CLASSES])
    print("   EF(NOx, diesel car, Euro 5) at 20/50/90 km/h = %.4f / %.4f / "
          "%.4f g/km" % tuple(
              emission_factor("NOx", "passenger_car_diesel", "Euro 5", v)
              for v in (20.0, 50.0, 90.0)))
    table = emission_factor_table("NOx", "passenger_car_diesel", "Euro 6",
                                  n_samples=5)
    print("   table sample: %s"
          % ["%.0f:%.4f" % (r["speedKmh"], r["efGPerKm"]) for r in table])
    one = hot_emissions("NOx", "passenger_car_diesel", "Euro 5", 30.0, 2.5)
    ten = hot_emissions("NOx", "passenger_car_diesel", "Euro 5", 30.0, 2.5,
                        10.0)
    print("   LINEARITY in vehicle count: 1 veh %.5f g, 10 veh %.5f g, "
          "ratio exact %s g" % (one, ten, (ten - 10.0 * one) == 0.0))

    # -- 3. fleet mixing and cold start ------------------------------------
    print("\n-- 3. fleet mixing and the COPERT cold-start excess --")
    fleet = FleetComposition({
        ("passenger_car_petrol", "Euro 6"): 0.38,
        ("passenger_car_petrol", "Euro 5"): 0.07,
        ("passenger_car_diesel", "Euro 6"): 0.22,
        ("passenger_car_diesel", "Euro 5"): 0.11,
        ("lgv_diesel", "Euro 6"): 0.12,
        ("bus_diesel", "Euro 6"): 0.04,
        ("motorcycle", "Euro 4"): 0.06,
    }, "inner-city AM peak")
    print("   fleet %r: %d entries, shares sum to %.10f"
          % (fleet.name, len(fleet.keys_sorted()), fleet.total()))
    print("   by category: %s"
          % {k: round(v, 3) for k, v in sorted(
              fleet.category_shares().items())})
    print("   normalised total = %.10f" % fleet.normalised().total())
    for pol in ("NOx", "PM2.5", "CO2"):
        print("      fleet EF %-5s at 15/30/60 km/h = %9.4f / %9.4f / %9.4f "
              "g/km" % (pol, fleet_emission_factor(fleet, pol, 15.0),
                        fleet_emission_factor(fleet, pol, 30.0),
                        fleet_emission_factor(fleet, pol, 60.0)))
    print("   fleet NOx over 1200 veh x 0.8 km at 25 km/h = %.1f g"
          % fleet_hot_emissions(fleet, "NOx", 25.0, 0.8, 1200.0))
    print("   cold-start beta: 4 km trip at 20 C = %.4f, at 0 C = %.4f, "
          "20 km at 20 C = %.4f" % (cold_start_beta(4.0, 20.0),
                                    cold_start_beta(4.0, 0.0),
                                    cold_start_beta(20.0, 20.0)))
    print("   cold/hot ratio (CO, petrol car) at 20 C = %.3f, at -5 C = %.3f"
          % (cold_start_ratio("CO", 20.0), cold_start_ratio("CO", -5.0)))
    cs = cold_start_excess("CO", "passenger_car_petrol", "Euro 5", 25.0, 4.0,
                           600.0, 5.0)
    print("   600 cold starts, 4 km trips at 5 C: hot %.0f g, excess %.0f g "
          "(%.1f %% of total %.0f g)"
          % (cs.hot_grams, cs.excess_grams, 100.0 * cs.excess_share,
             cs.total_grams))

    # -- 4. VSP and the MOVES operating-mode bins --------------------------
    print("\n-- 4. Vehicle Specific Power and MOVES operating modes --")
    print("   coefficients: %s (m=%.4f t) | %s (m=%.1f t)"
          % (LIGHT_DUTY_VSP.name, LIGHT_DUTY_VSP.mass_tonnes,
             HEAVY_DUTY_VSP.name, HEAVY_DUTY_VSP.mass_tonnes))
    print("   VSP(0, 0) = %.1f kW/t EXACTLY (%s)"
          % (vehicle_specific_power(0.0, 0.0),
             vehicle_specific_power(0.0, 0.0) == 0.0))
    v = kmh_to_ms(50.0)
    closed = (LIGHT_DUTY_VSP.rolling_a * v
              + LIGHT_DUTY_VSP.rotating_b * v * v
              + LIGHT_DUTY_VSP.drag_c * v ** 3) / LIGHT_DUTY_VSP.mass_tonnes
    print("   cruise at 50 km/h flat: VSP %.6f kW/t, closed form %.6f, "
          "err %.2e" % (vehicle_specific_power(v), closed,
                        abs(vehicle_specific_power(v) - closed)))
    print("   grade +5 %% at 50 km/h: VSP %.4f kW/t | tractive power %.3f kW"
          % (vehicle_specific_power(v, 0.0, 0.05),
             road_load_power(v, 0.0, 0.05)))
    print("   %d MOVES bins: %s" % (len(OPMODE_IDS), OPMODE_IDS))
    for (sp, ac) in ((0.0, 0.0), (8.0, 0.2), (18.0, 0.0), (28.0, 0.5),
                     (20.0, -1.5)):
        vsp = vehicle_specific_power(sp, ac)
        b = operating_mode_bin(sp, ac, vsp)
        print("      v=%5.1f m/s a=%+5.2f m/s2 VSP=%+7.2f -> bin %2d %-24s "
              "(mid VSP %.1f, CO2 %.4f g/s)"
              % (sp, ac, vsp, b, opmode_label(b), opmode_vsp_midpoint(b),
                 moves_opmode_rate("CO2", b)))
    print("   rate parameters: %s"
          % {k: MOVES_RATE_PARAMS[k] for k in sorted(MOVES_RATE_PARAMS)})

    # -- 5. traces, drive cycles and modal emissions -----------------------
    print("\n-- 5. speed traces and second-by-second modal emissions --")
    cruise = constant_speed_cycle(60.0, 300.0)
    urban = stop_go_cycle(40.0, 6, cruise_s=35.0, idle_s=18.0)
    synth = synthetic_drive_cycle(seed=20240917, duration_s=900.0,
                                  target_mean_kmh=28.0)
    manual = trace_from_speeds([0.0, 3.0, 7.0, 11.0, 11.0, 6.0, 0.0, 0.0], 1.0)
    for tr in (cruise, urban, synth, manual):
        s = tr.summary()
        print("   %-10s %5d s %7.3f km  mean %5.1f km/h  running %5.1f km/h  "
              "idle %5.1f s  %2d stops  PKE %.3f"
              % (s["name"], int(s["durationS"]), s["distanceKm"],
                 s["meanSpeedKmh"], s["meanRunningSpeedKmh"], s["idleTimeS"],
                 s["stops"], s["pkeMs2"]))
    print("   seeded cycle is reproducible: %s"
          % (synthetic_drive_cycle(seed=20240917, duration_s=900.0,
                                   target_mean_kmh=28.0).speeds
             == synth.speeds))
    dist = opmode_distribution(urban, LIGHT_DUTY_VSP)
    print("   urban cycle occupies %d bins over %.0f s (idle %.0f s, "
          "braking %.0f s)" % (len(dist.bins()), dist.total_seconds(),
                               dist.idle_seconds(), dist.braking_seconds()))
    top = sorted(dist.fractions().items(), key=lambda kv: (-kv[1], kv[0]))[:4]
    print("   busiest bins: %s"
          % ["%d %s %.1f%%" % (b, opmode_label(b), 100.0 * f) for b, f in top])
    modal = modal_emissions(urban)
    print("   modal masses over the cycle: %s"
          % {p: round(modal.grams[p], 5) for p in modal.pollutants})
    print("   implied g/km: %s | g/h: %s"
          % ({p: round(v, 4) for p, v in modal.grams_per_km().items()},
             {p: round(v, 2) for p, v in modal.grams_per_hour().items()}))
    print("   modal CO2 %.1f g/km vs COPERT petrol Euro 5 at %.1f km/h "
          "%.1f g/km" % (modal.grams_per_km()["CO2"], modal.mean_speed_kmh,
                         emission_factor("CO2", "passenger_car_petrol",
                                         "Euro 5", modal.mean_speed_kmh)))

    # -- 6. fuel: Akcelik-Biggs four-mode ----------------------------------
    print("\n-- 6. fuel: the Akcelik-Biggs elemental model --")
    for params in (PETROL_CAR, DIESEL_CAR, URBAN_BUS):
        print("   %-18s idle %.3f mL/s, mass %.0f kg, fuel %s"
              % (params.name, params.alpha_ml_s, params.mass_kg, params.fuel))
    v50 = kmh_to_ms(50.0)
    print("   at 50 km/h flat: tractive %.3f kW, fuel rate %.4f mL/s"
          % (tractive_power(v50), instantaneous_fuel_rate(v50)))
    print("   accelerating at 1.2 m/s2: %.3f kW, %.4f mL/s"
          % (tractive_power(v50, 1.2), instantaneous_fuel_rate(v50, 1.2)))
    for (sp, ac) in ((0.2, 0.0), (v50, 0.0), (v50, 1.2), (v50, -1.2)):
        mode, rate = four_mode_fuel(sp, ac)
        print("      v=%5.2f a=%+5.2f -> %-11s %.4f mL/s" % (sp, ac, mode,
                                                             rate))
    fuel = fuel_from_trace(urban, PETROL_CAR)
    print("   urban cycle: %.4f L total (idle %.4f, cruise %.4f, accel %.4f, "
          "decel %.4f)" % (fuel.total_litres, fuel.idle_litres,
                           fuel.cruise_litres, fuel.accelerate_litres,
                           fuel.decelerate_litres))
    print("   -> %.2f L/100 km, %.1f g CO2, %.2f MJ"
          % (fuel.litres_per_100km, fuel.co2_grams, fuel.energy_mj()))
    elem = elemental_fuel_model(urban.distance_m() / 1000.0,
                                urban.idle_time_s(), urban.stop_count(), 40.0)
    print("   elemental F = f1 d + f2 idle + f3 stops = %.4f L "
          "(%.2f L/100 km) against %.4f L from the trace"
          % (elem.total_litres, elem.litres_per_100km, fuel.total_litres))

    # -- 7. the linear edge proxy and its calibration ----------------------
    print("\n-- 7. the edge proxy E = a f t + b idle + c stop --")
    specs = ((45.0, 2, 900.0), (30.0, 5, 1400.0), (20.0, 9, 1800.0),
             (55.0, 1, 700.0), (25.0, 7, 1600.0), (35.0, 4, 1200.0),
             (15.0, 11, 950.0), (50.0, 3, 1100.0))
    observations = []
    for i, (kmh, stops, flow) in enumerate(specs):
        tr = stop_go_cycle(kmh, stops, cruise_s=30.0, idle_s=16.0,
                           name="edge%d" % i)
        res = modal_emissions(tr, ("CO2",))
        observations.append(EdgeObservation(
            edge_id="e%02d" % i, flow_veh_h=flow,
            travel_time_h=tr.duration_s() / 3600.0,
            idle_veh_h=flow * tr.idle_time_s() / 3600.0,
            stops=flow * tr.stop_count(),
            modal_grams=flow * res.grams["CO2"]))
    cal = calibrate_edge_proxy(observations, "CO2")
    co = cal.coefficients
    print("   calibrated on %d edges: a=%.2f g/veh-h, b=%.2f g/idle-veh-h, "
          "c=%.4f g/stop" % (cal.n_observations, co.a, co.b, co.c))
    print("   R^2 = %.6f, RMSE = %.1f g, MAE = %.1f g, active terms %s"
          % (cal.r_squared, cal.rmse, cal.mean_abs_error, cal.active_terms))
    nn = calibrate_edge_proxy(observations, "CO2", non_negative=True)
    print("   sign-constrained refit: a=%.2f b=%.2f c=%.4f, R^2 %.6f"
          % (nn.coefficients.a, nn.coefficients.b, nn.coefficients.c,
             nn.r_squared))
    rank = proxy_rank_consistency(co, observations)
    print("   rank agreement with the modal model: tau_b = %.4f over %d "
          "pairs (%d concordant, %d discordant, monotone %s)"
          % (rank.tau_b, rank.n_pairs, rank.concordant, rank.discordant,
             rank.monotone))
    sample = observations[2]
    print("   edge %s proxy = %.1f g vs modal %.1f g"
          % (sample.edge_id,
             edge_emission_proxy(co, sample.flow_veh_h, sample.travel_time_h,
                                 sample.idle_veh_h, sample.stops),
             sample.modal_grams))
    print("   network total: proxy %.0f g vs modal %.0f g (%.2f %% apart)"
          % (network_emission_proxy(co, observations),
             math.fsum([o.modal_grams for o in observations]),
             100.0 * abs(network_emission_proxy(co, observations)
                         - math.fsum([o.modal_grams for o in observations]))
             / math.fsum([o.modal_grams for o in observations])))
    print("   proxy LINEARITY: doubling every term doubles E exactly (%s)"
          % (abs(edge_emission_proxy(co, 2.0 * sample.flow_veh_h,
                                     sample.travel_time_h,
                                     2.0 * sample.idle_veh_h,
                                     2.0 * sample.stops)
                 - 2.0 * edge_emission_proxy(co, sample.flow_veh_h,
                                             sample.travel_time_h,
                                             sample.idle_veh_h,
                                             sample.stops)) < 1e-9))
    print("   coefficients as a dict: %s" % co.summary())
    print("   one observation as a dict: %s" % sample.summary())

    # -- 8. non-exhaust particulate matter ---------------------------------
    print("\n-- 8. non-exhaust PM: brake, tyre and road wear --")
    print("   TSP factors (g/km): %s"
          % {k: NON_EXHAUST_TSP["passenger_car_petrol"][k]
             for k in sorted(NON_EXHAUST_TSP["passenger_car_petrol"])})
    print("   size fractions: %s"
          % {s: NON_EXHAUST_SIZE_FRACTIONS[s]["PM2.5"]
             for s in sorted(NON_EXHAUST_SIZE_FRACTIONS)})
    print("   load factors: %s"
          % {k: NON_EXHAUST_MASS_FACTOR[k]
             for k in sorted(NON_EXHAUST_MASS_FACTOR)})
    print("   speed corrections at 20 / 60 / 110 km/h: brake %.3f/%.3f/%.3f, "
          "tyre %.3f/%.3f/%.3f"
          % (brake_wear_speed_factor(20.0), brake_wear_speed_factor(60.0),
             brake_wear_speed_factor(110.0), tyre_wear_speed_factor(20.0),
             tyre_wear_speed_factor(60.0), tyre_wear_speed_factor(110.0)))
    for cat in ("passenger_car_petrol", "hgv_diesel"):
        ne = non_exhaust_emission_factor(cat, 30.0, "PM10")
        print("      %-22s PM10 brake %.5f tyre %.5f road %.5f = %.5f g/km "
              "(shares %s)" % (cat, ne.brake, ne.tyre, ne.road, ne.total,
                               {k: round(v, 3)
                                for k, v in sorted(ne.shares().items())}))
    mass = non_exhaust_emissions("passenger_car_petrol", 30.0, 1.2, 1500.0,
                                 "PM2.5")
    print("   1500 cars x 1.2 km at 30 km/h -> %.1f g PM2.5 non-exhaust"
          % mass.total)
    exhaust = fleet_hot_emissions(fleet, "PM2.5", 30.0, 1.2, 1500.0)
    print("   exhaust PM2.5 over the same movement = %.1f g -> non-exhaust is "
          "%.1f x the exhaust" % (exhaust, mass.total / exhaust))

    # -- 9. acoustics fundamentals -----------------------------------------
    print("\n-- 9. decibel algebra, spreading, ground and barriers --")
    print("   two 70 dB sources sum to %.6f dB (+%.6f); ten sum to %.6f "
          "(+%.6f)" % (db_sum([70.0, 70.0]), db_sum([70.0, 70.0]) - 70.0,
                       db_sum([70.0] * 10), db_sum([70.0] * 10) - 70.0))
    print("   db_subtract(73.0103, 70) = %.6f dB | energy mean of "
          "[60, 70, 80] = %.4f dB" % (db_subtract(db_sum([70.0, 70.0]), 70.0),
                                      db_energy_mean([60.0, 70.0, 80.0])))
    print("   duration-weighted Leq of [70, 60] over [1, 3] h = %.4f dB"
          % db_weighted_energy_mean([70.0, 60.0], [1.0, 3.0]))
    print("   POINT source doubling: %.9f dB | LINE source doubling: %.9f dB"
          % (point_source_attenuation(20.0, 10.0),
             line_source_attenuation(20.0, 10.0)))
    print("   geometric_spreading dispatch agrees: point %s line %s"
          % (geometric_spreading(20.0, 10.0, "point")
             == point_source_attenuation(20.0, 10.0),
             geometric_spreading(20.0, 10.0, "line")
             == line_source_attenuation(20.0, 10.0)))
    print("   atmospheric absorption over 500 m at 500 Hz, 20 C, 70 %% RH = "
          "%.3f dB" % atmospheric_absorption(500.0, 500.0, 20.0, 70.0))
    print("   ISO 9613-2 ground effect at 100 m over soft ground = %.3f dB"
          % ground_attenuation(100.0, 2.0, 1.0))
    delta = path_difference((0.0, 0.5), (12.0, 4.0), (30.0, 1.5))
    print("   barrier 4 m high at 12 m: path difference %.4f m, Fresnel N = "
          "%.4f, Maekawa loss %.3f dB"
          % (delta, fresnel_number(delta), maekawa_barrier_attenuation(delta)))

    # -- 10. road-traffic noise ---------------------------------------------
    print("\n-- 10. road-traffic noise: TNM-lite and CNOSSOS-lite --")
    print("   acoustic categories: %s" % (NOISE_CATEGORIES,))
    print("   REMELs (a, b): %s"
          % {k: (FHWA_REMELS[k].a, FHWA_REMELS[k].b)
             for k in sorted(FHWA_REMELS)})
    v55 = mph_to_kmh(55.0)
    print("   PUBLISHED CHECK: an auto at 55 mph is %.2f dB(A) at 15 m "
          "(FHWA quotes 63.9)" % fhwa_remel("auto", v55))
    print("   heavy truck at 55 mph = %.2f dB(A); motorcycle = %.2f dB(A)"
          % (fhwa_remel("heavy_truck", v55), fhwa_remel("motorcycle", v55)))
    print("   CNOSSOS sound power: auto %.2f, heavy truck %.2f dB re 1 pW"
          % (cnossos_sound_power("auto", v55),
             cnossos_sound_power("heavy_truck", v55)))
    print("   CNOSSOS coefficients for a light vehicle: %s"
          % (CNOSSOS_COEFFICIENTS["auto"],))
    print("   line power at 1000 veh/h: %.2f dB(A) re 1 pW/m"
          % cnossos_line_sound_power("auto", 1000.0, v55))
    print("   TNM hourly Leq, 1000 auto/h at 15 m = %.2f dB(A); at 60 m = "
          "%.2f dB(A)" % (fhwa_leq("auto", 1000.0, v55),
                          fhwa_leq("auto", 1000.0, v55, 60.0)))
    streams = [TrafficStream("auto", 1450.0, 48.0),
               TrafficStream("medium_truck", 90.0, 45.0),
               TrafficStream("heavy_truck", 45.0, 42.0),
               TrafficStream("bus", 30.0, 35.0),
               TrafficStream("motorcycle", 25.0, 50.0)]
    print("   streams on the link: %s"
          % [s.summary() for s in streams[:2]])
    free = road_noise_level(streams, 25.0, "fhwa", ground_factor=0.5)
    print("   mixed stream at 25 m (FHWA): %.2f dB(A) [source %.2f, "
          "spreading %.2f, ground %.2f]"
          % (free.level_db, free.source_db, free.spreading_db, free.ground_db))
    cn = road_noise_level(streams, 25.0, "cnossos", ground_factor=0.5)
    print("   the same stream through CNOSSOS-lite: %.2f dB(A) (%.2f dB "
          "apart -- the two models are tied at the 55 mph reference and part "
          "company at urban speeds, where CNOSSOS keeps a propulsion floor "
          "the TNM log-speed REMEL does not)"
          % (cn.level_db, cn.level_db - free.level_db))
    shielded = road_noise_level(streams, 25.0, "fhwa", ground_factor=0.5,
                                barrier_path_difference_m=delta,
                                include_absorption=True)
    print("   with the barrier and absorption: %.2f dB(A) (barrier %.2f, "
          "absorption %.2f, total attenuation %.2f dB)"
          % (shielded.level_db, shielded.barrier_db, shielded.absorption_db,
             shielded.total_attenuation_db()))
    hourly = [free.level_db - 8.0] * 24
    for h in range(7, 19):
        hourly[h] = free.level_db
    for h in (19, 20, 21, 22):
        hourly[h] = free.level_db - 3.0
    day = l_aeq(hourly[7:19])
    eve = l_aeq(hourly[19:23])
    night = l_night(hourly)
    den = l_den(day, eve, night)
    print("   L_day %.2f, L_evening %.2f, L_night %.2f -> L_den %.2f dB(A)"
          % (day, eve, night, den))
    print("   penalties are exact: equal periods with zero penalty give "
          "L_den = L_day (%.9f vs %.9f)"
          % (l_den(day, day, day, 0.0, 0.0), day))
    print("   facade level (+3 dB reflection) = %.2f dB(A)"
          % facade_level(den))
    print("   dose-response at L_den %.1f: %.2f %% highly annoyed, %.2f %% "
          "annoyed; at L_night %.1f, %.2f %% sleep-disturbed"
          % (den, percent_highly_annoyed(den), percent_annoyed(den), night,
             percent_sleep_disturbed(night)))

    # -- 11. Gaussian-plume dispersion --------------------------------------
    print("\n-- 11. Gaussian-plume dispersion --")
    print("   stability classes: %s"
          % ["%s (%s)" % (c, stability_description(c).split(" --")[0])
             for c in STABILITY_CLASSES])
    print("   terrains: %s" % (TERRAIN_TYPES,))
    print("   1.5 m/s + strong sun -> %s; 7 m/s + slight sun -> %s; "
          "2.5 m/s clear night -> %s"
          % (pasquill_stability(1.5, "strong"),
             pasquill_stability(7.0, "slight"),
             pasquill_stability(2.5, daytime=False, cloud_cover_oktas=1.0)))
    print("   %-9s %10s %10s %10s %10s" % ("class", "sy urban", "sz urban",
                                           "sy rural", "sz rural"))
    for cls in STABILITY_CLASSES:
        print("   %-9s %10.3f %10.3f %10.3f %10.3f"
              % (cls, briggs_sigma_y(500.0, cls), briggs_sigma_z(500.0, cls),
                 briggs_sigma_y(500.0, cls, "rural"),
                 briggs_sigma_z(500.0, cls, "rural")))
    q, u = 12.0, 3.5
    sy, sz = briggs_sigma_y(800.0, "D"), briggs_sigma_z(800.0, "D")
    centre = gaussian_plume_concentration(q, u, sy, sz)
    closed_form = q / (math.pi * sy * sz * u)
    print("   ORACLE centreline: %.12e g/m3 vs Q/(pi sy sz u) %.12e, "
          "err %.3e" % (centre, closed_form, abs(centre - closed_form)))
    flux = plume_mass_flux(q, u, sy, sz)
    print("   ORACLE mass conservation: flux %.12f g/s vs Q %.1f g/s, "
          "relative error %.3e" % (flux, q, abs(flux - q) / q))
    flux_h = plume_mass_flux(q, u, sy, sz, 45.0)
    print("   the same with a 45 m release height: %.12f g/s (rel err %.3e)"
          % (flux_h, abs(flux_h - q) / q))
    print("   off-centre (y=60 m, z=10 m): %.6e g/m3 = %.4f ug/m3"
          % (gaussian_plume_concentration(q, u, sy, sz, 60.0, 10.0),
             ug_per_m3(gaussian_plume_concentration(q, u, sy, sz, 60.0,
                                                    10.0))))
    print("   with a 500 m mixing lid at z=250 m: %.6e g/m3"
          % gaussian_plume_concentration(q, u, sy, sz, 0.0, 250.0, 0.0, True,
                                         500.0))
    print("   SCALE: Q from 1e-6 to 1e6 g/s scales C exactly linearly (%s)"
          % (abs(gaussian_plume_concentration(1e6, u, sy, sz)
                 / gaussian_plume_concentration(1e-6, u, sy, sz) - 1e12)
             < 1e-3))
    rise = plume_rise(1.8, 15.0, 430.0, 288.0, 4.0, "D")
    print("   Briggs plume rise (1.8 m stack, 15 m/s, 430 K into 288 K at "
          "4 m/s): %.2f m; in stable F air %.2f m"
          % (rise, plume_rise(1.8, 15.0, 430.0, 288.0, 4.0, "F")))
    q_line = 0.0025
    cwi = crosswind_integrated_concentration(q_line, u, sz)
    seg = line_source_concentration(q_line, u, sy, sz, -5.0e5, 5.0e5)
    print("   infinite line %.9e g/m3 vs erf-integrated segment %.9e "
          "(err %.2e)" % (cwi, seg, abs(cwi - seg)))
    print("   a 200 m segment gives %.9e g/m3 (%.1f %% of the infinite line)"
          % (line_source_concentration(q_line, u, sy, sz, -100.0, 100.0),
             100.0 * line_source_concentration(q_line, u, sy, sz, -100.0,
                                               100.0) / cwi))
    ef_nox = fleet_emission_factor(fleet, "NOx", 30.0)
    link_flow = 1600.0
    q_road = link_flow * ef_nox / (3600.0 * 1000.0)
    print("   link at 1600 veh/h, fleet NOx %.4f g/km -> %.4e g/(m s)"
          % (ef_nox, q_road))
    profile = []
    for d in (5.0, 10.0, 20.0, 50.0, 100.0, 200.0):
        profile.append((d, road_link_concentration(q_road, 3.0, d,
                                                   stability="D",
                                                   initial_sigma_z_m=1.5,
                                                   background_ug_m3=18.0)))
    print("   kerbside NOx with an 18 ug/m3 background: %s"
          % ["%.0f m: %.1f" % (d, c) for d, c in profile])

    # -- 12. box and street-canyon models -----------------------------------
    print("\n-- 12. box and street-canyon models --")
    box = box_model_concentration(4.0, 3.0, 300.0, 2000.0)
    print("   ORACLE box steady state: %.12e g/m3 vs Q/(u H W) %.12e"
          % (box, 4.0 / (3.0 * 300.0 * 2000.0)))
    build = box_model_buildup(4.0, 3.0, 300.0, 2000.0, 6000.0, 4000.0, 250.0)
    print("   build-up over %.0f s in %d samples: %.4f -> %.4f ug/m3 "
          "(steady %.4f, residence %.0f s)"
          % (build[-1][0], len(build), ug_per_m3(build[0][1]),
             ug_per_m3(build[-1][1]), ug_per_m3(box), 6000.0 / 3.0))
    canyon = StreetCanyon(21.0, 18.0, "Marylebone Road")
    print("   canyon %r: H/W = %.3f, regime %s, sky view factor %.4f"
          % (canyon.name, canyon.aspect_ratio(), canyon.regime(),
             sky_view_factor(canyon.height_m, canyon.width_m)))
    cres = canyon_concentration(canyon, q_road, 4.0, 90.0,
                                background_ug_m3=18.0,
                                traffic_turbulence_ms=0.25)
    print("   leeward %.2f ug/m3, windward %.2f, recirculation %.2f, direct "
          "%.2f, w_t %.3f m/s, L_rec %.1f m, trapped %.3f"
          % (cres.leeward_ug_m3, cres.windward_ug_m3,
             cres.recirculation_ug_m3, cres.direct_ug_m3,
             cres.exchange_velocity_ms, cres.recirculation_length_m,
             cres.trapped_fraction))
    print("   leeward increment over background = %.2f ug/m3, worst facade "
          "%.2f ug/m3" % (cres.leeward_increment_ug_m3, cres.max_ug_m3))
    along = canyon_concentration(canyon, q_road, 4.0, 0.0,
                                 background_ug_m3=18.0)
    print("   wind straight along the street: no vortex (L_rec %.1f m), "
          "leeward falls to %.2f ug/m3"
          % (along.recirculation_length_m, along.leeward_ug_m3))
    windy = canyon_concentration(canyon, q_road, 40.0, 90.0,
                                 background_ug_m3=18.0)
    print("   at 40 m/s the canyon flushes to %.2f ug/m3 (background 18.00)"
          % windy.leeward_ug_m3)

    # -- 13. background and population-weighted exposure --------------------
    print("\n-- 13. background and exposure --")
    print("   add_background(22.4, 18.0) = %.2f ug/m3"
          % add_background(22.4, 18.0))
    zones = [
        ExposureZone("canyon frontage", 4200.0, cres.leeward_ug_m3, 0.18),
        ExposureZone("side streets", 18500.0, 26.5, 1.4),
        ExposureZone("park edge", 6100.0, 21.0, 0.9),
        ExposureZone("suburb", 52000.0, 17.5, 11.0),
    ]
    print("   population-weighted mean = %.3f ug/m3 over %d zones"
          % (population_weighted_exposure(zones), len(zones)))
    for z in zones:
        print("      %-16s %7.0f people, %6.2f ug/m3, %6.0f /km2"
              % (z.name, z.population, z.concentration_ug_m3,
                 z.density_per_km2()))
    exc = exceedance_population(zones, 25.0)
    print("   above the 25 ug/m3 limit: %.0f people (%.1f %% of %.0f), zones "
          "%s, worst %s" % (exc["population"], 100.0 * exc["share"],
                            exc["totalPopulation"], exc["zones"],
                            exc["worstZone"]))
    series = [22.0 + 8.0 * math.sin(i / 3.7) for i in range(240)]
    gappy = [None if i % 7 == 0 else series[i] for i in range(240)]
    print("   annual mean from a complete series %.4f, with 14 %% gaps "
          "%.4f ug/m3" % (annual_mean_from_hourly(series),
                          annual_mean_from_hourly(gappy)))

    # -- 14. health impact ---------------------------------------------------
    print("\n-- 14. health impact --")
    print("   concentration-response functions:")
    for pol in HEALTH_POLLUTANTS:
        crf = CONCENTRATION_RESPONSE[pol]
        print("      %-6s RR %.3f per 10 ug/m3 above %.0f ug/m3 -- %s"
              % (pol, crf["rrPer10"], crf["counterfactualUgM3"],
                 crf["endpoint"]))
    pm = population_weighted_exposure(zones) * 0.62
    rr = relative_risk("PM2.5", pm)
    af = attributable_fraction(rr)
    print("   PM2.5 at %.2f ug/m3 -> RR %.5f -> AF %.5f (identity "
          "AF = 1 - 1/RR err %.2e)" % (pm, rr, af, abs(af - (1.0 - 1.0 / rr))))
    print("   AF is exactly zero at RR = 1: %.1f" % attributable_fraction(1.0))
    deaths = premature_deaths(80800.0, 950.0, af)
    print("   80 800 people at a 950/100k baseline -> %.2f attributable "
          "deaths, %.1f years of life lost"
          % (deaths, years_of_life_lost(deaths)))
    hia = health_impact_assessment("PM2.5", pm, 80800.0, 950.0,
                                   value_per_life_year=60000.0)
    print("   full assessment: %.2f deaths (%.2f per 100k), %.0f YLL, "
          "monetised %.2f M" % (hia.deaths, hia.deaths_per_100k,
                                hia.years_of_life_lost,
                                hia.monetised / 1.0e6))
    no2 = health_impact_assessment("NOx", 34.0, 80800.0, 950.0)
    print("   NO2 at 34 ug/m3 (20 cut-off): RR %.5f, %.2f deaths"
          % (no2.relative_risk, no2.deaths))
    print("   NO2 at the cut-off itself gives RR exactly 1: %.1f"
          % relative_risk("NOx", 20.0))

    # -- 15. carbon accounting -----------------------------------------------
    print("\n-- 15. carbon accounting and the social cost of carbon --")
    co2e = co2_equivalent({"CO2": 1.0e6, "CH4": 420.0, "N2O": 35.0})
    print("   1 t CO2 + 420 g CH4 + 35 g N2O = %.1f g CO2e (%.4f t)"
          % (co2e, grams_to_tonnes(co2e)))
    print("   discount factors at 3.5 %%: t=0 %.6f, t=10 %.6f, t=30 %.6f; "
          "zero rate at t=30 %.1f"
          % (discount_factor(0.035, 0.0), discount_factor(0.035, 10.0),
             discount_factor(0.035, 30.0), discount_factor(0.0, 30.0)))
    fleet_tonnes = grams_to_tonnes(
        fleet_hot_emissions(fleet, "CO2", 30.0, 1.2, 1600.0) * 16.0 * 250.0)
    print("   link CO2: %.1f t/year at 1600 veh/h x 16 h x 250 days"
          % fleet_tonnes)
    print("   social cost now %.0f, in 20 years %.0f (price growing 2 %%/yr, "
          "discounted at 3.5 %%)"
          % (social_cost_of_carbon(fleet_tonnes, 120.0),
             social_cost_of_carbon(fleet_tonnes, 120.0, 20.0, 0.035, 0.02)))
    account = carbon_account([fleet_tonnes * (1.0 - 0.03 * i)
                              for i in range(20)], 120.0, 0.035, 0.02,
                             2025.0, "link CO2 appraisal")
    print("   20-year account: %.0f t, NPV %.0f, undiscounted %.0f, effective "
          "price %.2f / t" % (account.total_tonnes,
                              account.net_present_value,
                              account.undiscounted_total,
                              account.effective_price()))

    # -- 16. urban heat and thermal comfort ----------------------------------
    print("\n-- 16. urban heat and thermal comfort --")
    print("   sky view factor: open field %.4f, H/W=0.5 %.4f, H/W=1 %.4f, "
          "H/W=3 %.4f" % (sky_view_factor(0.0, 20.0),
                          sky_view_factor(10.0, 20.0),
                          sky_view_factor(20.0, 20.0),
                          sky_view_factor(60.0, 20.0)))
    print("   asymmetric canyon (24 m and 8 m facades, 18 m wide): %.4f; the "
          "symmetric 16 m case: %.4f"
          % (sky_view_factor_asymmetric(24.0, 8.0, 18.0),
             sky_view_factor(16.0, 18.0)))
    print("   saturation vapour pressure at 20 / 30 C = %.3f / %.3f hPa; at "
          "55 %% RH and 30 C, e = %.3f hPa"
          % (saturation_vapour_pressure(20.0),
             saturation_vapour_pressure(30.0), vapour_pressure(30.0, 55.0)))
    print("   log profile: 5 m/s at 10 m becomes %.3f m/s at 1.1 m and %.3f "
          "m/s at 25 m (exactly %.3f at the reference)"
          % (wind_speed_at_height(5.0, 1.1), wind_speed_at_height(5.0, 25.0),
             wind_speed_at_height(5.0, 10.0)))
    svf = sky_view_factor(canyon.height_m, canyon.width_m)
    mrt_shade = mean_radiant_temperature(31.0, 120.0, svf, 45.0)
    mrt_sun = mean_radiant_temperature(31.0, 780.0, svf, 45.0)
    print("   MRT in the canyon at 31 C: shaded (120 W/m2) %.2f C, sunlit "
          "(780 W/m2) %.2f C" % (mrt_shade, mrt_sun))
    print("   IDENTITY: zero radiation, unit sky emissivity, open sky -> MRT "
          "= air temperature exactly (%.6f vs 31.0)"
          % mean_radiant_temperature(31.0, 0.0, 1.0, 45.0,
                                     surface_excess_c=0.0,
                                     sky_emissivity=1.0))
    v_ped = wind_speed_at_height(5.0, 1.1)
    for label, mrt in (("shaded", mrt_shade), ("sunlit", mrt_sun)):
        idx = utci_lite(31.0, mrt, v_ped, 45.0)
        print("      %-7s UTCI-lite %.2f C (%s) | PET-lite %.2f C"
              % (label, idx, thermal_stress_category(idx),
                 pet_lite(31.0, mrt, v_ped, 45.0)))
    print("   IDENTITY: Tmrt = Ta at 0.5 m/s and 50 %% RH -> UTCI %.6f, "
          "PET %.6f (both exactly 31)"
          % (utci_lite(31.0, 31.0, 0.5, 50.0), pet_lite(31.0, 31.0, 0.1,
                                                        50.0)))
    print("   stress bands: %s"
          % [("%+.0f C: %s" % (t, thermal_stress_category(t)))
             for t in (-45.0, -5.0, 15.0, 29.0, 35.0, 48.0)])

    # -- 17. payloads --------------------------------------------------------
    print("\n-- 17. frontend payloads --")
    noise_profile = [(d, road_noise_level(streams, d, "fhwa",
                                          ground_factor=0.5).level_db)
                     for d in (10.0, 15.0, 25.0, 40.0, 60.0, 100.0, 160.0)]
    payload = build_emissions_environment_payload(
        fleet, pollutant="NOx", modal=modal, fuel=fuel, cold_start=cs,
        calibration=cal, rank=rank, non_exhaust=mass, noise=free,
        noise_profile=noise_profile, dispersion_profile=profile,
        canyon=cres, zones=zones, health=hia, carbon=account,
        curve_points=30)
    ui = build_a2ui_payload(fleet, pollutant="NOx", modal=modal, fuel=fuel,
                            noise=free, noise_profile=noise_profile,
                            dispersion_profile=profile, canyon=cres,
                            health=hia, carbon=account, l_den_db=den,
                            utci_c=utci_lite(31.0, mrt_sun, v_ped, 45.0))
    print("   payload keys: %s" % sorted(payload))
    print("   emission curve %d points | exposure zones %d | carbon years %d"
          % (len(payload["emissionCurve"]), len(payload["exposure"]["zones"]),
             len(payload["carbon"]["years"])))
    print("   a2ui charts: %s" % sorted(ui["charts"]))
    print("   a2ui cards: %s"
          % ["%s=%s %s" % (c["label"], c["value"], c["unit"])
             for c in ui["statCards"][:4]])
    print("   both payloads are json-serialisable: %d and %d characters"
          % (len(json.dumps(payload)), len(json.dumps(ui))))

    # -- 18. input validation ------------------------------------------------
    print("\n-- 18. input validation --")
    for bad in (lambda: emission_factor("O3", "passenger_car_petrol",
                                        "Euro 5", 30.0),
                lambda: emission_factor("NOx", "hovercraft", "Euro 5", 30.0),
                lambda: gaussian_plume_concentration(1.0, 0.0, 10.0, 5.0),
                lambda: attributable_fraction(0.8),
                lambda: db_subtract(60.0, 70.0),
                lambda: FleetComposition({}),
                lambda: line_source_concentration(1.0, 2.0, 5.0, 3.0, 10.0,
                                                  -10.0),
                lambda: annual_mean_from_hourly([1.0, None, None, None])):
        try:
            bad()
            print("   ERROR: no exception raised")
        except EmissionsError as exc:
            print("   EmissionsError: %s" % exc)

    print("\n" + "=" * 74)
    print("  demo OK")
    print("=" * 74)


if __name__ == "__main__":
    _demo()
