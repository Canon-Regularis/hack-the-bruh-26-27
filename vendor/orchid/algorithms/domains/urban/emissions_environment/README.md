# Emissions & Environment

A self-contained, **zero-dependency** toolkit for the **environmental consequence of a
traffic pattern** — the layer that turns speeds, stops and idle time into grams of NOx,
litres of fuel, decibels at a facade, micrograms per cubic metre at a kerb, and the
population, health and carbon cost of all of it. Feed it a link with a flow, a mean
speed and a stop count; get back a COPERT-style emission factor, a MOVES operating-mode
breakdown, an Akcelik–Biggs fuel split, an `L_den`, a Gaussian-plume
concentration, a street-canyon facade level, a population-weighted exposure, an
attributable fraction and a discounted social cost of carbon.

It sits deliberately **downstream of `traffic_flow`**, which manufactures the speeds,
accelerations and stop counts that arrive here as input, and deliberately **apart from
`green_infrastructure`**, which is about habitat and canopy rather than pollution.
Everything is pure standard library: the error function, the plume quadrature, the
decibel algebra, the barrier diffraction, the rational emission curves and the
non-negative least squares are all built by hand.

```python
from emissions_environment_template import (
    FleetComposition, fleet_emission_factor, stop_go_cycle, modal_emissions,
    road_link_concentration, TrafficStream, road_noise_level, l_den,
    percent_highly_annoyed)

fleet = FleetComposition({("passenger_car_petrol", "Euro 6"): 0.6,
                          ("passenger_car_diesel", "Euro 5"): 0.3,
                          ("bus_diesel", "Euro 6"): 0.1}, "city centre")

ef    = fleet_emission_factor(fleet, "NOx", 30.0)     # g/km at 30 km/h
q     = 1600.0 * ef / (3600.0 * 1000.0)               # g per metre per second
kerb  = road_link_concentration(q, 3.0, 10.0,         # ug/m3 at 10 m
                                initial_sigma_z_m=1.5, background_ug_m3=18.0)
modal = modal_emissions(stop_go_cycle(40.0, 6))       # MOVES bins -> grams

noise = road_noise_level([TrafficStream("auto", 1450.0, 48.0),
                          TrafficStream("heavy_truck", 45.0, 42.0)], 25.0)
annoy = percent_highly_annoyed(l_den(noise.level_db, noise.level_db - 3.0,
                                     noise.level_db - 8.0))
```

Traffic units are **(km, h)**; modal units are **SI (m, s)**. Concentrations are
**g/m³ internally** and **µg/m³ for reporting**. Sound levels are A-weighted decibels
re 20 µPa. Everything stochastic takes an explicit `seed`, every tie breaks on a stored
value, and the demo is byte-identical across `PYTHONHASHSEED` values.

---

## Pick

| You have | You want | Call |
| --- | --- | --- |
| A link flow and a mean speed | Grams of a pollutant | `fleet_emission_factor` → `fleet_hot_emissions` |
| A second-by-second speed trace | Grams, properly modal | `modal_emissions` (VSP → MOVES bins) |
| A trace and a vehicle mass | Litres of fuel, split four ways | `fuel_from_trace` |
| Distance, idle time, stop count | Litres without the trace | `elemental_fuel_model` |
| Modal totals on many edges | A cheap `a·f·t + b·idle + c·stop` surrogate | `calibrate_edge_proxy` → `proxy_rank_consistency` |
| Brake, tyre and road wear | Non-exhaust PM10 / PM2.5 | `non_exhaust_emissions` |
| Vehicle streams and a distance | dB(A) at a receptor | `road_noise_level` → `facade_level` |
| Day / evening / night levels | `L_den`, `L_night`, % highly annoyed | `l_den`, `l_night`, `percent_highly_annoyed` |
| A source strength and a wind | A concentration field | `briggs_sigma_*` → `gaussian_plume_concentration` |
| A road and a receptor offset | Kerbside µg/m³ | `road_link_concentration` |
| Two facades and a roof wind | Leeward / windward µg/m³ | `canyon_concentration` |
| A basin, a wind and a lid | The simplest possible answer | `box_model_concentration` |
| Zones with populations | Population-weighted exposure | `population_weighted_exposure` |
| An exposure and a population | Deaths and years of life lost | `health_impact_assessment` |
| A stream of annual tonnages | A discounted social cost | `carbon_account` |
| Canyon geometry and radiation | Sky view factor, MRT, UTCI | `sky_view_factor`, `mean_radiant_temperature`, `utci_lite` |

---

## Install

Nothing to install. Pure standard library (`math`, `random`, `dataclasses`, `typing`,
plus `json` inside the demo only), Python 3.8+.

```bash
python emissions_environment_template.py   # runs the demo, prints "demo OK"
pytest _stress_test.py -q                  # 210 oracle-based tests
```

`pytest>=7.0` is the only uncommented line in `requirements.txt`. NumPy and SciPy are
**optional** and used only by the suite's differential tier, which skips cleanly when
they are absent.

---

## What's inside

### 1 — Units, constants and registries

`POLLUTANTS` (NOx, PM10, PM2.5, CO, HC, CO2), `VEHICLE_CATEGORIES`, `EURO_CLASSES`,
`GWP100`, `FUEL_PROPERTIES`, and the conversions that keep the two unit worlds apart.

### 2 — Speed-dependent emission factors

The COPERT-5 rational curve `EF(v) = (a v² + b v + c + d/v) / (e v² + f v + g)`, carried
for six vehicle categories × six pollutants with `EURO_REDUCTION_FACTORS` on top. Every
built-in curve is the hyperbolic subform `A/v + B + C v`, so it is **convex with an
interior minimum** — `curve_minimum` locates it by grid scan plus golden section, and an
import-time validator refuses any curve whose minimum sits at an endpoint. That U shape
is the whole point: congestion costs air quality at both ends.

### 3 — Fleet mixing and cold start

`FleetComposition` over `(category, Euro class)` shares, summed in sorted key order.
`cold_start_beta` is the COPERT mileage fraction driven with a cold engine,
`cold_start_ratio` the temperature-dependent cold/hot ratio, and `cold_start_excess`
splits hot mass from excess.

### 4 — Vehicle Specific Power and MOVES operating modes

`vehicle_specific_power` in kW/tonne — **exactly zero at zero speed**, and at constant
speed on a flat grade exactly the road-load polynomial. `operating_mode_bin` implements
the published MOVES partition (braking first, then idle, then three speed bands × VSP
bins; the gaps at 26, 31, 32, 34 and 36 are genuine). `moves_opmode_rate` gives a
VSP-linear rate per bin.

### 5 — Traces, drive cycles and modal emissions

`SpeedTrace` with accelerations, trapezoidal distance, idle time, hysteresis-banded
`stop_count` and positive kinetic energy; `constant_speed_cycle`, `stop_go_cycle` and
the seeded Ornstein–Uhlenbeck `synthetic_drive_cycle`; `modal_emissions` integrating bin
rates over the `opmode_distribution`.

### 6 — Fuel: the Akcelik–Biggs elemental model

`tractive_power`, `instantaneous_fuel_rate` and the four-mode classification (idle /
cruise / accelerate / decelerate). `fuel_from_trace` integrates over a trace;
`elemental_fuel_model` is Akcelik's `F = f₁·d + f₂·idle + f₃·stops` for when you have no
trace at all.

### 7 — The linear edge proxy and its calibration

The white paper's network objective uses the deliberately cheap
`E_e = a·f_e·t_e + b·idle_e + c·stop_e`, because an assignment or signal-optimisation
loop cannot afford a modal model in its inner iteration. This module implements **both
ends of that bargain**: the proxy itself, an exact least-squares (optionally
non-negative, over all eight supports) fit of `(a, b, c)` against modal output, and the
Kendall τ-b certificate that the proxy still **orders links the way the modal model
does**. A proxy you have not calibrated is a guess; a proxy you have is a surrogate.

### 8 — Non-exhaust particulate matter

Brake, tyre and road-surface wear with the EMEP/EEA speed corrections, load factors and
size fractions. On a modern Euro 6 fleet the non-exhaust stream is several times the
exhaust PM — the demo prints the ratio.

### 9 — Acoustics fundamentals

Exact decibel algebra via `math.fsum` over linear energies, geometric spreading for
point (`20 log₁₀ d`) and line (`10 log₁₀ d`) sources, ISO 9613-1 absorption, ISO 9613-2
ground effect, and Maekawa / Kurze–Anderson barrier diffraction via `path_difference` →
`fresnel_number` → `maekawa_barrier_attenuation`.

### 10 — Road-traffic noise

Two source models — FHWA/TNM-lite REMELs and CNOSSOS-EU-lite sound powers — with an
import-time validator enforcing that they agree within 0.5 dB at the 55 mph reference
for every class. `road_noise_level` assembles spreading, ground, barrier and absorption;
`l_aeq` / `l_den` / `l_night` / `facade_level` give the indicators, and
`percent_highly_annoyed` / `percent_annoyed` / `percent_sleep_disturbed` the
dose-response.

### 11 — Gaussian-plume dispersion

`pasquill_stability` (the published day/night table), Briggs (1973) sigmas for rural and
urban terrain, Briggs `plume_rise`, and the plume itself with ground reflection and
optional mixing-lid images. Companions: `crosswind_integrated_concentration`, the
closed-form erf-integrated `line_source_concentration`, the practical
`road_link_concentration`, and `plume_mass_flux`, which integrates `u·C` over a downwind
plane and must return `Q`.

### 12 — Box and street-canyon models

`box_model_concentration` (exactly `Q / (u H W)`) and `box_model_buildup` (the closed
exponential with time constant `L / u`). `canyon_concentration` is an OSPM-lite two-box
model: a recirculating vortex extending `2H` from the upwind facade, ventilated only
through its roof at `w_t = √(σ_w² + σ_traffic²)`, plus a through-flow box for the rest
of the carriageway.

### 13 — Background and exposure

`add_background`, `ExposureZone`, `population_weighted_exposure`,
`exceedance_population`, and `annual_mean_from_hourly` — which refuses to report an
annual mean below 75 % data capture rather than quietly averaging a fortnight into a
year.

### 14 — Health impact

`relative_risk` (log-linear CRF), `attributable_fraction` (`(RR − 1)/RR`),
`premature_deaths`, `years_of_life_lost`, and the whole chain in
`health_impact_assessment`.

### 15 — Carbon accounting

`co2_equivalent` over GWP100, `discount_factor`, `social_cost_of_carbon` with price
escalation, and `carbon_account` for a tonnage stream with an NPV and an effective price
per tonne.

### 16 — Urban heat and thermal comfort

The exact two-dimensional canyon `sky_view_factor` and its asymmetric generalisation,
Magnus vapour pressures, the logarithmic `wind_speed_at_height`, the radiation budget in
`mean_radiant_temperature`, and the `utci_lite` / `pet_lite` surrogates with
`thermal_stress_category`.

### 17 — Payloads

`build_emissions_environment_payload` and `build_a2ui_payload` return JSON-able dicts
with curves, series, stat cards and colour-coded bands.

---

## Recipes

**Is this link's proxy safe to optimise against?**

```python
cal  = calibrate_edge_proxy(observations, "CO2")
rank = proxy_rank_consistency(cal.coefficients, observations)
assert cal.r_squared > 0.99 and rank.monotone      # never reverses the modal order
```

**How far does the kerbside excess reach?**

```python
profile = [(d, road_link_concentration(q, 3.0, d, stability="D",
                                       initial_sigma_z_m=1.5,
                                       background_ug_m3=18.0))
           for d in (5.0, 10.0, 25.0, 50.0, 100.0, 200.0)]
```

**Does a barrier buy enough?**

```python
delta  = path_difference((0.0, 0.5), (12.0, 4.0), (30.0, 1.5))
shield = road_noise_level(streams, 30.0, barrier_path_difference_m=delta)
gain   = road_noise_level(streams, 30.0).level_db - shield.level_db
```

**What does a scheme cost in carbon over twenty years?**

```python
acc = carbon_account([tonnes * (1 - 0.03 * i) for i in range(20)],
                     price_per_tonne=120.0, discount_rate=0.035,
                     price_growth=0.02)
print(acc.net_present_value, acc.effective_price())
```

---

## Design notes

**The plume is kept honest by keeping the sigmas out of it.**
`gaussian_plume_concentration` takes `sigma_y` and `sigma_z` directly rather than
a downwind distance, so the ground-level centreline concentration is *visibly and
testably* `Q / (π σ_y σ_z u)` with no table lookup in the way; `briggs_sigma_y` /
`briggs_sigma_z` supply the sigmas as a separate, separately testable step.

**Mass conservation is an executable oracle, not a comment.** `plume_mass_flux` runs a
tensor-product composite Simpson rule over the downwind plane and returns `u·∫∫C`, which
must equal `Q` for any release height — it does so to ~2 × 10⁻¹⁵ relative. The integrand
is separable, and a tensor Simpson rule on a separable integrand factorises *exactly*
into the product of its one-dimensional rules, so the routine is the full 2-D rule at
`n_y + n_z` cost instead of `n_y · n_z`. The suite re-does the quadrature with a genuine
nested loop over the *public* plume function, and again through
`scipy.integrate.dblquad`, so the shortcut is never the thing being trusted.

**Scale discipline.** Urban quantities span many orders of magnitude — 1e-9 g/m³ and
1e9 currency units turn up in the same run — so **no guard compares a data-scaled
quantity against an absolute epsilon**. Every guard is either an exact-zero test
(`if l_rec == 0.0`, `if sst == 0.0`) or relative to its own data (the Cholesky pivot
tolerance is `_EPS × max|A|`; the ridge penalty is scaled to the mean diagonal). The
suite checks that concentration is exactly proportional to `Q` from 1e-6 to 1e6, and
that dimensionless outputs survive multiplying the natural scale by 1e±6.

**Determinism is structural.** Fleet shares, exposure zones, GWP gases, plume image
terms and edge observations are all iterated in **sorted** order, never dict or set
order. The original `(edge_id, id(obs))` tie-break was replaced with a total value key —
`id()` is a memory address, so two observations sharing an edge id could have been
summed in a different order in the next process. The demo's md5 is identical under
`PYTHONHASHSEED` 0, 1, 1010 and 424242.

**The "-lite" models are labelled and bounded.** `utci_lite` is a deliberate three-term
surrogate, not the official 200-term sixth-order UTCI polynomial; it reproduces the
reference condition (`T_mrt == T_a`, 0.5 m/s, 50 % RH -> `UTCI == T_a`) *exactly* and the published sign and rough magnitude of each offset, and
says so. `pet_lite` is built on the genuine operative temperature, a convex blend whose
weights sum to one, so the same identity holds at any wind speed. The OSPM-lite canyon
is a strict mass balance on the vortex rather than a tuned fit.

**Import-time validators.** Three invariants are checked when the module loads, because
a table that fails them is a typo rather than a physical model: every emission curve is
positive, finite and interior-minimised; FHWA and CNOSSOS agree within 0.5 dB at the
55 mph reference; and the annoyance curves nest (`%A ≥ %HA`) over the 42–85 dB window.

---

## Testing

`_stress_test.py` carries **210 tests** across four tiers and uses **no golden outputs**:
every number it asserts is a closed form, a published worked example, an exact identity
or an invariant.

- **Closed forms** — the plume centreline is exactly `Q / (π σ_y σ_z u)` (60 randomised
  operating points, to `1e-15`); the box model is exactly `Q / (u H W)`; a point source
  loses exactly `20 log₁₀ 2 = 6.0206` dB per distance doubling and an infinite line
  exactly `10 log₁₀ 2 = 3.0103` dB, both to `1e-9`; `N` equal incoherent sources add
  exactly `10 log₁₀ N`; VSP at constant speed on a flat grade is the road-load
  polynomial; `AF = (RR − 1)/RR`; the sky view factor is `1 / √(1 + (2H/W)²)`; MRT, UTCI
  and PET return the air temperature exactly at their degenerate reference conditions.
  Where no closed form exists the ground truth is a **published worked example hard-coded
  from the source**: the FHWA/TNM REMEL puts an automobile at 55 mph at 63.9 dB(A) at
  15 m (the module gives 63.91), the Briggs (1973) sigma formulae are evaluated by hand
  for both terrains, and the EU road dose-response anchors (6.4 % highly annoyed at
  `L_den` 55, 16.2 % at 65) are asserted directly.
- **Exact identities** — `plume_mass_flux` returns `Q` to a part in 1e-6, verified three
  ways; the finite line source tends to the crosswind-integrated form and its segments
  are additive; decibel addition is commutative and associative to `1e-12` under 200
  fuzzed cases; `L_den` applies exactly +5 dB evening and +10 dB night and collapses to
  `L_day` when the periods are equal and the penalties removed; the canyon vortex closes
  its own mass balance; emissions are exactly linear in flow, distance and vehicle count
  and the edge proxy exactly linear in each of its three terms; least squares recovers
  planted coefficients to `1e-8`.
- **Metamorphic / invariant** — every emission curve is U-shaped with an interior
  minimum; sigmas grow with distance and shrink with stability; the canyon flushes to
  the background as the wind rises; concentration is exactly proportional to `Q` over
  twelve orders of magnitude; the sky view factor, the trapped fraction, Kendall τ and
  the exceedance share are unchanged when their natural scale is multiplied by 1e±6;
  shuffling inputs never changes a result.
- **Differential** (behind `pytest.importorskip`) — `scipy.integrate.dblquad` and `quad`
  against the plume quadrature and the crosswind integral, `scipy.special.erf` against
  the closed-form line source, `scipy.stats.kendalltau` against the rank certificate,
  `numpy.linalg.lstsq` against the proxy calibration, `numpy.trapezoid` against the box
  build-up and the trace distance, and `numpy` against the decibel algebra and the curve
  minima. NumPy 2.x removed `np.trapz`, so the tier uses `np.trapezoid`.

```bash
python emissions_environment_template.py                  # demo OK
pytest _stress_test.py -q                                 # 210 passed
PYTHONHASHSEED=0    pytest _stress_test.py -q             # 210 passed
PYTHONHASHSEED=1    pytest _stress_test.py -q             # 210 passed
PYTHONHASHSEED=1010 pytest _stress_test.py -q             # 210 passed
```

**Defects the suite caught.** Two real ones. First, the annoyance dose-response table
carried a **negative linear term on the road and rail `%A` cubics**, which put "annoyed"
*below* "highly annoyed" at every level in the reporting range — an impossible ordering,
since annoyance is a superset of high annoyance. With the sign corrected the family
nests and reproduces the published road anchors (≈15 % at 55 dB, ≈31.5 % at 65, ≈42.5 %
at 70); an import-time validator now enforces the nesting. Second, the edge-observation
sort used `(edge_id, id(obs))` as its tie-break, and `id()` is a **memory address**, not
a stable value — two observations sharing an edge id could have been summed in a
different order in the next process and shifted the last bits of an `fsum`. It is now a
total value key.

Two oracles were *relaxed with reasons rather than weakened*: the tyre-wear speed
correction is discontinuous by 4e-4 at 40 km/h and 1.4e-3 at 90 km/h because the
published EMEP/EEA coefficients are rounded, so continuity is asserted to that rounding
(the brake correction closes exactly and is asserted with no slack at all); and the
`db_subtract`/`db_sum` round-trip is only well conditioned when the component removed is
within ~25 dB of the total — beyond that the sum genuinely no longer carries the
information, and inside it the round-trip is exact to `1e-9` dB. A third apparent
failure turned out to be physics: a ground-level release with **no** initial vertical
mixing really does peak a few metres from the kerb for a receptor at 1.5 m, which is
exactly why CALINE gives a road source a 1–2 m wake mixing depth. The suite now tests
both the near-field rise and the monotone profile with the depth switched on.

---

## References

- EMEP/EEA — *Air Pollutant Emission Inventory Guidebook*, Tier 3 road transport (the
  COPERT speed curves, cold start, and the brake/tyre/road wear factors).
- US EPA — *MOVES* technical documentation (Vehicle Specific Power and the
  operating-mode bins).
- Akcelik & Biggs (1983) — the four-mode elemental fuel model.
- FHWA — *Traffic Noise Model (TNM)* technical manual; European Commission —
  *CNOSSOS-EU* common noise assessment methods.
- Miedema & Oudshoorn (2001) and the EU *Position Paper on Dose–Response Relationships*.
- Briggs (1973) — *Diffusion estimation for small emissions*; Pasquill & Gifford;
  Turner — *Workbook of Atmospheric Dispersion Estimates*.
- Berkowicz (2000) — *OSPM: a parameterised street pollution model*.
- WHO — *HRAPIE* recommendations and *AirQ+* (the concentration–response functions).
- Oke — *Boundary Layer Climates*; Bröde et al. (2012) — the UTCI; Höppe (1999) — PET.
- ISO 9613-1 / 9613-2; Maekawa (1968) and Kurze & Anderson (1971) — barrier diffraction.

---

*Pure Python, standard library only (`math`, `random`, `dataclasses`, `typing`; `json`
inside the demo). `__version__ = "1.0.0"`, 153 public names in `__all__`.*
