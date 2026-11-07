"""Single import point for the ORCHID algorithms vendored in vendor/orchid/.

    from engine.orchid import street_network, traffic_assignment
    graph = street_network.RoadGraph()

The modules are verbatim copies of ORCHID at the commit recorded in
vendor/orchid/MANIFEST.json, made by scripts/sync_orchid.py.  Never edit anything under
vendor/ -- change the lists in that script and re-run it.  Run code from the repo root
(e.g. ``python -m src.graph.build_graph``) so that ``engine`` is importable.

Modules held back (unconverged ORCHID suites, known defects) and modules ORCHID has not
built yet raise ImportError with the reason when imported from here; see UNAVAILABLE.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

VENDOR_DIR = Path(__file__).resolve().parents[1] / "vendor" / "orchid"
_MANIFEST_PATH = VENDOR_DIR / "MANIFEST.json"
if not _MANIFEST_PATH.is_file():
    raise ImportError("%s not found -- run: python scripts/sync_orchid.py" % _MANIFEST_PATH)
MANIFEST = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
COMMIT = MANIFEST["source"]["commit"]

if str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))  # the templates' own `from algorithms.core ...` resolve here too

# Dotted imports only -- never import a template by its flat file name (that creates a second
# copy of every class and error type).  `python scripts/sync_orchid.py --check` keeps this list
# equal to the exposed modules in MANIFEST.json.

# urban -- hardened ORCHID suites
from algorithms.domains.urban.street_network import street_network_template as street_network  # noqa: E402
from algorithms.domains.urban.traffic_assignment import traffic_assignment_template as traffic_assignment  # noqa: E402
from algorithms.domains.urban.travel_demand import travel_demand_template as travel_demand  # noqa: E402
from algorithms.domains.urban.pedestrian_flow import pedestrian_flow_template as pedestrian_flow  # noqa: E402
from algorithms.domains.urban.emissions_environment import emissions_environment_template as emissions_environment  # noqa: E402
from algorithms.domains.urban.traffic_flow import traffic_flow_template as traffic_flow  # noqa: E402
from algorithms.domains.urban.accessibility_equity import accessibility_equity_template as accessibility_equity  # noqa: E402
from algorithms.domains.urban.signal_control import signal_control_template as signal_control  # noqa: E402
# urban -- vendored before its ORCHID hunts converged (see scripts/sync_orchid.py)
from algorithms.domains.urban.public_voice import public_voice_template as public_voice  # noqa: E402

# core -- called directly by the plan
from algorithms.core.ml.data_splitting import data_splitting_template as data_splitting  # noqa: E402
from algorithms.core.ml.ensembles import ensembles_template as ensembles  # noqa: E402
from algorithms.core.optimisation.multi_objective import multi_objective_template as multi_objective  # noqa: E402
from algorithms.core.optimisation.stochastic_optimisation import stochastic_optimisation_template as stochastic_optimisation  # noqa: E402

_stray = sorted(
    name for name, mod in list(sys.modules.items())
    if name.split(".")[0] == "algorithms" and getattr(mod, "__file__", None)
    and VENDOR_DIR not in Path(mod.__file__).resolve().parents
)
if _stray:  # a regular `algorithms` package, or an ORCHID checkout on sys.path, won the import
    raise ImportError("ORCHID modules loaded from outside %s: %s -- remove the other 'algorithms' package "
                      "or ORCHID checkout from sys.path / PYTHONPATH" % (VENDOR_DIR, _stray))

UNAVAILABLE = {e["name"]: e["reason"] for e in MANIFEST["held"] + MANIFEST["not_built"]}

__all__ = [
    "street_network", "traffic_assignment", "travel_demand", "pedestrian_flow",
    "emissions_environment", "traffic_flow", "accessibility_equity", "signal_control",
    "public_voice",
    "data_splitting", "ensembles", "multi_objective", "stochastic_optimisation",
]


def __getattr__(name: str):
    """PEP 562: makes `from engine.orchid import policy_compiler` fail with the reason it is held."""
    if name in UNAVAILABLE:
        raise ImportError("engine.orchid.%s is not vendored: %s" % (name, UNAVAILABLE[name]))
    raise AttributeError("module %r has no attribute %r" % (__name__, name))
