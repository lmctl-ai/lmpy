# Exports the statsmodels-bundled datasets used by the examples/python parity recipes to CSV (no downloads: every file
# ships inside the statsmodels 0.14.6 wheel). Run from any directory with the oracle venv's python:
#   native/.deps/oracle-venv/bin/python examples/python/data/export-statsmodels-datasets.py
# Provenance and licence (statsmodels' own dataset COPYRIGHT notes):
#   longley.csv  - Longley (1967) US macroeconomic data, the NIST StRD "Longley" set: "This is public domain."
#   sunspots.csv - yearly Wolf sunspot numbers 1700-2008: "This data is public domain."
#   grunfeld.csv - Grunfeld (1950) investment data, 11 US firms 1935-1954: "This is public domain."
#   randhie.csv  - RAND Health Insurance Experiment doctor visits (Deb and Trivedi 2002, statsmodels' count-model example
#                  data): "This is in the public domain."
#   anes96.csv   - 1996 American National Election Studies extract: "This is public domain."
#   engel.csv    - Engel (1857) Belgian household food expenditure: "This is public domain."
import os
import statsmodels.api as sm

here = os.path.dirname(os.path.abspath(__file__))
for name in ["longley", "sunspots", "grunfeld", "randhie", "anes96", "engel"]:
    df = getattr(sm.datasets, name).load_pandas().data
    df.to_csv(os.path.join(here, f"{name}.csv"), index=False)  # shortest round-trip floats
