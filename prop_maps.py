# Default reference temperature used throughout
DEFAULT_TEMP_K = 298.0

PROP_MAP = {
    "ABS": {
        "formula": "C 17.03 H 18.903 N 1",
        "hof_kjmol": 90.31,
        "phase": "G",
        "temp_k": DEFAULT_TEMP_K,
        "is_fuel": True,
    },
    "Sorbitol": {
        "formula": "C 6 H 14 O 6",
        "hof_kjmol": -1353.7,
        "phase": "G",
        "temp_k": DEFAULT_TEMP_K,
        "is_fuel": True,
    },
    "PMMA": {
        "formula": "C 5 O 2 H 8",
        "hof_kjmol": -184.48,
        "phase": "G",
        "temp_k": DEFAULT_TEMP_K,
        "is_fuel": True,
    },
    "PBAN": {
        "formula": "C 654 H 848 N 89 O 4",
        "hof_kjmol": 6544.8054,
        "phase": "G",
        "temp_k": DEFAULT_TEMP_K,
        "is_fuel": True,
    },
    "Paraffin": {
        "formula": "C 25 H 52",
        "hof_kjmol": -621.94,
        "phase": "G",
        "temp_k": DEFAULT_TEMP_K,
        "is_fuel": True,
    },
    "HTPB": {
        "formula": "C 10 H 15.4 O 0.07",
        "hof_kjmol": -51.88,
        "phase": "G",
        "temp_k": DEFAULT_TEMP_K,
        "is_fuel": True,
    },
    "H2": {
        "formula": "H 2",
        "hof_kjmol": 0.0,
        "phase": "G",
        "temp_k": DEFAULT_TEMP_K,
        "is_fuel": True,
    },
    "PLA": {
        "formula": "C 3 H 4 O 2",
        "hof_kjmol": -790.0,
        "phase": "G",
        "temp_k": DEFAULT_TEMP_K,
        "is_fuel": True,
    },
    "Wood (Spanish Oak)": {
        "formula": "C 1.74 H 2.48 O 1",
        "hof_kjmol": -305.0,
        "phase": "G",
        "temp_k": DEFAULT_TEMP_K,
        "is_fuel": True,
    },
    "N2O": {
        "formula": "N 2 O 1",
        "hof_kjmol": 82.05,
        "phase": "G",
        "temp_k": DEFAULT_TEMP_K,
        "is_fuel": False,
    },
    "O2": {
        "formula": "O 2",
        "hof_kjmol": 0.0,
        "phase": "G",
        "temp_k": DEFAULT_TEMP_K,
        "is_fuel": False,
    },
    "O3": {
        "formula": "O 3",
        "hof_kjmol": 142.67,
        "phase": "G",
        "temp_k": DEFAULT_TEMP_K,
        "is_fuel": False,
    },
}
