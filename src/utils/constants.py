from collections import OrderedDict

ATOM_ENCODING = OrderedDict(
    {
        "C": 0,
        "N": 1,
        "O": 2,
        "S": 3,
        "H": 4,
    }
)

ATOM_ENCODING_COLOR = OrderedDict(
    {
        "C": "black",
        "N": "blue",
        "O": "red",
        "S": "yellow",
        "H": "white",
    }
)

ATOM_COLOR_MAP = OrderedDict(
    {
        "#": "orange",
        "H": "white",
        "C": "black",
        "N": "blue",
        "O": "red",
        "S": "yellow",
    }
)

NUM_TO_ATOM_TYPE = {1: "H", 6: "C", 7: "N", 8: "O"}

AA_TO_NUM = {
    "ALA": 0,
    "A": 0,
    "ARG": 1,
    "R": 1,
    "ASN": 2,
    "N": 2,
    "ASP": 3,
    "D": 3,
    "CYS": 4,
    "C": 4,
    "GLN": 5,
    "Q": 5,
    "GLU": 6,
    "E": 6,
    "GLY": 7,
    "G": 7,
    "HIS": 8,
    "H": 8,
    "ILE": 9,
    "I": 9,
    "LEU": 10,
    "L": 10,
    "LYS": 11,
    "K": 11,
    "MET": 12,
    "M": 12,
    "PHE": 13,
    "F": 13,
    "PRO": 14,
    "P": 14,
    "SER": 15,
    "S": 15,
    "THR": 16,
    "T": 16,
    "TRP": 17,
    "W": 17,
    "TYR": 18,
    "Y": 18,
    "VAL": 19,
    "V": 19,
}

ATOM_COLOR_MAP = {
    1: "white",  # Hydrogen - White (standard color in molecular viz)
    2: "#FFC0CB",  # Helium - Pink (light noble gas)
    3: "#FF0000",  # Lithium - Red (alkali metal)
    4: "#00FF00",  # Beryllium - Green (alkaline earth metal)
    5: "#FFB200",  # Boron - Orange
    6: "black",  # Carbon - Grey (standard color in molecular viz)
    7: "blue",  # Nitrogen - Blue (standard color in molecular viz)
    8: "red",  # Oxygen - Red (standard color in molecular viz)
    9: "#FFFF00",  # Fluorine - Yellow (standard color for halogens)
    10: "#FF1493",  # Neon - Deep pink (noble gas)
}
