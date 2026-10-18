#!/usr/bin/env python3
"""
Builds backend/targets.json, the offline list behind the search-box autocomplete.

Resolves Messier objects (M1-M110) and well-known named stars through SIMBAD/Sesame and stores
{name, alt, type, ra, dec} (J2000, degrees). Needs network access; run once and commit the result:

    backend/venv/bin/python bin/build_targets.py
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

from astropy.coordinates import SkyCoord
from astropy.utils.data import conf

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend", "targets.json")

MESSIER_NAMES = {
    1: "Crab Nebula", 2: "", 3: "", 4: "", 5: "", 6: "Butterfly Cluster", 7: "Ptolemy Cluster",
    8: "Lagoon Nebula", 9: "", 10: "", 11: "Wild Duck Cluster", 12: "", 13: "Hercules Cluster",
    14: "", 15: "", 16: "Eagle Nebula", 17: "Omega Nebula", 18: "", 19: "", 20: "Trifid Nebula",
    21: "", 22: "", 23: "", 24: "Sagittarius Star Cloud", 25: "", 26: "", 27: "Dumbbell Nebula",
    28: "", 29: "", 30: "", 31: "Andromeda Galaxy", 32: "", 33: "Triangulum Galaxy", 34: "",
    35: "", 36: "", 37: "", 38: "", 39: "", 40: "Winnecke 4", 41: "", 42: "Orion Nebula",
    43: "De Mairan's Nebula", 44: "Beehive Cluster", 45: "Pleiades", 46: "", 47: "", 48: "",
    49: "", 50: "", 51: "Whirlpool Galaxy", 52: "", 53: "", 54: "", 55: "", 56: "", 57: "Ring Nebula",
    58: "", 59: "", 60: "", 61: "", 62: "", 63: "Sunflower Galaxy", 64: "Black Eye Galaxy",
    65: "", 66: "", 67: "", 68: "", 69: "", 70: "", 71: "", 72: "", 73: "", 74: "Phantom Galaxy",
    75: "", 76: "Little Dumbbell", 77: "", 78: "", 79: "", 80: "", 81: "Bode's Galaxy",
    82: "Cigar Galaxy", 83: "Southern Pinwheel", 84: "", 85: "", 86: "", 87: "Virgo A", 88: "",
    89: "", 90: "", 91: "", 92: "", 93: "", 94: "", 95: "", 96: "", 97: "Owl Nebula", 98: "",
    99: "", 100: "", 101: "Pinwheel Galaxy", 102: "Spindle Galaxy", 103: "", 104: "Sombrero Galaxy",
    105: "", 106: "", 107: "", 108: "", 109: "", 110: "",
}

STARS = """
Sirius Canopus Arcturus Rigil Kentaurus|Alpha Centauri Vega Capella Rigel Procyon Betelgeuse
Achernar Hadar Altair Acrux Aldebaran Antares Spica Pollux Fomalhaut Deneb Mimosa Regulus
Adhara Castor Gacrux Shaula Bellatrix Elnath Miaplacidus Alnilam Alnair Alnitak Alioth Dubhe
Mirfak Wezen Sargas Kaus Australis Avior Alkaid Menkalinan Atria Alhena Peacock Alsephina
Mirzam Alphard Polaris Hamal Algieba Diphda Mizar Nunki Menkent Mirach Alpheratz Rasalhague
Kochab Saiph Denebola Algol Tiaki Muhlifain Aspidiske Suhail Alphecca Mintaka Sadr Eltanin
Schedar Naos Almach Caph Izar Dschubba Larawag Merak Ankaa Girtab Enif Scheat Sabik Phecda
Aludra Markeb Navi Markab Aljanah Acrab Zosma Arneb Megrez Albireo Alcor Thuban Alderamin
Alfirk Sheratan Mebsuta Rastaban Kornephoros Unukalhai Cor Caroli Vindemiatrix Porrima
Zubenelgenubi Zubeneschamali Alshain Tarazed Sadalsuud Sadalmelik Deneb Algedi Algenib
Ruchbah Segin Pherkad Sulafat Sheliak Rotanev Sualocin Alcyone Maia Electra Merope Taygeta
Celaeno Atlas Pleione Rukbat Nashira Alrescha Zaurak Cursa Menkar Mira Almaaz Hassaleh
Ras Algethi Kaffaljidhma Acamar Gienah Algorab Kraz Alchiba Mu Cephei|Garnet Star
""".strip()


def star_names():
    # Names are whitespace separated; multi-word names use "_" via the split below
    raw = STARS.replace("Rigil Kentaurus|Alpha Centauri", "Rigil_Kentaurus").replace(
        "Cor Caroli", "Cor_Caroli").replace("Kaus Australis", "Kaus_Australis").replace(
        "Deneb Algedi", "Deneb_Algedi").replace("Ras Algethi", "Ras_Algethi").replace(
        "Mu Cephei|Garnet Star", "Garnet_Star")
    out = []
    for tok in raw.split():
        n = tok.replace("_", " ")
        if n not in out:
            out.append(n)
    return out


def resolve(item):
    name, alt, kind, query = item
    try:
        with conf.set_temp("remote_timeout", 20):
            c = SkyCoord.from_name(query).icrs
        return {"name": name, "alt": alt, "type": kind, "ra": round(float(c.ra.deg), 5), "dec": round(float(c.dec.deg), 5)}
    except Exception as e:
        print(f"  failed: {name} ({type(e).__name__})", file=sys.stderr)
        return None


def main():
    items = [(f"M{n}", alt, "messier", f"M {n}") for n, alt in MESSIER_NAMES.items()]
    items += [(n, "", "star", n) for n in star_names() if n != "Garnet Star"]
    items.append(("Garnet Star", "Mu Cephei", "star", "mu Cep"))
    with ThreadPoolExecutor(max_workers=6) as ex:
        results = [r for r in ex.map(resolve, items) if r]
    with open(OUT, "w") as f:
        json.dump(results, f, separators=(",", ":"))
    print(f"Wrote {len(results)}/{len(items)} targets to {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
