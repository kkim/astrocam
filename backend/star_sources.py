"""
Star catalog abstraction layer.

A StarSource returns the stars (RA, Dec, magnitude) inside a cone on the sky.
Backends are interchangeable and selected by name via `get_star_source()`.

- "skyfield": Hipparcos catalog (bright stars, V <~ 9.5), parsed with Skyfield.
- "astropy":  Gaia DR3 via astroquery/Vizier (deep, V <~ 13), cached on disk.

Both download their data on first use and cache it in backend/data/.
"""
import hashlib
import json
import os
import threading
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# (ra_deg, dec_deg, magnitude, name) in J2000
Star = Tuple[float, float, float, str]

# Common names for the brightest Hipparcos stars; everything else is labelled "HIP <n>" / "Gaia <id>".
HIP_NAMES = {
    32349: "Sirius", 30438: "Canopus", 69673: "Arcturus", 91262: "Vega", 24608: "Capella",
    24436: "Rigel", 37279: "Procyon", 27989: "Betelgeuse", 7588: "Achernar", 97649: "Altair",
    21421: "Aldebaran", 80763: "Antares", 65474: "Spica", 37826: "Pollux", 113368: "Fomalhaut",
    102098: "Deneb", 49669: "Regulus", 11767: "Polaris", 54061: "Dubhe", 53910: "Merak",
    58001: "Phecda", 59774: "Megrez", 62956: "Alioth", 65378: "Mizar", 67301: "Alkaid",
    9884: "Hamal", 14135: "Menkar", 746: "Caph", 3179: "Schedar", 4427: "Gamma Cas",
    6686: "Ruchbah", 8886: "Segin", 85822: "Kochab", 82080: "Pherkad", 75097: "Gamma UMi",
    86614: "Eltanin", 95947: "Albireo", 100453: "Sadr", 107315: "Enif",
}


def _lst_deg(lon_deg: float, when: Optional[datetime] = None) -> float:
    """Local mean sidereal time in degrees (longitude east-positive)."""
    when = when or datetime.now(timezone.utc)
    d = when.timestamp() / 86400.0 + 2440587.5 - 2451545.0
    return (280.46061837 + 360.98564736629 * d + lon_deg) % 360.0


def altaz(ra_deg: float, dec_deg: float, lat_deg: float, lon_deg: float,
          when: Optional[datetime] = None) -> Tuple[float, float]:
    """(altitude, azimuth) in degrees of a J2000 RA/Dec; azimuth is from north through east (precession ignored)."""
    ha = np.deg2rad(_lst_deg(lon_deg, when) - ra_deg)
    lat, dec = np.deg2rad(lat_deg), np.deg2rad(dec_deg)
    sin_alt = np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(ha)
    north = np.sin(dec) * np.cos(lat) - np.cos(dec) * np.sin(lat) * np.cos(ha)
    east = -np.cos(dec) * np.sin(ha)
    return (float(np.rad2deg(np.arcsin(np.clip(sin_alt, -1.0, 1.0)))),
            float(np.rad2deg(np.arctan2(east, north)) % 360.0))


def altitude_deg(ra_deg: float, dec_deg: float, lat_deg: float, lon_deg: float,
                 when: Optional[datetime] = None) -> float:
    """Altitude above the horizon of a J2000 RA/Dec (precession ignored, <0.5 deg effect)."""
    return altaz(ra_deg, dec_deg, lat_deg, lon_deg, when)[0]


def _hip_name(hip: int) -> str:
    return HIP_NAMES.get(hip, f"HIP {hip}")


class StarSource(ABC):
    name: str = ""

    @abstractmethod
    def query(self, ra_deg: float, dec_deg: float, radius_deg: float) -> List[Star]:
        """Return stars within radius_deg of (ra, dec). May block on network I/O; raises on failure."""


def _angular_sep_mask(ra, dec, ra0, dec0, radius):
    r, d, r0, d0 = map(np.deg2rad, (ra, dec, ra0, dec0))
    cos_sep = np.sin(d) * np.sin(d0) + np.cos(d) * np.cos(d0) * np.cos(r - r0)
    return cos_sep >= np.cos(np.deg2rad(radius))


class SkyfieldSource(StarSource):
    """Hipparcos catalog through Skyfield, with proper motion applied to the current epoch."""
    name = "skyfield"
    MAG_LIMIT = 9.5

    def __init__(self):
        self._lock = threading.Lock()
        self._arrays: Optional[Dict[str, np.ndarray]] = None

    def _load(self):
        with self._lock:
            if self._arrays is not None:
                return self._arrays
            from skyfield.api import Loader
            from skyfield.data import hipparcos
            os.makedirs(DATA_DIR, exist_ok=True)
            loader = Loader(DATA_DIR, verbose=False)
            with loader.open(hipparcos.URL) as f:
                df = hipparcos.load_dataframe(f)
            df = df[df["magnitude"] <= self.MAG_LIMIT]
            self._arrays = {
                "hip": df.index.to_numpy(dtype=int),
                "ra": df["ra_degrees"].to_numpy(dtype=float),
                "dec": df["dec_degrees"].to_numpy(dtype=float),
                "mag": df["magnitude"].to_numpy(dtype=float),
                "pmra": df["ra_mas_per_year"].to_numpy(dtype=float),
                "pmdec": df["dec_mas_per_year"].to_numpy(dtype=float),
            }
            return self._arrays

    def query(self, ra_deg, dec_deg, radius_deg):
        a = self._load()
        years = datetime.now(timezone.utc).year - 1991.25  # Hipparcos epoch
        cosd = np.maximum(np.cos(np.deg2rad(a["dec"])), 1e-6)
        ra = a["ra"] + a["pmra"] * years / 3.6e6 / cosd  # Hipparcos pmra already includes cos(dec)
        dec = a["dec"] + a["pmdec"] * years / 3.6e6
        m = _angular_sep_mask(ra, dec, ra_deg, dec_deg, radius_deg)
        return [(float(r), float(d), float(g), _hip_name(int(h)))
                for r, d, g, h in zip(ra[m], dec[m], a["mag"][m], a["hip"][m])]


class AstropySource(StarSource):
    """
    Gaia DR3 (faint stars) + Hipparcos (bright stars) via astroquery/Vizier, cached per query on disk.
    Gaia saturates on bright stars (e.g. Polaris is missing), so Hipparcos stars brighter than
    HIP_MAG_LIMIT are taken from Hipparcos and any Gaia entry within MATCH_ARCSEC of one is dropped.
    """
    name = "astropy"
    MAG_LIMIT = 13.0
    HIP_MAG_LIMIT = 7.0
    MATCH_ARCSEC = 10.0
    ROW_LIMIT = 5000

    def _cache_path(self, ra, dec, radius):
        key = hashlib.md5(f"v2,{ra:.3f},{dec:.3f},{radius:.3f},{self.MAG_LIMIT}".encode()).hexdigest()[:12]
        return os.path.join(DATA_DIR, f"astropy_{key}.json")

    def query(self, ra_deg, dec_deg, radius_deg):
        path = self._cache_path(ra_deg, dec_deg, radius_deg)
        if os.path.exists(path):
            with open(path) as f:
                return [tuple(row) for row in json.load(f)]

        import astropy.units as u
        from astropy.coordinates import SkyCoord
        from astroquery.vizier import Vizier

        center = SkyCoord(ra_deg * u.deg, dec_deg * u.deg)

        hip = []
        res = Vizier(columns=["HIP", "RAICRS", "DEICRS", "Vmag"],
                     column_filters={"Vmag": f"<{self.HIP_MAG_LIMIT}"},
                     row_limit=self.ROW_LIMIT).query_region(center, radius=radius_deg * u.deg,
                                                           catalog="I/239/hip_main")
        if len(res):
            t = res[0]
            hip = [(float(r), float(d), float(m), _hip_name(int(h)))
                   for h, r, d, m in zip(t["HIP"], t["RAICRS"], t["DEICRS"], t["Vmag"])]

        gaia = []
        res = Vizier(columns=["Source", "RA_ICRS", "DE_ICRS", "Gmag"],
                     column_filters={"Gmag": f"<{self.MAG_LIMIT}"},
                     row_limit=self.ROW_LIMIT).query_region(center, radius=radius_deg * u.deg,
                                                           catalog="I/355/gaiadr3")
        if len(res):
            t = res[0]
            gaia = [(float(r), float(d), float(m), f"Gaia DR3...{str(int(sid))[-3:]}")
                    for sid, r, d, m in zip(t["Source"], t["RA_ICRS"], t["DE_ICRS"], t["Gmag"])
                    if np.isfinite(r) and np.isfinite(d) and np.isfinite(m)]

        if hip and gaia:
            hc = SkyCoord([h[0] for h in hip] * u.deg, [h[1] for h in hip] * u.deg)
            gc = SkyCoord([g[0] for g in gaia] * u.deg, [g[1] for g in gaia] * u.deg)
            sep = gc.match_to_catalog_sky(hc)[1]
            gaia = [g for g, s_ in zip(gaia, sep) if s_.arcsec > self.MATCH_ARCSEC]

        stars = hip + gaia
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(path, "w") as f:
            json.dump(stars, f)
        return stars


_SOURCES = {"skyfield": SkyfieldSource, "astropy": AstropySource}
_instances: Dict[str, StarSource] = {}


def available_backends() -> List[str]:
    return list(_SOURCES)


def get_star_source(name: str) -> StarSource:
    if name not in _SOURCES:
        raise ValueError(f"Unknown star backend '{name}'. Options: {available_backends()}")
    if name not in _instances:
        _instances[name] = _SOURCES[name]()
    return _instances[name]


PLANET_NAMES = {
    "mercury": "Mercury",
    "venus": "Venus",
    "mars": "Mars",
    "jupiter": "Jupiter",
    "saturn": "Saturn",
    "uranus": "Uranus",
    "neptune": "Neptune",
    "moon": "Moon",
    "sun": "Sun",
}


PLANET_DIAMETERS_KM = {
    "mercury": 4879.0,
    "venus": 12104.0,
    "mars": 6792.0,
    "jupiter": 142984.0,
    "saturn": 120536.0,
    "uranus": 50724.0,
    "neptune": 49244.0,
    "moon": 3474.8,
    "sun": 1392700.0,
}


PLANET_MAGNITUDES = {
    "mercury": 0.0,
    "venus": -4.3,
    "mars": 0.5,
    "jupiter": -2.6,
    "saturn": 0.6,
    "uranus": 5.7,
    "neptune": 7.8,
    "moon": -12.5,
    "sun": -26.7,
}


def get_planet_info(planet_key: str, lat_deg: float = 37.77, lon_deg: float = -122.42,
                    when: Optional[datetime] = None) -> Dict[str, Any]:
    """Compute apparent RA, Dec, distance (AU), and apparent angular diameter (arcsec)."""
    key = planet_key.lower().strip()
    if key not in PLANET_NAMES:
        raise ValueError(f"Unknown planet '{planet_key}'. Available: {list(PLANET_NAMES.values())}")
    
    from astropy.time import Time
    from astropy.coordinates import solar_system_ephemeris, get_body, EarthLocation
    import astropy.units as u

    when = when or datetime.now(timezone.utc)
    t = Time(when)
    loc = EarthLocation(lat=lat_deg * u.deg, lon=lon_deg * u.deg)
    with solar_system_ephemeris.set("builtin"):
        coord = get_body(key, t, loc)
    
    dist_au = float(coord.distance.au)
    diam_km = PLANET_DIAMETERS_KM.get(key, 10000.0)
    ang_diam_arcsec = float((diam_km / (dist_au * 149597870.7)) * 206265.0)
    mag = PLANET_MAGNITUDES.get(key, 1.0)
    return {
        "name": PLANET_NAMES[key],
        "key": key,
        "ra": float(coord.ra.deg),
        "dec": float(coord.dec.deg),
        "dist_au": dist_au,
        "ang_diam_arcsec": ang_diam_arcsec,
        "mag": mag,
    }


def get_planet_coords(planet_key: str, lat_deg: float = 37.77, lon_deg: float = -122.42,
                      when: Optional[datetime] = None) -> Tuple[str, float, float]:
    """Compute apparent RA and Dec (degrees) for a solar system body using Astropy's builtin ephemeris."""
    info = get_planet_info(planet_key, lat_deg, lon_deg, when)
    return info["name"], info["ra"], info["dec"]


def get_all_planets_coords(lat_deg: float = 37.77, lon_deg: float = -122.42,
                           when: Optional[datetime] = None) -> List[Dict[str, Any]]:
    """Return all solar system bodies with their current RA/Dec for the target catalog."""
    out = []
    when = when or datetime.now(timezone.utc)
    for key, display_name in PLANET_NAMES.items():
        try:
            info = get_planet_info(key, lat_deg, lon_deg, when)
            out.append({
                "name": info["name"],
                "alt": "Planet" if key not in ("moon", "sun") else key.capitalize(),
                "type": "planet",
                "ra": round(info["ra"], 5),
                "dec": round(info["dec"], 5),
                "mag": info["mag"],
                "ang_diam_arcsec": round(info["ang_diam_arcsec"], 2)
            })
        except Exception:
            pass
    return out


def get_visible_planets(ra0_deg: float, dec0_deg: float, radius_deg: float,
                        lat_deg: float = 37.77, lon_deg: float = -122.42,
                        when: Optional[datetime] = None) -> List[Any]:
    """Return solar system bodies (ra, dec, mag, name, ang_diam_arcsec) within radius_deg of (ra0, dec0)."""
    out: List[Any] = []
    when = when or datetime.now(timezone.utc)
    for key in PLANET_NAMES:
        try:
            info = get_planet_info(key, lat_deg, lon_deg, when)
            r, d, r0, d0 = map(np.deg2rad, (info["ra"], info["dec"], ra0_deg, dec0_deg))
            cos_sep = np.sin(d) * np.sin(d0) + np.cos(d) * np.cos(d0) * np.cos(r - r0)
            sep_deg = float(np.rad2deg(np.arccos(np.clip(cos_sep, -1.0, 1.0))))
            if sep_deg <= radius_deg:
                out.append((info["ra"], info["dec"], info["mag"], info["name"], info["ang_diam_arcsec"]))
        except Exception:
            pass
    return out


def resolve_name(query: str, lat_deg: float = 37.77, lon_deg: float = -122.42) -> Tuple[str, float, float]:
    """Resolve an object name (Polaris, Jupiter, M31, NGC 7000...) to (name, ra_deg, dec_deg)."""
    query = query.strip()
    if not query:
        raise ValueError("Empty search")
    
    # 1. Check solar system bodies first (instant, exact for current time)
    key = query.lower()
    if key in PLANET_NAMES:
        return get_planet_coords(key, lat_deg, lon_deg)

    # Fast offline shortcut for North Celestial Pole
    if key in ("ncp", "north pole", "true north pole", "north celestial pole"):
        return ("North Celestial Pole", 0.0, 90.0)

    # 2. Check local offline catalog of ~250 Messier & named stars
    targets_file = os.path.join(os.path.dirname(__file__), "targets.json")
    if os.path.exists(targets_file):
        try:
            with open(targets_file, "r") as f:
                targets = json.load(f)
            for t in targets:
                t_name = t.get("name", "").strip().lower()
                t_alt = (t.get("alt") or "").strip().lower()
                if key == t_name or (t_alt and key == t_alt):
                    display_name = f"{t['name']} ({t['alt']})" if t.get("alt") else t["name"]
                    return display_name, float(t["ra"]), float(t["dec"])
        except Exception:
            pass

    # 3. Astronomical coordinates via Sesame/SIMBAD
    from astropy.coordinates import SkyCoord
    from astropy.utils.data import conf
    try:
        with conf.set_temp("remote_timeout", 10):
            c = SkyCoord.from_name(query).icrs
    except Exception as e:
        raise ValueError(f"Could not resolve '{query}' ({type(e).__name__})") from e
    return query, float(c.ra.deg), float(c.dec.deg)
