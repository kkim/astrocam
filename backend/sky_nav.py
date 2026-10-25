"""
Navigation state shared by the whole backend: where the camera points (RA/Dec), where the rig is
(latitude/longitude), the optics (focal length, pixel size), a target-search history, and derived
values (azimuth/altitude, field of view, sidereal drift).

It is independent of the camera implementation, so it works with the real camera as well as the
mock one. A simulated camera subscribes to changes to redraw its star field; every capture and
recording stores a snapshot() next to the images.

Persistence: the "sky" and "sky_history" keys of config.json.
"""
import json
import os
import threading
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set

import numpy as np

from logger import event_logger
from star_sources import altaz, available_backends, resolve_name

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
HISTORY_LIMIT = 10
SIDEREAL_ARCSEC_PER_SEC = 15.0411  # 360 deg / 23h56m04s

# IMX179 (1.4 um pixels) behind a 500 mm lens; pointing at Vega.
DEFAULT_SKY: Dict[str, Any] = {
    "backend": "skyfield",
    "latitude": 37.77,
    "longitude": -122.42,
    "ra": 279.2347,
    "dec": 38.7837,
    "focal_length_mm": 500.0,
    "pixel_size_um": 1.4,
    "target": "Vega",
}

# listener(changed_keys) is called after every successful change
Listener = Callable[[Set[str]], None]


class SkyNav:
    def __init__(self, config_path: str = CONFIG_PATH):
        self.config_path = config_path
        self.lock = threading.RLock()
        self.listeners: List[Listener] = []
        # Star-field loading state, written by whichever camera renders the sky (idle otherwise)
        self.status: Dict[str, Any] = {"state": "idle", "stars": 0, "message": ""}

        data = self._read_config()
        self.sky: Dict[str, Any] = dict(DEFAULT_SKY)
        self.sky.update(data.get("sky", {}))
        self.history: List[Dict[str, Any]] = list(data.get("sky_history", []))[:HISTORY_LIMIT]

    # ---- derived values ----------------------------------------------------
    def arcsec_per_pixel(self) -> float:
        return 206.265 * self.sky["pixel_size_um"] / self.sky["focal_length_mm"]

    def drift_px_per_sec(self) -> float:
        """Diurnal drift of the stars across the image (westward) at the current pointing."""
        return SIDEREAL_ARCSEC_PER_SEC * float(np.cos(np.deg2rad(self.sky["dec"]))) / self.arcsec_per_pixel()

    def altaz(self):
        s = self.sky
        return altaz(s["ra"], s["dec"], s["latitude"], s["longitude"])

    def get_config(self, frame_width: int = 1920, frame_height: int = 1080) -> Dict[str, Any]:
        with self.lock:
            scale = self.arcsec_per_pixel()
            alt, az = self.altaz()
            return {
                **self.sky,
                "altitude": alt,
                "azimuth": az,
                "backends": available_backends(),
                "arcsec_per_pixel": scale,
                "fov_deg": [frame_width * scale / 3600.0, frame_height * scale / 3600.0],
                "drift_px_per_sec": self.drift_px_per_sec(),
                "history": list(self.history),
                "status": dict(self.status),
            }

    def snapshot(self) -> Dict[str, Any]:
        """Pointing/location info stored with captures and recordings."""
        with self.lock:
            alt, az = self.altaz()
            s = self.sky
            return {
                "target": s["target"],
                "ra": s["ra"],
                "dec": s["dec"],
                "altitude": alt,
                "azimuth": az,
                "latitude": s["latitude"],
                "longitude": s["longitude"],
                "focal_length_mm": s["focal_length_mm"],
                "pixel_size_um": s["pixel_size_um"],
                "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }

    # ---- changes -----------------------------------------------------------
    def subscribe(self, listener: Listener):
        self.listeners.append(listener)

    def unsubscribe(self, listener: Listener):
        if listener in self.listeners:
            self.listeners.remove(listener)

    def set(self, target: Optional[str] = None, **changes) -> bool:
        changed: Set[str] = set()
        with self.lock:
            if target and ("ra" not in changes or "dec" not in changes or changes.get("ra") is None or changes.get("dec") is None):
                try:
                    t_name, t_ra, t_dec = resolve_name(target, lat_deg=self.sky["latitude"], lon_deg=self.sky["longitude"])
                    if changes.get("ra") is None:
                        self.sky["ra"] = t_ra
                        changed.add("ra")
                    if changes.get("dec") is None:
                        self.sky["dec"] = t_dec
                        changed.add("dec")
                    target = t_name
                except Exception:
                    pass

            for k, v in changes.items():
                if v is None or k not in DEFAULT_SKY or k == "target":
                    continue
                if k == "backend":
                    if v not in available_backends():
                        return False
                    if v != self.sky[k]:
                        changed.add(k)
                    self.sky[k] = v
                    continue
                v = float(v)
                if k == "ra":
                    v %= 360.0
                elif k == "dec":
                    v = max(-90.0, min(90.0, v))
                if v != self.sky[k]:
                    changed.add(k)
                self.sky[k] = v
            if target:
                self.sky["target"] = target
                self._push_history(target, self.sky["ra"], self.sky["dec"])
                changed.add("target")
            elif changed & {"ra", "dec"}:
                self.sky["target"] = ""  # edited coordinates no longer match a named target
                changed.add("target")

            if changed & {"ra", "dec", "pixel_size_um", "focal_length_mm", "backend"}:
                self.status = {
                    "state": "loading",
                    "stars": 0,
                    "message": f"Loading {self.sky['backend']} catalog...",
                }

            self._save()
        for listener in list(self.listeners):
            try:
                listener(set(changed))
            except Exception as e:
                event_logger.log(f"Navigation listener error: {e}")
        return True

    def search_target(self, query: str) -> Dict[str, Any]:
        """Resolve an object name (e.g. 'Polaris', 'Jupiter', 'M31') and point the sky at it."""
        try:
            name, ra, dec = resolve_name(query, lat_deg=self.sky["latitude"], lon_deg=self.sky["longitude"])
        except Exception as e:
            return {"success": False, "error": str(e)}
        self.set(ra=ra, dec=dec, target=name)
        return {"success": True, "name": name, "ra": ra, "dec": dec}

    def _push_history(self, name: str, ra: float, dec: float):
        self.history = [h for h in self.history if h["name"].lower() != name.lower()]
        self.history.insert(0, {"name": name, "ra": ra, "dec": dec})
        del self.history[HISTORY_LIMIT:]

    # ---- persistence -------------------------------------------------------
    def _read_config(self) -> Dict[str, Any]:
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save(self):
        try:
            data = self._read_config()
            data["sky"] = dict(self.sky)
            data["sky_history"] = list(self.history)
            with open(self.config_path, "w") as f:
                json.dump(data, f)
        except Exception as e:
            event_logger.log(f"Error saving navigation config: {e}")


_nav: Optional[SkyNav] = None
_nav_lock = threading.Lock()


def get_sky_nav() -> SkyNav:
    global _nav
    with _nav_lock:
        if _nav is None:
            _nav = SkyNav()
        return _nav


def write_nav_sidecar(image_path: str, camera_settings: Optional[Dict[str, Any]] = None):
    """Stores the current navigation snapshot and camera settings as <image_path>.nav.json next to a captured image."""
    try:
        data = get_sky_nav().snapshot()
        if camera_settings:
            data["camera_settings"] = camera_settings
        with open(image_path + ".nav.json", "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        event_logger.log(f"Could not save navigation info for {os.path.basename(image_path)}: {e}")


def write_session_nav(session_dir: str, camera_settings: Optional[Dict[str, Any]] = None):
    """Stores the current navigation snapshot and camera settings as nav.json inside a recording session folder."""
    try:
        data = get_sky_nav().snapshot()
        if camera_settings:
            data["camera_settings"] = camera_settings
        with open(os.path.join(session_dir, "nav.json"), "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        event_logger.log(f"Could not save navigation info for {os.path.basename(session_dir)}: {e}")


def read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}
