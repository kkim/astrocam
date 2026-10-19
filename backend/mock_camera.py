import cv2
import threading
import time
import numpy as np
import os
import json
from typing import Optional, Dict, Any, Callable, Tuple
from logger import event_logger
from star_sources import get_star_source, altitude_deg, get_visible_planets, PLANET_NAMES
from sky_nav import get_sky_nav

class MockCamera:
    """
    Simulates starfield acquisition with realistic celestial drift,
    telescope mount compensation, optical noise, and realistic SV205 camera parameter controls.
    """
    def __init__(self, get_duty_callback: Optional[Callable[[], float]] = None):
        self.width = 1920
        self.height = 1080
        self.full_width = int(self.width * 2.0)
        self.full_height = int(self.height * 1.5)
        self.is_running = True
        self.frame_id: int = 0
        self.lock = threading.Lock()
        self.get_duty_callback = get_duty_callback

        self.format = "YUYV"
        self.params: Dict[str, Any] = {
            "brightness": 0, "contrast": 50, "saturation": 0,
            "gain": 64, "exposure": 1000, "sharpness": 50, "auto_exposure": 0,
            "auto_wb": 0, "gamma": 100
        }
        cfg_path = os.path.join(os.path.dirname(__file__), "config.json")
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, "r") as f:
                    saved = json.load(f).get("camera_settings", {})
                    for k in self.params:
                        if k in saved:
                            self.params[k] = saved[k]
            except Exception:
                pass

        # Drift position
        self.pos_x = self.width * 0.25
        self.pos_y = self.height * 0.25
        self.last_update_time = time.time()

        # Load persisted simulation parameters
        config_data = {}
        if os.path.exists("config.json"):
            try:
                with open("config.json", "r") as f:
                    config_data = json.load(f)
            except Exception:
                pass

        # Drift speed and direction are derived from the sky settings (see _update_motion).
        self.camera_angle = np.pi
        self.sim_drift_speed = 0.0
        self.sim_drift_angle = np.pi

        self.radius = 8
        self.noise_canvas = np.clip(
            24.0 + np.random.normal(0.0, 2.5, (self.height + 64, self.width + 64, 3)),
            0.0, 255.0
        ).astype(np.uint8)
        self._fallback_patches = self._setup_star_patches()
        self.star_patches = self._fallback_patches

        # Real-sky star field (catalog-driven). Pointing/optics live in the shared SkyNav.
        self.nav = get_sky_nav()
        self._sky_generation = 0
        self.nav.subscribe(self._on_nav_change)
        self._update_motion()
        self._recenter()
        self._reload_sky_async()
        self.raw_frame: Optional[np.ndarray] = None

        self.thread = threading.Thread(target=self._sim_loop, daemon=True)
        self.thread.start()

    def _create_star_patch(self, radius: int = 8, peak: float = 220.0, sigma: Optional[float] = None,
                           color: Optional[Tuple[float, float, float]] = None) -> np.ndarray:
        """Creates a smooth astronomical point spread function (PSF) of given radius."""
        y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
        dist_sq = x*x + y*y
        mask = dist_sq <= radius*radius
        if sigma is None:
            sigma = 0.35 * radius
        prof = np.exp(-dist_sq / (2.0 * sigma**2)) * mask * peak
        if color is not None:
            patch = np.zeros((2*radius+1, 2*radius+1, 3), dtype=np.float32)
            patch[:, :, 0] = prof * color[0]
            patch[:, :, 1] = prof * color[1]
            patch[:, :, 2] = prof * color[2]
        else:
            patch = np.repeat(prof[:, :, None], 3, axis=2)
        return patch.astype(np.float32)

    def _create_saturn_patch(self, globe_r: float) -> np.ndarray:
        """
        Realistic Saturn rendered in exact proportion to the lens focal length and plate scale:
        - Globe equatorial radius: globe_r
        - Ring outer radius: a ~ globe_r * 2.273 (274,000 km outer ring vs 120,536 km globe)
        - Major axis Position Angle: 93.3° (nearly horizontal, ~3.3° tilt)
        - Opening angle: 8.6° (sub-observer latitude)
        - Cassini division, bright B ring, outer A ring, inner C ring.
        - True 3D layering (Southern ring passes in front of the globe, Northern behind).
        """
        globe_r = max(2.5, globe_r)
        a = globe_r * 2.273
        b = max(1.5, a * np.sin(np.deg2rad(8.6)))
        patch_r = int(np.ceil(a + 4))
        patch = np.zeros((2*patch_r+1, 2*patch_r+1, 3), dtype=np.float32)
        y, x = np.ogrid[-patch_r:patch_r+1, -patch_r:patch_r+1]

        theta = np.deg2rad(3.3)
        xr = x * np.cos(theta) - y * np.sin(theta)
        yr = x * np.sin(theta) + y * np.cos(theta)

        ring_rad = np.sqrt((xr / a)**2 + (yr / b)**2)
        ring_intensity = np.zeros_like(ring_rad)
        ring_intensity[(ring_rad >= 0.45) & (ring_rad < 0.55)] = 45.0   # C ring
        ring_intensity[(ring_rad >= 0.55) & (ring_rad < 0.75)] = 210.0  # B ring
        ring_intensity[(ring_rad >= 0.75) & (ring_rad < 0.78)] = 20.0   # Cassini division
        ring_intensity[(ring_rad >= 0.78) & (ring_rad <= 1.02)] = 145.0 # A ring

        # Oblate globe: polar radius is 0.90 * equatorial
        globe_norm = np.sqrt((xr / globe_r)**2 + (yr / (globe_r * 0.90))**2)
        globe_mask = globe_norm <= 1.0
        mu = np.sqrt(np.maximum(0.0, 1.0 - np.clip(globe_norm**2, 0.0, 1.0)))
        globe_intensity = (120.0 + 135.0 * mu) * globe_mask

        # 3D occlusion: Southern ring (yr >= 0) in front, Northern ring (yr < 0) behind globe
        in_front_ring = (yr >= 0) & (ring_rad >= 0.45) & (ring_rad <= 1.02)
        combined = globe_intensity.copy()
        combined = np.where((yr < 0) & (~globe_mask), ring_intensity, combined)
        combined = np.where(in_front_ring, ring_intensity, combined)

        patch[:, :, 0] = combined * 0.65
        patch[:, :, 1] = combined * 0.88
        patch[:, :, 2] = combined * 1.00
        ksize = 5 if globe_r > 5 else 3
        return cv2.GaussianBlur(patch, (ksize, ksize), 1.0)

    def _create_jupiter_patch(self, radius_px: float) -> np.ndarray:
        """
        Realistic Jupiter rendered in proportion to the lens focal length and plate scale:
        - Oblate disc (equatorial radius = radius_px, polar = 0.935 * radius_px).
        - Limb darkening across disc.
        - Warm reddish-brown North & South equatorial cloud belts (NEB, SEB).
        """
        radius = max(2.5, radius_px)
        patch_r = int(np.ceil(radius + 4))
        patch = np.zeros((2*patch_r+1, 2*patch_r+1, 3), dtype=np.float32)
        y, x = np.ogrid[-patch_r:patch_r+1, -patch_r:patch_r+1]

        dist_norm = np.sqrt((x / radius)**2 + (y / (radius * 0.935))**2)
        disc_mask = dist_norm <= 1.0
        mu = np.sqrt(np.maximum(0.0, 1.0 - np.clip(dist_norm**2, 0.0, 1.0)))
        limb = 0.45 + 0.55 * mu

        # Cloud belts along horizontal
        yn = y / max(radius * 0.935, 1.0)
        belt1 = np.exp(-((yn - 0.22)**2) / 0.015)
        belt2 = np.exp(-((yn + 0.22)**2) / 0.015)
        belts = belt1 + belt2

        base = 240.0 * limb * disc_mask
        patch[:, :, 0] = base * np.clip(0.85 - 0.30 * belts, 0.0, 1.0)
        patch[:, :, 1] = base * np.clip(0.92 - 0.24 * belts, 0.0, 1.0)
        patch[:, :, 2] = base * np.clip(0.98 - 0.13 * belts, 0.0, 1.0)
        ksize = 5 if radius > 5 else 3
        return cv2.GaussianBlur(patch, (ksize, ksize), 1.0)

    def _create_venus_patch(self, radius_px: float) -> np.ndarray:
        """
        Realistic Venus rendered in proportion:
        - Apparent disc radius in pixels.
        - Crescent phase illumination (~3% crescent for mid-October 2026).
        - Brilliant white illumination with atmospheric limb.
        """
        radius = max(2.5, radius_px)
        patch_r = int(np.ceil(radius + 4))
        patch = np.zeros((2*patch_r+1, 2*patch_r+1, 3), dtype=np.float32)
        y, x = np.ogrid[-patch_r:patch_r+1, -patch_r:patch_r+1]

        theta = np.deg2rad(143.0)
        xr = x * np.cos(theta) - y * np.sin(theta)
        yr = x * np.sin(theta) + y * np.cos(theta)

        dist_circle = np.hypot(xr, yr)
        in_disc = dist_circle <= radius
        k = 0.03
        x_term = radius * (2.0 * k - 1.0) * np.sqrt(np.maximum(0.0, 1.0 - np.clip((yr / radius)**2, 0.0, 1.0)))
        lit_mask = in_disc & (xr >= x_term)

        intensity = np.zeros_like(xr)
        intensity[lit_mask] = 255.0
        patch[:, :, 0] = intensity
        patch[:, :, 1] = intensity
        patch[:, :, 2] = intensity
        ksize = 5 if radius > 5 else 3
        return cv2.GaussianBlur(patch, (ksize, ksize), 1.5)

    def _create_planet_patch(self, name: str, mag: float, ang_diam_arcsec: float, scale_arcsec: float) -> np.ndarray:
        """Returns an astronomically proportioned patch matching the lens focal length."""
        radius_px = max(2.0, (ang_diam_arcsec / max(scale_arcsec, 0.001)) / 2.0)
        key = name.capitalize()
        if key == "Saturn":
            return self._create_saturn_patch(globe_r=radius_px)
        elif key == "Jupiter":
            return self._create_jupiter_patch(radius_px=radius_px)
        elif key == "Venus":
            return self._create_venus_patch(radius_px=radius_px)
        elif key == "Mars":
            r_int = int(np.ceil(radius_px))
            return self._create_star_patch(radius=max(6, r_int + 2), peak=255.0, sigma=max(2.0, radius_px * 0.6), color=(0.50, 0.70, 1.00))
        elif key == "Mercury":
            r_int = int(np.ceil(radius_px))
            return self._create_star_patch(radius=max(6, r_int + 2), peak=240.0, sigma=max(2.0, radius_px * 0.6), color=(0.85, 0.85, 0.85))
        elif key == "Moon":
            r_moon = int(min(300.0, radius_px))
            patch = np.zeros((2*r_moon+1, 2*r_moon+1, 3), dtype=np.float32)
            y, x = np.ogrid[-r_moon:r_moon+1, -r_moon:r_moon+1]
            dist = np.hypot(x, y)
            mask = dist <= r_moon
            prof = np.clip((r_moon - dist) * 15.0, 0.0, 255.0) * mask
            patch[:, :, 0] = prof * 0.95
            patch[:, :, 1] = prof * 0.98
            patch[:, :, 2] = prof * 1.00
            return patch
        elif key == "Sun":
            r_sun = int(min(300.0, radius_px))
            y, x = np.ogrid[-r_sun:r_sun+1, -r_sun:r_sun+1]
            dist = np.hypot(x, y)
            core = (dist <= r_sun) * 255.0
            glow = np.exp(-dist / max(r_sun * 0.25, 4.0)) * 160.0
            total = np.clip(core + glow, 0.0, 255.0)
            patch = np.zeros((2*r_sun+1, 2*r_sun+1, 3), dtype=np.float32)
            patch[:, :, 0] = total * 0.70
            patch[:, :, 1] = total * 0.95
            patch[:, :, 2] = total * 1.00
            return patch
        elif key == "Uranus":
            return self._create_star_patch(radius=max(6, int(radius_px)+2), peak=200.0, sigma=max(1.8, radius_px*0.6), color=(1.00, 0.95, 0.70))
        elif key == "Neptune":
            return self._create_star_patch(radius=max(5, int(radius_px)+2), peak=160.0, sigma=max(1.5, radius_px*0.6), color=(1.00, 0.80, 0.50))
        else:
            return self._create_star_patch(radius=int(max(8, radius_px+2)), peak=255.0, sigma=max(2.4, radius_px*0.5))

    def _setup_star_patches(self) -> list:
        """
        Precomputes realistic star profiles matching SV205 telescopic observations:
        - One bright guide star (radius 5, peak ~220)
        - 1-2 dim stars of brightness ~10 (radius 5)
        - 3-5 very dim stars of brightness ~1-3 (radius 5)
        """
        radius = self.radius

        # Guide star positioned near center of drift track
        bright_star = (int(self.full_width * 0.45), int(self.full_height * 0.45), 220.0)

        # 2 dim stars (~10 counts above baseline)
        dim_stars = [
            (bright_star[0] + 250, bright_star[1] - 170, 10.0),
            (bright_star[0] - 320, bright_star[1] + 180, 9.5),
        ]

        # 4 very dim stars (1-3 counts above baseline)
        very_dim_stars = [
            (bright_star[0] - 140, bright_star[1] - 210, 2.8),
            (bright_star[0] + 370, bright_star[1] + 120, 2.0),
            (bright_star[0] + 160, bright_star[1] + 250, 1.6),
            (bright_star[0] - 410, bright_star[1] - 150, 2.2),
        ]

        all_stars = [bright_star] + dim_stars + very_dim_stars
        return [(sx, sy, self._create_star_patch(radius=radius, peak=peak)) for sx, sy, peak in all_stars]

    # ---- Catalog-driven sky -------------------------------------------------
    @property
    def sky(self) -> Dict[str, Any]:
        return self.nav.sky

    def _on_nav_change(self, changed):
        """React to pointing/optics changes made through the shared navigation state."""
        if changed & {"dec", "pixel_size_um", "focal_length_mm"}:
            self._update_motion()
        if changed & {"ra", "dec"}:
            self._recenter()
        if changed & {"ra", "dec", "pixel_size_um", "focal_length_mm", "backend"}:
            self._reload_sky_async()

    def _update_motion(self):
        """
        Derive the diurnal drift from the pointing and plate scale. Stars move west (toward -x in
        the image, since north is up and east is right).
        """
        self.sim_drift_speed = self.nav.drift_px_per_sec()
        self.camera_angle = np.pi
        self.sim_drift_angle = np.pi

    def _recenter(self):
        """Put the pointing (RA/Dec) at the middle of the frame."""
        self.pos_x = (self.full_width - self.width) / 2.0
        self.pos_y = (self.full_height - self.height) / 2.0

    def _reload_sky_async(self):
        self._sky_generation += 1
        gen = self._sky_generation
        self.sky_status = {"state": "loading", "stars": 0, "message": f"Loading {self.sky['backend']} catalog..."}
        threading.Thread(target=self._reload_sky, args=(gen, dict(self.sky)), daemon=True).start()

    def _reload_sky(self, gen: int, sky: Dict[str, Any]):
        """Query the catalog and project stars into virtual-canvas coordinates."""
        try:
            scale_rad = (sky["pixel_size_um"] * 1e-3 / sky["focal_length_mm"])  # radians per pixel
            scale_arcsec = float(np.rad2deg(scale_rad) * 3600.0)
            half_diag_px = 0.5 * float(np.hypot(self.full_width, self.full_height))
            radius_deg = float(np.rad2deg(half_diag_px * scale_rad)) * 1.02
            stars = get_star_source(sky["backend"]).query(sky["ra"], sky["dec"], radius_deg)
            planets = get_visible_planets(sky["ra"], sky["dec"], radius_deg, sky["latitude"], sky["longitude"])
            all_objects = list(planets) + list(stars)

            ra0, dec0 = np.deg2rad(sky["ra"]), np.deg2rad(sky["dec"])
            patches = []
            visible = []  # (mag, name) of objects landing on the simulated canvas
            planet_names_set = set(PLANET_NAMES.values())

            for obj in all_objects:
                ra, dec, mag, name = obj[0], obj[1], obj[2], obj[3]
                ang_diam_arcsec = obj[4] if len(obj) > 4 else 0.0
                r, d = np.deg2rad(ra), np.deg2rad(dec)
                # Gnomonic (tangent-plane) projection
                cosc = np.sin(dec0) * np.sin(d) + np.cos(dec0) * np.cos(d) * np.cos(r - ra0)
                if cosc <= 0:
                    continue
                xi = np.cos(d) * np.sin(r - ra0) / cosc
                eta = (np.cos(dec0) * np.sin(d) - np.sin(dec0) * np.cos(d) * np.cos(r - ra0)) / cosc
                x = int(round(self.full_width / 2 + xi / scale_rad))   # East to the right
                y = int(round(self.full_height / 2 - eta / scale_rad))  # North up

                is_planet = name in planet_names_set
                if is_planet:
                    patch = self._create_planet_patch(name, mag, ang_diam_arcsec, scale_arcsec)
                else:
                    # Magnitude scaling: anchor magnitude 1.0 at peak = 220.0 counts.
                    # Every magnitude fainter is ~2.5x dimmer (flux ~ 10^(-0.4 * (mag - 1.0))).
                    flux = 220.0 * (10.0 ** (-0.4 * (mag - 1.0)))

                    # For stars mag <= 2.0 (bright stars like Polaris, Vega, Rigel), widen sigma and radius
                    if mag <= 2.0:
                        # e.g. mag 2.0 -> sigma ~ 2.8, mag 0.0 -> sigma ~ 4.2, mag -1.4 (Sirius) -> sigma ~ 5.2
                        star_sigma = 2.8 + max(0.0, 2.0 - mag) * 0.7
                        star_radius = int(min(16, round(star_sigma * 2.5)))
                    else:
                        star_radius = self.radius  # 8 px
                        star_sigma = 2.4

                    peak = float(np.clip(flux, 0.5, 255.0))
                    patch = self._create_star_patch(radius=star_radius, peak=peak, sigma=star_sigma)

                pr = patch.shape[0] // 2
                if not (pr <= x < self.full_width - pr and pr <= y < self.full_height - pr):
                    continue

                patches.append((x, y, patch))
                visible.append((float(mag), name))

            alt = altitude_deg(sky["ra"], sky["dec"], sky["latitude"], sky["longitude"])
            if gen != self._sky_generation:
                return  # superseded by a newer request
            self.star_patches = patches
            brightest = sorted(visible)[:3]
            listing = ", ".join(f"{m:.1f} {n}" for m, n in brightest)
            if len(visible) > len(brightest):
                listing += ", ..."
            note = f"{len(patches)} stars" + (f" ({listing})" if listing else "") + f"; target altitude {alt:.0f}°"
            if alt < 0:
                note += " (below horizon)"
            self.sky_status = {"state": "ready", "stars": len(patches), "altitude": alt, "message": note}
            self.nav.status = dict(self.sky_status)
            event_logger.log(f"Sky loaded: {note}")
        except Exception as e:
            if gen != self._sky_generation:
                return
            self.star_patches = self._fallback_patches
            self.sky_status = {"state": "error", "stars": 0,
                               "message": f"{sky['backend']} failed ({e}); using fake stars"}
            self.nav.status = dict(self.sky_status)
            event_logger.log(f"Sky catalog error: {e}; using fake stars")

    def _get_current_duty(self) -> float:
        if self.get_duty_callback:
            try:
                return float(self.get_duty_callback())
            except Exception:
                pass
        return 80.0

    def _sim_loop(self):
        while self.is_running:
            t_start = time.perf_counter()
            now = time.time()
            dt = now - self.last_update_time

            duty = self._get_current_duty()

            # Diurnal drift vector
            v_diurnal_x = self.sim_drift_speed * np.cos(self.sim_drift_angle)
            v_diurnal_y = self.sim_drift_speed * np.sin(self.sim_drift_angle)

            # Mount compensation vector: at 85% duty cycle, mount speed matches sim_drift_speed
            v_mount_mag = - (duty / 85.0) * self.sim_drift_speed
            v_mount_x = v_mount_mag * np.cos(self.camera_angle)
            v_mount_y = v_mount_mag * np.sin(self.camera_angle)

            v_x = v_diurnal_x + v_mount_x
            v_y = v_diurnal_y + v_mount_y

            self.pos_x = (self.pos_x + v_x * dt) % (self.full_width - self.width)
            self.pos_y = (self.pos_y + v_y * dt) % (self.full_height - self.height)
            self.last_update_time = now

            frame = self._generate_mock_frame(duty)
            with self.lock:
                self.raw_frame = frame
                self.frame_id += 1

            # Frame rate is tied directly to exposure integration time:
            # 1 unit = 100 µs (0.0001 s) -> FPS = 10000 / exposure.
            # e.g.:
            #   exposure = 100  (10 ms)  -> 100 FPS
            #   exposure = 200  (20 ms)  -> 50 FPS
            #   exposure = 300  (30 ms)  -> 33.3 FPS
            #   exposure = 500  (50 ms)  -> 20 FPS
            #   exposure = 1000 (100 ms) -> 10 FPS
            #   exposure = 5000 (500 ms) -> 2 FPS
            #   exposure = 10000 (1.0 s) -> 1 FPS
            exp_val = float(self.params.get("exposure", 300))
            exp_sec = max(exp_val * 0.0001, 0.005)

            elapsed = time.perf_counter() - t_start
            sleep_time = max(0.0005, exp_sec - elapsed)
            time.sleep(sleep_time)

    def _generate_mock_frame(self, duty: float) -> np.ndarray:
        # Dynamic slice from precomputed noise canvas to simulate sensor read noise
        oy = (self.frame_id * 17) % 64
        ox = (self.frame_id * 29) % 64
        frame = self.noise_canvas[oy : oy + self.height, ox : ox + self.width].copy()

        # 1. Exposure and Gain controls
        exp_val = float(self.params.get("exposure", 300))
        gain_val = float(self.params.get("gain", 0))

        exp_factor = max(exp_val / 300.0, 0.01)
        gain_factor = 1.0 + (gain_val / 64.0)
        flux_factor = exp_factor * gain_factor

        cx = int(self.pos_x)
        cy = int(self.pos_y)
        r = self.radius

        for sx, sy, patch in self.star_patches:
            fx = sx - cx
            fy = sy - cy
            pr = patch.shape[0] // 2
            if (fx - pr) >= 0 and (fx + pr + 1) <= self.width and (fy - pr) >= 0 and (fy + pr + 1) <= self.height:
                scaled = np.clip(patch * flux_factor, 0.0, 255.0).astype(np.uint8)
                frame[fy - pr : fy + pr + 1, fx - pr : fx + pr + 1] = cv2.add(
                    frame[fy - pr : fy + pr + 1, fx - pr : fx + pr + 1], scaled
                )

        # 2. Brightness (black level offset, range -10 to 266)
        brightness_val = float(self.params.get("brightness", 0))
        if brightness_val != 0:
            diff = int(brightness_val * 1.5)
            if diff > 0:
                frame = cv2.add(frame, diff)
            elif diff < 0:
                frame = cv2.subtract(frame, -diff)

        # 3. Contrast (S-Curve)
        contrast_val = float(self.params.get("contrast", 32))
        if contrast_val != 32:
            slope = contrast_val / 32.0
            x = np.arange(256, dtype=np.float32)
            lut = np.clip(128.0 + (x - 128.0) * slope, 0.0, 255.0).astype(np.uint8)
            frame = cv2.LUT(frame, lut)

        # 4. Sharpness (Edge Enhancement)
        sharpness_val = float(self.params.get("sharpness", 0))
        if sharpness_val > 0:
            k = sharpness_val * 0.35
            kernel = np.array([
                [0,    -k,          0],
                [-k,   1.0 + 4 * k, -k],
                [0,    -k,          0]
            ], dtype=np.float32)
            frame = cv2.filter2D(frame, -1, kernel)

        # 5. Saturation (Chroma)
        saturation_val = float(self.params.get("saturation", 64))
        if saturation_val == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        elif saturation_val != 64:
            sat_scale = max(saturation_val / 64.0, 0.0)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[:, :, None].astype(np.float32)
            f_flt = frame.astype(np.float32)
            frame = np.clip(gray + (f_flt - gray) * sat_scale, 0.0, 255.0).astype(np.uint8)

        # 6. Gamma (Power-law tone curve)
        gamma_val = float(self.params.get("gamma", 100))
        if gamma_val != 100 and gamma_val > 0:
            power = 100.0 / gamma_val
            x = np.arange(256, dtype=np.float32) / 255.0
            lut = np.clip((x ** power) * 255.0, 0.0, 255.0).astype(np.uint8)
            frame = cv2.LUT(frame, lut)

        return frame

    def reconnect(self) -> bool:
        """Mock camera reconnect is instantaneous."""
        event_logger.log("Camera: Mock camera reconnected.")
        return True

    def get_raw_frame(self) -> Optional[np.ndarray]:
        with self.lock:
            if self.raw_frame is None:
                return None
            return self.raw_frame.copy()

    def get_fresh_frame(self, last_frame_id: int = -1) -> Tuple[Optional[np.ndarray], int]:
        """Returns the frame ONLY if it is newer than last_frame_id, along with current frame_id."""
        with self.lock:
            if self.raw_frame is None or self.frame_id == last_frame_id:
                return None, self.frame_id
            return self.raw_frame.copy(), self.frame_id

    def set_format(self, fmt: str) -> bool:
        self.format = fmt.upper()
        event_logger.log(f"Mock Camera: Format set to {self.format}")
        return True

    def set_camera_param(self, prop: str, value: float) -> bool:
        with self.lock:
            if prop == "exposure":
                self.params["auto_exposure"] = 0
            self.params[prop] = value
        return True

    def get_camera_params(self) -> Dict[str, Any]:
        return {**self.params, "format": self.format}

    def get_camera_status(self) -> Dict[str, Any]:
        return {
            "connected": True,
            "width": self.width,
            "height": self.height,
            "camera_mode": "mock",
            "device_name": "Simulated Starfield Camera",
            "format": self.format,
            "sim_drift_angle": float(np.rad2deg(self.sim_drift_angle)),
            "sim_drift_speed": self.sim_drift_speed,
            "camera_angle": float(np.rad2deg(self.camera_angle))
        }

    def close(self):
        self.is_running = False
        try:
            self.nav.unsubscribe(self._on_nav_change)
        except Exception:
            pass
