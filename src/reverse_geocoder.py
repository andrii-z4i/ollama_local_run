import json
import os
import time
from typing import Optional


class ReverseGeocoder:
    """Converts GPS coordinates to place names using Nominatim with local caching."""

    def __init__(self, cache_path: str = ".geocode_cache.json"):
        self._cache_path = cache_path
        self._cache = self._load_cache()
        self._last_request_time = 0.0
        self._geolocator = None

    def _get_geolocator(self):
        if self._geolocator is None:
            from geopy.geocoders import Nominatim
            self._geolocator = Nominatim(user_agent="personal-photo-memory")
        return self._geolocator

    def _load_cache(self) -> dict:
        if os.path.exists(self._cache_path):
            try:
                with open(self._cache_path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return {}

    def _save_cache(self):
        try:
            with open(self._cache_path, "w") as f:
                json.dump(self._cache, f, indent=2)
        except IOError:
            pass

    def _make_cache_key(self, lat: float, lon: float) -> str:
        # Round to ~1km precision so nearby photos share cache entries
        return f"{round(lat, 2)},{round(lon, 2)}"

    def geocode(self, lat: float, lon: float) -> Optional[str]:
        if lat is None or lon is None:
            return None

        cache_key = self._make_cache_key(lat, lon)
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            # Nominatim requires max 1 request per second
            elapsed = time.time() - self._last_request_time
            if elapsed < 1.1:
                time.sleep(1.1 - elapsed)

            geolocator = self._get_geolocator()
            location = geolocator.reverse(f"{lat}, {lon}", language="en", timeout=10)
            self._last_request_time = time.time()

            if location and location.address:
                self._cache[cache_key] = location.address
                self._save_cache()
                return location.address
        except Exception:
            pass

        return None
