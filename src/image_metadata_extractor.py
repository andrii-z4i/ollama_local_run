import os
from datetime import datetime
from typing import Optional, Dict, Any

from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass


class ImageMetadataExtractor:
    """Extracts EXIF metadata from image files using Pillow."""

    @staticmethod
    def extract(file_path: str) -> Dict[str, Any]:
        metadata = {
            "file_name": os.path.basename(file_path),
            "file_size": os.path.getsize(file_path),
            "date_taken": None,
            "gps_lat": None,
            "gps_lon": None,
            "camera_make": None,
            "camera_model": None,
            "image_width": None,
            "image_height": None,
        }

        try:
            with Image.open(file_path) as img:
                metadata["image_width"] = img.width
                metadata["image_height"] = img.height

                exif_data = img._getexif()
                if exif_data is None:
                    return metadata

                exif = {TAGS.get(k, k): v for k, v in exif_data.items()}

                # Date taken
                date_str = exif.get("DateTimeOriginal") or exif.get("DateTime")
                if date_str:
                    try:
                        dt = datetime.strptime(str(date_str), "%Y:%m:%d %H:%M:%S")
                        metadata["date_taken"] = dt.isoformat()
                    except (ValueError, TypeError):
                        pass

                # Camera info
                make = exif.get("Make")
                model = exif.get("Model")
                metadata["camera_make"] = str(make).strip() if make else None
                metadata["camera_model"] = str(model).strip() if model else None

                # GPS coordinates
                gps_info = exif.get("GPSInfo")
                if gps_info:
                    gps = {GPSTAGS.get(k, k): v for k, v in gps_info.items()}
                    lat = ImageMetadataExtractor._convert_gps_to_decimal(
                        gps.get("GPSLatitude"), gps.get("GPSLatitudeRef"))
                    lon = ImageMetadataExtractor._convert_gps_to_decimal(
                        gps.get("GPSLongitude"), gps.get("GPSLongitudeRef"))
                    if lat is not None and lon is not None:
                        metadata["gps_lat"] = lat
                        metadata["gps_lon"] = lon

        except Exception:
            pass

        return metadata

    @staticmethod
    def _convert_gps_to_decimal(coords, ref) -> Optional[float]:
        if not coords or not ref:
            return None
        try:
            degrees = float(coords[0])
            minutes = float(coords[1])
            seconds = float(coords[2])
            decimal = degrees + minutes / 60.0 + seconds / 3600.0
            if ref in ("S", "W"):
                decimal = -decimal
            return round(decimal, 6)
        except (TypeError, ValueError, IndexError):
            return None
