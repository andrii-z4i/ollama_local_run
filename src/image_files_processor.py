import hashlib
import os
from typing import Iterable, List, Tuple, Dict, Any

from src.image_embedding_manager import ImageEmbeddingManager
from src.image_metadata_extractor import ImageMetadataExtractor
from src.vision_analyzer import VisionAnalyzer
from src.reverse_geocoder import ReverseGeocoder


class ImageFilesProcessor:
    """Walks a directory of images, analyzes each with a vision model, and stores results in ChromaDB."""

    def __init__(self,
                 embedding: ImageEmbeddingManager,
                 vision_analyzer: VisionAnalyzer,
                 geocoder: ReverseGeocoder,
                 directory: str,
                 extensions: List[str],
                 exclude_subfolders: List[str],
                 reload: bool = False,
                 verbose: bool = False) -> None:
        self.embedding = embedding
        self.vision_analyzer = vision_analyzer
        self.geocoder = geocoder
        self.directory = directory
        self.extensions = [ext.lower().lstrip('.') for ext in extensions]
        self.exclude_subfolders = exclude_subfolders
        self.reload = reload
        self.verbose = verbose

    def _calculate_checksum(self, file_path: str) -> str:
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _enumerate_files(self, directory: str) -> Iterable[Tuple[str, str]]:
        try:
            entries = sorted(os.listdir(directory))
        except PermissionError:
            if self.verbose:
                print(f"  Skipping (permission denied): {directory}")
            return

        for filename in entries:
            if filename in self.exclude_subfolders or filename.startswith('.'):
                continue
            full_path = os.path.join(directory, filename)
            ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
            if ext in self.extensions:
                yield (full_path, self._calculate_checksum(full_path))
            elif os.path.isdir(full_path):
                yield from self._enumerate_files(full_path)

    def _build_metadata(self, file_path: str, checksum: str,
                        exif: Dict[str, Any], location: str) -> Dict[str, Any]:
        metadata = {
            "source": file_path,
            "checksum": checksum,
            "file_name": exif.get("file_name", os.path.basename(file_path)),
            "file_size": exif.get("file_size", 0),
        }
        optional_fields = {
            "date_taken": exif.get("date_taken"),
            "gps_lat": exif.get("gps_lat"),
            "gps_lon": exif.get("gps_lon"),
            "camera_make": exif.get("camera_make"),
            "camera_model": exif.get("camera_model"),
            "image_width": exif.get("image_width"),
            "image_height": exif.get("image_height"),
        }
        if location:
            optional_fields["location"] = location

        for key, value in optional_fields.items():
            if value is not None:
                metadata[key] = value

        return metadata

    def process_files(self):
        if self.verbose:
            print(f"Scanning directory: {self.directory}")
            print(f"Extensions: {self.extensions}")
            print(f"Reload mode: {self.reload}")

        files = list(self._enumerate_files(self.directory))
        total = len(files)

        if self.verbose:
            print(f"Found {total} image files to process")

        processed = 0
        skipped = 0
        failed = 0

        for idx, (file_path, checksum) in enumerate(files, 1):
            if self.verbose:
                print(f"\n[{idx}/{total}] {os.path.basename(file_path)}")

            if not self.embedding.should_reprocess(file_path, checksum, self.reload):
                skipped += 1
                if self.verbose:
                    print(f"  Skipped (already processed)")
                continue

            self.embedding.delete_by_source(file_path)

            # Step 1: Extract EXIF metadata
            if self.verbose:
                print(f"  Extracting metadata...")
            exif = ImageMetadataExtractor.extract(file_path)

            # Step 2: Reverse geocode if GPS available
            location = None
            if exif.get("gps_lat") and exif.get("gps_lon"):
                if self.verbose:
                    print(f"  Reverse geocoding GPS ({exif['gps_lat']}, {exif['gps_lon']})...")
                location = self.geocoder.geocode(exif["gps_lat"], exif["gps_lon"])
                if self.verbose and location:
                    print(f"  Location: {location}")

            # Step 3: Analyze with vision model
            if self.verbose:
                print(f"  Analyzing image with vision model...")
            description = self.vision_analyzer.analyze(file_path)

            if not description:
                failed += 1
                print(f"  Failed: Could not get description for {file_path}")
                continue

            if self.verbose:
                preview = description[:150] + "..." if len(description) > 150 else description
                print(f"  Description: {preview}")

            # Step 4: Build metadata and store
            metadata = self._build_metadata(file_path, checksum, exif, location)

            # Prepend date and location context for better retrieval
            context_parts = []
            if exif.get("date_taken"):
                context_parts.append(f"Photo taken on {exif['date_taken']}.")
            if location:
                context_parts.append(f"Location: {location}.")
            enriched_description = " ".join(context_parts + [description])

            self.embedding.store_image_description(enriched_description, metadata)
            processed += 1

            if self.verbose:
                print(f"  ✓ Stored successfully")

        # Clean up files that were removed from disk
        stored_files = self.embedding.get_list_of_stored_files()
        current_files = {f[0] for f in files}
        removed = 0
        for stored in stored_files:
            if stored not in current_files:
                self.embedding.delete_by_source(stored)
                removed += 1
                if self.verbose:
                    print(f"  Removed from DB (file deleted): {stored}")

        print(f"\n{'='*50}")
        print(f"Processing complete:")
        print(f"  Total files found: {total}")
        print(f"  Processed:         {processed}")
        print(f"  Skipped:           {skipped}")
        print(f"  Failed:            {failed}")
        print(f"  Removed from DB:   {removed}")
        print(f"{'='*50}")
