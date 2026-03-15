from src.image_embedding_manager import ImageEmbeddingManager
from src.image_files_processor import ImageFilesProcessor
from src.vision_analyzer import VisionAnalyzer
from src.reverse_geocoder import ReverseGeocoder
from src.arguments.analyze_photos import RunArguments


if __name__ == "__main__":
    run_args = RunArguments().parse()

    embedding = ImageEmbeddingManager(
        ollama_base_url=run_args.ollama_url,
        chroma_db_name=run_args.chroma_db_name,
        chroma_db_path=run_args.chroma_db_path,
        debug=run_args.verbose,
    )

    vision = VisionAnalyzer(
        model_name=run_args.vision_model,
        ollama_base_url=run_args.ollama_url,
    )

    geocoder = ReverseGeocoder(cache_path=run_args.geocode_cache)

    processor = ImageFilesProcessor(
        embedding=embedding,
        vision_analyzer=vision,
        geocoder=geocoder,
        directory=run_args.directory_to_analyze,
        extensions=run_args.extensions,
        exclude_subfolders=run_args.exclude_subdirectories,
        reload=run_args.reload,
        verbose=run_args.verbose,
    )

    processor.process_files()
