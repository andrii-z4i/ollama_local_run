from embeding import Embedding
from files_processing_run_arguments import RunArguments
from files_processor import FilesProcessor


if __name__ == "__main__":
    run_args = RunArguments().parse()

    embedding = Embedding(
        chroma_db_name=run_args.chroma_db_name,
        chroma_db_path=run_args.chroma_db_path,
        debug=run_args.verbose
        )


    files_processor = FilesProcessor(
        embedding, 
        run_args.directory_to_analyze, 
        run_args.extensions, 
        run_args.exclude_subdirectories, 
        run_args.reload,
        run_args.verbose)

    files_processor.process_files()
    
    print("Files processing completed.")