from embeding import Embedding, EmbeddingProcessMode
import asyncio
from files_processing_run_arguments import RunArguments
from files_processor import FilesProcessor


if __name__ == "__main__":

    args = [
        '--directory-to-analyze', 'C:\\Users\\andriikozin\\prj\\ms\\tct-doc\\TCS\\OCE\\DNS\\Monitoring\\Thousand Eyes',
        '--extensions', 'md',
        '--verbose'
    ]

    run_args = RunArguments().parse(args)

    embedding = Embedding(
        chroma_db_name=run_args.chroma_db_name,
        chroma_db_path=run_args.chroma_db_path,
        debug=run_args.verbose
        )
    
    if all([run_args.force_reload, run_args.soft_reload]):
        raise ValueError("You can only specify one of the reload modes.")
    
    if run_args.force_reload:
        process_mode = EmbeddingProcessMode.FORCE_RELOAD
    elif run_args.soft_reload:
        process_mode = EmbeddingProcessMode.SOFT_RELOAD

    files_processor = FilesProcessor(
        embedding, 
        run_args.directory_to_analyze, 
        run_args.extensions, 
        run_args.exclude_subdirectories, 
        process_mode)

    files_processor.process_files()
    
    print("Files processing completed.")