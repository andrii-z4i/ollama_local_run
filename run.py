from LLM import OllamaLLM
from embeding import Embedding, EmbeddingProcessMode
import asyncio
from run_arguments import RunArguments
from files_processor import FilesProcessor


if __name__ == "__main__":

    run_args = RunArguments().parse()

    embedding = Embedding(
        chroma_db_name=run_args.chroma_db_name,
        chroma_db_path=run_args.chroma_db_path,
        debug=run_args.verbose
        )
    ollama = OllamaLLM(embedding, system_prompt=run_args.system_prompt)
    
    if all([run_args.force_reload, run_args.soft_reload]):
        raise ValueError("You can only specify one of the reload modes.")
    
    if run_args.force_reload:
        process_mode = EmbeddingProcessMode.FORCE_RELOAD
    elif run_args.soft_reload:
        process_mode = EmbeddingProcessMode.SOFT_RELOAD

    files_processor = FilesProcessor(embedding, run_args.directory_to_analyze, run_args.extensions, run_args.exclude_subdirectories, process_mode)

    asyncio.run(files_processor.process_files())
    
    # Enter into an interactive loop for conversation
    while True:
        prompt = input("You: ")
        if prompt.lower() in ["exit", "quit"]:
            print("Exiting the conversation.")
            break
        response = ollama.talk(prompt)
        print(f"Ollama: {response}")