from LLM import OllamaLLM
from chat_run_arguments import RunArguments


if __name__ == "__main__":

    run_args = RunArguments().parse()

    ollama = OllamaLLM(
        chroma_db_name=run_args.chroma_db_name,
        chroma_db_path=run_args.chroma_db_path,
        system_prompt=run_args.system_prompt
    )
    
    # Enter into an interactive loop for conversation
    while True:
        prompt = input("You: ")
        if prompt.lower() in ["exit", "quit"]:
            print("Exiting the conversation.")
            break
        response = ollama.talk(prompt)
        print(f"Ollama: {response}")