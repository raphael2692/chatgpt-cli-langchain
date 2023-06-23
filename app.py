import os
import json

import yaml
import pyfiglet

from langchain.memory import ConversationBufferMemory
from langchain import LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# read openai api key and model configuration
with open(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "secrets.json"), "r"
) as jsonfile:
    secrets = json.load(jsonfile)
    os.environ["OPENAI_API_KEY"] = secrets["openai_key"]

# raed chat settings
with open(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.yaml"), "r"
) as yamlfile:
    config = yaml.safe_load(yamlfile)

# read chat templates
with open(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "templates.yaml"), "r"
) as yamlfile:
    templates = yaml.safe_load(yamlfile)

if __name__ == "__main__":
    welcome_message = pyfiglet.figlet_format("Command Line ChatGPT", font="rectangles")
    print(welcome_message)

    while True:
        try:
            templates_list = "\n".join([f"* {t}" for t in templates.keys()])
            print("system: Select one of the templates below or type 'quit' to exit:")
            print(f"\n{templates_list}\n")
            template_choice = input("user: ")

            if template_choice == "quit":
                print("system: Goodbye!")
                break

            if template_choice not in templates:
                print("system: Invalid template choice")
                continue

        except KeyboardInterrupt:
            break

        print(
            f"system You have chosen \"{template_choice}\" template. Start prompting or type 'quit' to exit:"
        )

        llm_chain = LLMChain(
            llm=ChatOpenAI(
                model_name=config["model_name"],
                temperature=config["temperature"],
                max_tokens=config["max_tokens"],
                streaming=True,
                callbacks=[StreamingStdOutCallbackHandler()],
            ),
            prompt=PromptTemplate(
                input_variables=["chat_history", "human_input"],
                template=templates[template_choice],
            ),
            verbose=False,
            memory=ConversationBufferMemory(memory_key="chat_history"),
        )

        # conversation loop
        while True:
            try:
                print("")
                human_input = input("user: ")
                if human_input == "quit":
                    print("system: Quitting conversation...")
                    break
                else:
                    print("bot: ")
                    llm_chain.predict(human_input=human_input)
            except KeyboardInterrupt:
                break
