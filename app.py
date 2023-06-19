import os
import json
import yaml
from langchain.memory import ConversationBufferMemory
from langchain import LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
from rich.console import Console
from rich.text import Text
from rich.prompt import Prompt
from rich.markdown import Markdown
import pyfiglet


# helper functions to pretty print in the console
def rich_print(role: str, text: str):
    if role == 'system':
        console.print(Text("SYSTEM: ", style="bold yellow"),  Markdown(text))
    elif role == 'bot':
        console.print(Text("BOT: ", style="bold blue"),  Markdown(text))
    elif role == 'user':
        console.print(Text("User: ", style="bold green"),  Markdown(text))

def rich_input(role: str = 'user') -> str:
    if role == 'user':
        return Prompt.ask("[bold green]USER")

# read openai api key and model configuration
with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "secrets.json"), "r") as jsonfile:
    secrets = json.load(jsonfile)
    os.environ["OPENAI_API_KEY"] = secrets["openai_key"]

# raed chat settings
with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.yaml"), "r") as yamlfile:
    config = yaml.safe_load(yamlfile)

# read chat templates
with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "templates.yaml"), "r") as yamlfile:
    templates = yaml.safe_load(yamlfile)

if __name__ == "__main__":
    # init rich console session
    console = Console()
    
    # welcome screen
    welcome_message = pyfiglet.figlet_format("Command Line ChatGPT", font='rectangles')
    console.print(Text(welcome_message, style="bold yellow on black"))

    # prompt template picking loop
    while True:
        
        try:
            templates_list = "\n".join([f"* {t}" for t in templates.keys()])
            rich_print('system', 'Select one of the templates below or type \'quit\' to exit:')
            print(templates_list)
            template_choice = rich_input()
            
            if template_choice == "quit":
                rich_print('system', 'Goodbye!')
                break

            if template_choice not in templates:
                rich_print('system', 'Invalid template choice')
                continue 
        
        except KeyboardInterrupt:
            break

        # initialize conversation
        rich_print('system', f'You have chosen \"{template_choice}\" template. Start prompting or type \'quit\' to exit:')
        prompt = PromptTemplate(
            input_variables=["chat_history", "human_input"], template=templates[template_choice]
        )

        # use memory
        memory = ConversationBufferMemory(memory_key="chat_history")

        llm_chain = LLMChain(
            llm=ChatOpenAI(model_name=config["model_name"], temperature=config["temperature"], max_tokens=config["max_tokens"]),
            prompt=prompt,
            verbose=False,
            memory=memory,
        )

        # conversation loop
        while True:
            try:
                human_input = rich_input()
                if human_input == "quit":
                    rich_print('system', 'Quitting conversation...')
                    break
                else:
                    response = llm_chain.predict(human_input=human_input)
                    rich_print('bot', f'{response}')
            except KeyboardInterrupt:
                break