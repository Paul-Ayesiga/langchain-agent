import os
from dotenv import load_dotenv
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_mistralai.chat_models import ChatMistralAI
from langchain.tools import tool
from langchain.prompts import ChatPromptTemplate
from langchain_community.utilities import SerpAPIWrapper # Keep this import

load_dotenv()

mistralai_api_key = os.getenv('MISTRAL_API_KEY')
serpapi_key = os.getenv('SERPAPI_API_KEY') # Get SerpAPI key early

# --- Initial Checks for API Keys ---
if not mistralai_api_key:
    print("Error: MISTRAL_API_KEY not found in environment variables. Please set it.")
    exit() # Exit if critical key is missing

if not serpapi_key:
    print("Error: SERPAPI_API_KEY not found in environment variables. Please set it.")
    print("You can get one from: https://serpapi.com/")
    exit() # Exit if critical key is missing

llm = ChatMistralAI(
    model = "mistral-large",
    api_key = mistralai_api_key
)

@tool
def math(a:int, b:int) -> int:
    """Add two numbers together and return their sum.
    Args:
        a: First number to add
        b: Second number to add
    Returns:
        int: The sum of a and b
    """
    return a + b

# Initialize SerpAPIWrapper with the key
try:
    serpapi = SerpAPIWrapper(serpapi_api_key=serpapi_key)
    # # Test SerpAPI connectivity by making a simple direct call
    # print("Attempting a direct SerpAPI test...")
    # test_search_result = serpapi.run("current date")
    # print(f"Direct SerpAPI test successful! Result snippet: {test_search_result[:50]}...")
except Exception as e:
    print(f"Error initializing or testing SerpAPIWrapper: {e}")
    print("Please check your SERPAPI_API_KEY and internet connection.")
    exit() # Exit if SerpAPI isn't working

@tool
def searchInternet(query:str) -> str:
    """Search the internet for latest information.
    Args:
        query: The search query
    Returns:
        str: The search results
    """
    return serpapi.run(query)

tools = [math, searchInternet]

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. **For any question requiring current, real-time, or highly specific external information (like current time, weather, latest news, recent stock prices, etc.), you MUST use the 'searchInternet' tool.** Only if a question can be answered accurately and fully from your internal knowledge, you may answer directly. Always provide the most up-to-date information possible by leveraging your tools."
        ),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ]
)

agent = create_tool_calling_agent(
    llm,
    tools,
    prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

def main():
    while True:
        try:
            question = input("Enter your question (or 'exit' to quit): ")
            if question.lower() == 'exit':
                break
            result = agent_executor.invoke({"input": question})
            print(result)
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"An error occurred during agent execution: {str(e)}")

if __name__ == "__main__":
    main()
