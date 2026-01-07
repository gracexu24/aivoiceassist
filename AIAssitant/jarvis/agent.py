from google.adk.agents import Agent
from google.adk.tools import google_search

root_agent = Agent(
    name="jarvis",
    model="gemini-2.5-flash",
    instruction= """You are a helpful assistant that can answer questions with the help of the internet.
     ## Don't provide any other information than the answer. 
      """, 
      tools=[
        google_search,
        #try adding messaging tools later (need authentication for that)
        ]
    
)