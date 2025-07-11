from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
import os

import phi.api
import phi
from phi.playground import Playground, serve_playground_app

load_dotenv()

phi.api=os.getenv("PHI_API_KEY")

web_serach_agent=Agent(
    name="Web-search_agent",
    role="Search the web for information",
    model=Groq(id="meta-llama/llama-4-maverick-17b-128e-instruct"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True,   
) 

# Financial Agent

financial_agent=Agent(
    name="Financial_agent",
    model=Groq(id="meta-llama/llama-4-maverick-17b-128e-instruct"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True, company_news=True)],
    instructions=["Use tables to display the data"],
    show_tool_calls=True,
    markdown=True,
)

app=Playground(agents=[financial_agent,web_serach_agent]).get_app()

if __name__=="__main__":
    serve_playground_app("playground:app",reload=True)