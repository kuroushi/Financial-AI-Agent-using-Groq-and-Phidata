from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
import os
load_dotenv()

Groq_api_key = Groq(api_key=os.environ.get("GROQ_API_KEY"))


# Web search agent
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


multi_ai_agent=Agent(
    team=[web_serach_agent,financial_agent],
    model=Groq(id="meta-llama/llama-4-maverick-17b-128e-instruct"),
    instructions=["Always include sources ","Use table to display the data"],
    show_tool_calls=True,
    markdown=True,
)

multi_ai_agent.print_response("Summarize analyst recommandation and share the latest stock prices for Apple", stream=True)