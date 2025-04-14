import streamlit as st
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize agents
web_agent = Agent(
    name="Web Agent",
    role="Search the web for information",
    model=OpenAIChat(id="gpt-4o"),
    tools=[DuckDuckGoTools()],
    instructions="Always include sources",
    show_tool_calls=True,
    markdown=True,
)

finance_agent = Agent(
    name="Finance Agent",
    role="Get financial data",
    model=OpenAIChat(id="gpt-4o"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, company_info=True)],
    instructions="Use tables to display data",
    show_tool_calls=True,
    markdown=True,
)

agent_team = Agent(
    team=[web_agent, finance_agent],
    model=OpenAIChat(id="gpt-4o"),
    instructions=["Always include sources", "Use tables to display data"],
    show_tool_calls=True,
    markdown=True,
)

# Streamlit UI
st.set_page_config(page_title="Agno Agents", layout="wide")
st.title("🔍 Agno Multi-Agent System")

st.sidebar.header("Choose an Agent")
selected_agent = st.sidebar.radio("Select an Agent:", ["Web Agent", "Finance Agent", "Agent Team"])

st.sidebar.header("User Query")
user_query = st.sidebar.text_area("Enter your query:")

if st.sidebar.button("Submit Query"):
    st.sidebar.success("Processing...")
    if selected_agent == "Web Agent":
        response = web_agent.run(user_query)
    elif selected_agent == "Finance Agent":
        response = finance_agent.run(user_query)
    else:
        response = agent_team.run(user_query)
    
    st.subheader("Response:")
    st.markdown(response.get_content_as_string())
