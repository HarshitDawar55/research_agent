import logging
import os

from fastapi import FastAPI, HTTPException
from langchain.agents import AgentType, initialize_agent
from langchain_openai import ChatOpenAI
from schemas import AgentQuery, ListOfPapers
from tools import (
    call_openai,
    find_whether_a_research_paper_is_relevant_to_user_query,
    transform_user_query_for_essay,
    transform_user_query_for_literature_review,
    transform_user_query_for_research_gaps,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s -  %(message)s",
)

# Initializing FastAPI app
app = FastAPI(title="Research Agent", version="1.0.0")

openai_key = os.getenv("OPENAI_API_KEY")
all_tools = [
    call_openai,
    find_whether_a_research_paper_is_relevant_to_user_query,
    transform_user_query_for_literature_review,
    transform_user_query_for_essay,
    transform_user_query_for_research_gaps,
]


@app.get("/")
async def health_check():
    """
    Health check endpoint.
    """
    return {"status": "ok", "message": "Research Agent API is running."}


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    openai_api_key=openai_key,
    stop=["\nObservation"],
)


# Initializing Agent
agent = initialize_agent(
    tools=all_tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)


@app.post("/agent")
async def call_agent(input: AgentQuery):
    """
    Used to automate the tasks of essay generation & extracting research gaps on a particular topic.
    """
    try:
        logging.info("Staring the agent for other tasks!")
        result = agent.invoke({"input": input})
        return {"result": result}
    except Exception as e:
        logging.error(f"Exception Found in the main agent function: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/literature_review")
async def literature_review(input: ListOfPapers):
    """
    Used to automate the literature review task!
    """
    try:
        logging.info("Starting the agent for the literature review task!")
        result = agent.invoke({"input": input})
        return {"result": result}
    except Exception as e:
        logging.error(
            f"Exception Found in the literature review main function: {str(e)}"
        )
        raise HTTPException(status_code=500, detail=str(e))
