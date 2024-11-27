import logging
import os
from ctypes import Union

from fastapi import FastAPI, HTTPException
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain.tools.render import render_text_description
from langchain_openai import ChatOpenAI
from schemas import AgentQuery, ListOfPapers
from tools import (
    call_openai,
    find_relevant_research_papers,
    return_tool_by_name,
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
    find_relevant_research_papers,
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


template = """
        Answer the following questions as best you can. You have access to the following tools:

        {tools}

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        Begin!

        Question: {input}
        Thought: {agent_scratchpad}
        """

final_prompt = PromptTemplate.from_template(template=template).partial(
    tools=render_text_description(all_tools),
    tool_names=", ".join(t.name for t in all_tools),
)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    openai_api_key=openai_key,
    stop=["\nObservation"],
)

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"]),
    }
    | final_prompt
    | llm
    | ReActSingleInputOutputParser()
)

# @app.post("/literature_review_agent")


@app.post("/agent")
async def call_agent(input: AgentQuery):
    """
    Used to automate the tasks of essay generation & extracting research gaps on a particular topic.
    """
    try:
        logging.info("Staring the agent for other tasks!")
        intermedidate_steps = []
        agent_step = ""
        steps_taken = 1

        while not isinstance(agent_step, AgentFinish):
            logging.info(f"Step: {steps_taken}\n")
            agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
                {"input": input.query, "agent_scratchpad": intermedidate_steps}
            )
            logging.info(f"Agent Step is: {agent_step}")

            if isinstance(agent_step, AgentAction):
                tool_name_to_execute = agent_step.tool
                tool_to_use = return_tool_by_name(all_tools, tool_name_to_execute)
                tool_input = agent_step.tool_input

                response = tool_to_use.func(str(tool_input))
                intermedidate_steps.append(
                    (agent_step, str(response))
                )  # Providing agent with the reasning history & the output of each step as well

            steps_taken += 1

        return agent_step.return_values
    except Exception as e:
        logging.error(f"Exception Found in the main agent function: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
