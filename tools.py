import json
import logging
import os

import requests
from langchain.agents import tool
from langchain.prompts import PromptTemplate
from langchain.tools.base import Tool
from langchain_openai import ChatOpenAI
from schemas import ListOfPapers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s -  %(message)s",
)

URL_OF_RELEVANCE_SCORE_MODEL = os.getenv("URL_OF_RELEVANCE_SCORE_MODEL")
openai_key = os.getenv("OPENAI_API_KEY")


@tool
def find_relevant_research_papers(details: ListOfPapers) -> int:
    """
    This tool is used to find whether the given paper title and paper abstract are relevant to the user query or not!
    """
    try:
        logging.info("Executing the Tool of finding the relevant research Papers")
        papers = []

        for i in range(len(details.data)):
            text_input = json.loads(
                f'{"query" : {details.data[i].query}, "title" : {details.data[i].title}, "abstract" : {details.data[i].abstract}}'
            )

            response = requests.post(url=URL_OF_RELEVANCE_SCORE_MODEL, json=text_input)
            relevance = int(response.json()["relevance_score"])
            if relevance == 1:
                papers.append({f"{details.data[i].title} {details.data[i].abstract}"})
        return papers
    except Exception as e:
        logging.error(f"Exception Found in finding relevant papers tool: {str(e)}")
        return 400


@tool
def call_openai(query):
    """
    This tool is used to invoke Large Language Model to generate a response for a given query. When
    no other tool is looking useful to perform a particular task, then this tool can be used
    to solve the query. Generally this tool is used as a last step in the complete execution
    to solve a particular problem!
    """
    try:
        logging.info("Using the Tool to generate response from OpenAI's GPT Model")
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
            openai_api_key=openai_key,
        )

        response = llm.invoke(query)
        return response
    except Exception as e:
        logging.error(f"Exception Found in calling openai tool: {str(e)}")
        return 400


@tool
def transform_user_query_for_literature_review(query):
    """
    This tool is used to transform user query for literature review report!
    """
    try:
        logging.info("Using the tool to transform user query for Literature review!")
        literature_review_prompt = PromptTemplate(
            input_variables=["query"],
            template=f"As an expert researcher and expert literature reviewer, write a literature review on the topic: {query}",
        )
        return literature_review_prompt
    except Exception as e:
        logging.error(
            f"Exception Found in transforming tool for literature review: {str(e)}"
        )
        return 400


@tool
def transform_user_query_for_essay(topic):
    """
    This tool is used to transform user query for writing an essay!
    """
    try:
        logging.info("Using the tool to transform user query for writing Essay!")
        essay_prompt = PromptTemplate(
            input_variables=["topic"],
            template=f"As a language expert and expert in essay writing, write an essay on the topic: {topic}",
        )
        return essay_prompt
    except Exception as e:
        logging.error(f"Exception Found: {str(e)}")
        return 400


@tool
def transform_user_query_for_research_gaps(topic):
    """
    This tool is used to transform user query for identifying research gaps!
    """
    try:
        logging.info(
            "Using the tool to transform user query to identigy the Research Gaps!"
        )
        research_gaps_prompt = PromptTemplate(
            input_variables=["topic"],
            template=f"As an expert researcher, identify research gaps on the topic: {topic}",
        )

        return research_gaps_prompt
    except Exception as e:
        logging.error(f"Exception Found: {str(e)}")
        return 400
