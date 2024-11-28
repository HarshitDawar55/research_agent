import json
import logging
import os
from typing import Dict, List

import requests
from langchain.agents import tool
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from schemas import ListOfPapers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s -  %(message)s",
)

URL_OF_RELEVANCE_SCORE_MODEL = os.getenv("URL_OF_RELEVANCE_SCORE_MODEL")
openai_key = os.getenv("OPENAI_API_KEY")


@tool
def find_whether_a_research_paper_is_relevant_to_user_query(details):
    """
    This tool is only used as the first step in the literature review task by finding whether the given paper title
    and paper abstract are relevant to the user query or not. It will find out the relevant
    papers using this function.

    Input:
    details: A string containing key value pair for user query, title, and abstract in JSON format without wrapping explicitly using "```json".
    """
    try:
        # try:
        logging.info("Executing the Tool of finding the relevant research Papers")
        logging.info(f"Details received {details}, type: {type(details)}")
        details = json.loads(details.strip())

        logging.info(
            f" After conversion, Details received {details}, type: {type(details)}"
        )
        #     for i in range(len(details)):
        #         logging.warning(
        #             f"Details received: {details[i]}, type: {type(details[i])}"
        #         )
        #     logging.info(f"Query in Details received: {details['query']}")
        #     papers = []
        # except Exception as e:
        #     logging.error(f"Exception occurred while converting the details: {str(e)}")

        # for i in range(len(details)):
        try:
            text_input = json.dumps(
                {
                    "query": details["query"],
                    "title": details["title"],
                    "abstract": details["abstract"],
                }
            )

            response = requests.post(
                url=URL_OF_RELEVANCE_SCORE_MODEL,
                json=text_input,
                headers={"Content-Type": "application/json"},
            )
        except Exception as e:
            logging.error(
                f"Exception occurred in sending the details to the relevance model API: {str(e)}"
            )
        relevance = int(response.json()["relevance_score"])
        if relevance == 1:
            return f"Paper Title: {details['title']} and Paper Abstract: {details['abstract']} are relevant"
            # papers.append(
            #     {"title": details[i]["title"], "abstract": details[i]["abstract"]}
            # )
        # if papers:
        #     return f"Relevant papers list is: {papers}"
        # return "No relevant papers found, hence not literature review can be done"
        return f"Paper Title: {details['title']} and Paper Abstract: {details['abstract']} are not relevant"
    except Exception as e:
        logging.error(f"Exception Found in finding relevant papers tool: {str(e)}")
        return "Exception Found in finding relevant papers tool"


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
        return "Exception Found in calling openai tool"


@tool
def transform_user_query_for_literature_review(query):
    """
    This tool is used to transform user query for literature review report. This tool has to be used
    only when there is a query for literature review and generating an appropriate response for the
    same!
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
        return "Exception Found in transforming tool for literature review"


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
        logging.error(f"Exception Found in essay transformation tool: {str(e)}")
        return "Exception Found in essay transformation tool"


@tool
def transform_user_query_for_research_gaps(topic):
    """
    This tool is used to transform user query for identifying research gaps. If there is a requirement
    where the gaps needs to be identified in the existing work, or it feels that the work is incomplete,
    then this tool has to be used.!
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
        logging.error(f"Exception Found in finding research gaps tool: {str(e)}")
        return "Exception Found in finding research gaps tool"
