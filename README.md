# Research_Agent
This project is a Research Agent that can perform various tasks like (by using custom defined tools):
* Perform a Literature Review based on the Research Papers
* Write an essay on a particular topic
* Extract Research Gaps on a particular topic

# Instructions To Run

## Pre-Requisites
* OpenAI Key
* Relevance Scording Model running at some IP Address (its image present at [Docker Hub](https://hub.docker.com/r/harshitdawar/research_paper_relevance/tags))
* Docker or any other equivalent tool

## Steps
* Pull the docker image of this Research Agent from Docker Hub using the command:
``` bash
docker pull harshitdawar/research_agent:latest
```
* Start the container by running the following command:
```bash
docker run -dit -p <Any available port on your system>:80 -e OPENAI_API_KEY="<openai api key>" -e URL_OF_RELEVANCE_SCORE_MODEL=<complete url where the model is running (with http/https)> harshitdawar/research_agent:latest
```

# API Endpoints
* **GET** request at **"/"**: For healthcheck
* **POST** request at **"/agent"**: For Essay generation or extracting research gaps task
* **POST** request at **"/literature_review"**: For Literature Review Task