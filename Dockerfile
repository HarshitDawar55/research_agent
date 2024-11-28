FROM python:3.9-slim

RUN python -m pip install pip --upgrade && pip3 install poetry

ENV POETRY_VIRTUALENVS_CREATE=false
COPY ./poetry.lock ./pyproject.toml /

RUN poetry install

COPY ./main.py ./schemas.py ./tools.py /

EXPOSE 80

CMD  ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]