FROM python:3.10-slim-bookworm

RUN apt-get update --fix-missing && apt-get install -y --fix-missing build-essential

WORKDIR /code

COPY infra/requirements.txt .

RUN python -m pip install --upgrade pip

RUN pip install -r requirements.txt

COPY ./src/api.py .
COPY ./src/api_tools.py .

RUN mkdir /chroma

ENV PYTHONUNBUFFERED 1

CMD ["python","./api.py"]
