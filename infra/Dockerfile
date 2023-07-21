FROM python:3.11-slim-bookworm as builder

#RUN apt-get update -qq
#RUN apt-get install python3.10 python3-pip -y --no-install-recommends && rm -rf /var/lib/apt/lists_/*
RUN apt-get update --fix-missing && apt-get install -y --fix-missing build-essential

RUN mkdir /install
WORKDIR /install

COPY infra/requirements.txt .

RUN pip install --no-cache-dir --upgrade --prefix="/install" -r requirements.txt

FROM python:3.11-slim-bookworm as final

RUN mkdir /chroma

ENV CHROMADB_PERSIST_DIRECTORY /chroma

COPY --from=builder /install /usr/local
#COPY ./bin/docker_entrypoint.sh /docker_entrypoint.sh

COPY ./src/api.py .
COPY ./src/api_tools.py .

ENV PYTHONUNBUFFERED 1

CMD ["python","./api.py"]




#FROM python:3.10-slim-bullseye
#
#RUN apt-get update --fix-missing && apt-get install -y --fix-missing build-essential
#
#WORKDIR /code
#
#COPY infra/requirements.txt .
#
#RUN python -m pip install --upgrade pip
#
#RUN pip install pysqlite3
#
#RUN pip install -r requirements.txt
#
#COPY ./src/api.py .
#COPY ./src/api_tools.py .
#
#RUN mkdir /chroma
#
#ENV PYTHONUNBUFFERED 1
#
#CMD ["python","./api.py"]
