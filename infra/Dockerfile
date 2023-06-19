FROM python:3.9.17-slim

WORKDIR /code

COPY infra/requirements.txt .

RUN python -m pip install --upgrade pip

RUN pip install -r requirements.txt

COPY ./.env .
COPY ./api.py .
COPY ./api_tools.py .

ENV PYTHONUNBUFFERED 1

CMD ["python","./api.py"]
