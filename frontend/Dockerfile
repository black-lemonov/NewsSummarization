FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir poetry

COPY pyproject.toml poetry.lock ./

RUN poetry config virtualenvs.create false && \
    poetry install --only main --no-interaction --no-ansi --no-root &&\
    pip install gunicorn

COPY . .

ENV BASE_API_URL=http://backend:8000

CMD ["poetry", "run", "gunicorn", "-b", "0.0.0.0:5000", "app:app"]