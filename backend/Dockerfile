FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir poetry

COPY pyproject.toml poetry.lock ./

RUN poetry config virtualenvs.create false && \
    poetry install --only main --no-interaction --no-ansi --no-root

RUN apt-get update && apt-get install -y wget unzip

RUN python -c "import nltk; nltk.download('punkt', download_dir='/usr/local/nltk_data')"
RUN python -c "import nltk; nltk.download('stopwords', download_dir='/usr/local/nltk_data')"
RUN python -c "import nltk; nltk.download('punkt_tab', download_dir='/usr/local/nltk_data')"

ENV NLTK_DATA=/usr/local/nltk_data

COPY . .

CMD ["poetry", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]