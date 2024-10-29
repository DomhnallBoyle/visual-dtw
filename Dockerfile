FROM ubuntu:18.04 as base
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    cron \
    build-essential \
    git-all \
    libpq-dev \
    python3 \
    python3-pip \
    run-one \
    vim

WORKDIR /opt/visual-dtw
COPY requirements.txt .
RUN pip3 install -r requirements.txt
ENV PYTHONPATH=app:$PYTHONPATH

FROM base as development
ENTRYPOINT scripts/background_jobs.sh && \
    env=dev scripts/cronjobs.sh && \
    watchmedo auto-restart --recursive --pattern="*.py" --directory="." python3 app/main/server.py -- --port=$PORT

FROM base as testing
ENTRYPOINT scripts/background_jobs.sh && \
    echo "UNIT TESTS" && coverage run --source=app/main,scripts -m pytest -vv -s --disable-pytest-warnings app/tests/unit && coverage report -m && \
    echo "INTEGRATION TESTS" && coverage run --source=app/main,scripts -m pytest -vv -s --disable-pytest-warnings app/tests/integration && coverage report -m && \
    echo "FLAKE8" && flake8 --docstring-convention=google --import-order-style=pycharm --per-file-ignores="*tests*.py:D *__init__.py:D" --ignore=D413,W503 --exclude="research" app && \
    echo "DONE"

FROM base as production
RUN apt-get install -y \
    uwsgi \
    uwsgi-plugin-python3
ENTRYPOINT scripts/background_jobs.sh && \
    env=prod scripts/cronjobs.sh && \
    uwsgi --ini config/wsgi/app.ini

FROM base as research
ENTRYPOINT scripts/background_jobs.sh && \
    scripts/cronjobs.sh && \
    python3 app/main/server.py --port=$PORT

FROM base as celery
ENTRYPOINT watchmedo auto-restart --recursive --pattern="*.py" --directory="." -- celery -A "main.utils.tasks" worker --loglevel=DEBUG
