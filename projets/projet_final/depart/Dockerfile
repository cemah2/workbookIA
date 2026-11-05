# OPTIONAL (PF.11): the prediction API of the project, on CPU.
#   build: docker build -t projet-api .      run: docker run -p 8000:8000 projet-api
# The artifact folder (written by « projet train --out artifacts/model ») must exist next to this file.
FROM python:3.13.18-slim-trixie

# settings, never secrets: unbuffered logs, no .pyc, no pip cache, the folder of the model
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_DIR=/app/artifacts/model

WORKDIR /app

# the dependencies first (this layer stays in the cache while only the code changes); torch from the CPU index
COPY pyproject.toml README.md ./
RUN pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install numpy pandas pyyaml scikit-learn fastapi uvicorn pydantic

# then the code and the artifact of the model
COPY src/ src/
RUN pip install --no-deps .
COPY artifacts/model/ artifacts/model/

# a user without privileges runs the server
RUN groupadd --system app && useradd --system --no-log-init --gid app app
USER app

EXPOSE 8000
# exec: the server replaces the shell and receives the SIGTERM of « docker stop »
CMD ["sh", "-c", "exec python -m projet.serve --model \"$MODEL_DIR\" --host 0.0.0.0 --port 8000"]
