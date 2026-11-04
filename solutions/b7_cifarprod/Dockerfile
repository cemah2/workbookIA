# The prediction API of cifarprod, on CPU. Build: docker build -t cifar-api .  Run: docker run -p 8000:8000 cifar-api
# (an artifact folder, written by cifarprod train --out artifact, must be next to this file)
FROM python:3.13.18-slim-trixie

# settings, never secrets: unbuffered logs, no .pyc, no pip cache, the folder of the model
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_DIR=/app/artifact

WORKDIR /app

# the dependencies first: this layer stays in the cache while only the code changes; torch from the CPU index
COPY requirements-serve.txt .
RUN pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements-serve.txt

# then the code and the artifact of the model
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-deps .
COPY artifact/ artifact/

# a user without privileges runs the server
RUN groupadd --system app && useradd --system --no-log-init --gid app app
USER app

EXPOSE 8000
CMD ["python", "-m", "cifarprod.serve", "--host", "0.0.0.0", "--port", "8000"]
