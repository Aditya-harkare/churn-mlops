# ─────────────────────────────────────────
# Base Image
# ─────────────────────────────────────────
# Every Dockerfile starts with a base image — a pre-built starting point.
# We use python:3.11-slim, not python:3.13, for two reasons:
#   1. "slim" is a minimal Linux image with only Python installed
#      (no unnecessary tools) — keeps image size small
#   2. 3.11 is more stable and widely tested with our dependencies
#      than 3.13 which is very new. In production, stability > cutting edge.
#
# Think of this as choosing which "blank canvas" to start with.

FROM python:3.12-slim

# ─────────────────────────────────────────
# Working Directory
# ─────────────────────────────────────────
# WORKDIR sets the directory all subsequent commands run inside.
# /app is a convention — it's where application code lives in containers.
# If /app doesn't exist, Docker creates it automatically.
# This is equivalent to doing "mkdir /app && cd /app" on the container.

WORKDIR /app

# ─────────────────────────────────────────
# Install Dependencies
# ─────────────────────────────────────────
# We copy requirements_api.txt BEFORE copying our code.
# This is a critical Docker optimization called "layer caching":
#
# Docker builds images layer by layer. Each instruction is one layer.
# If a layer hasn't changed, Docker reuses the cached version
# instead of rebuilding it — making subsequent builds much faster.
#
# Dependencies change rarely. Code changes frequently.
# By copying requirements first and installing them as a separate layer,
# code changes don't trigger a full reinstall of all packages.
#
# If we copied ALL files first, then installed dependencies,
# every single code change would force Docker to reinstall
# all packages from scratch — very slow.

COPY requirements_api.txt .

# RUN executes a command inside the container during build.
# --no-cache-dir: don't store pip's download cache inside the image
#                 (reduces image size, since we don't need to re-download)
# --upgrade pip: ensures we have the latest pip before installing

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements_api.txt

# ─────────────────────────────────────────
# Copy Application Files
# ─────────────────────────────────────────
# Now we copy our actual code and model.
# We copy these AFTER installing dependencies (layer caching benefit above).
#
# We only copy what the API actually needs at runtime:
#   src/serving/api.py  — the FastAPI application
#   model_store/        — the exported model file
#
# We do NOT copy: venv/, data/, mlflow.db, mlruns/, monitoring/, notebooks/
# The container only needs what it runs, nothing else.

COPY src/serving/api.py .
COPY model_store/ ./model_store/

# ─────────────────────────────────────────
# Environment Variable
# ─────────────────────────────────────────
# Sets the MODEL_PATH environment variable inside the container.
# Our api.py reads this with os.getenv("MODEL_PATH", "model_store/churn_model.pkl")
# Since WORKDIR is /app, the full path becomes /app/model_store/churn_model.pkl
# This matches where we copied the model above.

ENV MODEL_PATH=model_store/churn_model.pkl

# ─────────────────────────────────────────
# Expose Port
# ─────────────────────────────────────────
# EXPOSE tells Docker "this container listens on port 8000".
# This is documentation — it doesn't actually open the port.
# The actual port mapping happens when you run the container
# with "docker run -p 8000:8000"
# Think of it as labeling which door the container uses.

EXPOSE 8000

# ─────────────────────────────────────────
# Start Command
# ─────────────────────────────────────────
# CMD is the command that runs when the container starts.
# We use the list format ["command", "arg1", "arg2"] — called "exec form".
# This is preferred over string form because it runs the process directly
# without a shell wrapper, making signals (like Ctrl+C) work correctly.
#
# python api.py starts our FastAPI server which runs uvicorn internally.
# Since WORKDIR is /app and we copied api.py there, this works directly.

CMD ["python", "api.py"]