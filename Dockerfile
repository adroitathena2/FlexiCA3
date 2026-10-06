# Paytriq API — deployable image.
#
# Build:   docker build -t paytriq .
# Run:     docker run -p 18780:18780 paytriq
#          (or set PORT / GEMINI_API_KEY / CLEF_* via `docker run -e ...` —
#          see .env.example; no secrets are baked into this image)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install deps first for layer caching, then the project itself.
# NOTE: `pip install -e .` must run AFTER the source tree is present --
# installing editable from only pyproject.toml/README records an empty
# package mapping. Deps stay cached in their own layer; the editable
# install runs after the full COPY.
COPY requirements.txt pyproject.toml README.md LICENSE ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir -e .

EXPOSE 18780

# Image defaults; override with `docker run -e ...` or the platform's env config.
ENV GEMINI_MODEL=gemini-2.5-flash \
    CLEF_TIMEOUT_S=5

# Provenance: the real commit SHA is injected at build time, never invented at
# runtime. Build with:
#   docker build --build-arg GIT_COMMIT=$(git rev-parse HEAD) \
#                --build-arg CODE_SHA256=$(git rev-parse HEAD) -t paytriq .
# CI (.github/workflows/ci.yml) and Render (render.yaml) pass the same args /
# env so golden_summary.json and trace provenance record the build's commit.
# A zero-commit checkout records "uncommitted-working-tree" explicitly.
ARG GIT_COMMIT=uncommitted-working-tree
ARG CODE_SHA256=uncommitted-working-tree
ENV GIT_COMMIT=${GIT_COMMIT} \
    CODE_SHA256=${CODE_SHA256}

# Liveness probe against the dependency-free endpoint (no subsystems touched,
# so a missing model never restarts the container). No curl in slim images,
# so plain stdlib urllib.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import os,sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:'+os.getenv('PORT','18780')+'/healthz',timeout=5).getcode()==200 else 1)"

# Exec (JSON-array) form so the container's PID 1 is the server and receives
# SIGTERM directly. `sh -c` remains for $PORT expansion (Render/Fly/Heroku
# inject PORT at runtime; plain exec form cannot expand env vars), and the
# inner `exec` replaces the shell with uvicorn so no middleman stays PID 1.
CMD ["sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-18780}"]
