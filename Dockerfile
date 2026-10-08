# --- frontend build ---------------------------------------------------------
FROM node:22-alpine AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --ignore-scripts
COPY frontend/ ./
RUN npm run build

# --- runtime ----------------------------------------------------------------
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ARTIFACTS_DIR=/data \
    FRONTEND_DIST=/app/frontend/dist
WORKDIR /app/backend
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/creature ./creature
COPY --from=web /web/dist /app/frontend/dist
RUN useradd --uid 1000 --create-home creature && mkdir /data && chown creature /data
USER creature
EXPOSE 8080
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=5 \
  CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8080/api/health', timeout=2)"
CMD ["uvicorn", "creature.main:app", "--host", "0.0.0.0", "--port", "8080"]
