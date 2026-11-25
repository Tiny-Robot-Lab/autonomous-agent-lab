FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src:/app/contracts
WORKDIR /app
COPY services/orchestrator/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN groupadd --system --gid 10001 app && useradd --system --uid 10001 --gid app --home-dir /nonexistent --shell /usr/sbin/nologin app
COPY scripts/container/redis_secret_entrypoint.py /usr/local/bin/redis-secret-entrypoint.py
COPY packages/contracts/src /app/contracts
COPY services/orchestrator/src /app/src
EXPOSE 8080
ENTRYPOINT ["python", "/usr/local/bin/redis-secret-entrypoint.py"]
CMD ["uvicorn", "orchestrator.presentation.api:app", "--host", "0.0.0.0", "--port", "8080"]
