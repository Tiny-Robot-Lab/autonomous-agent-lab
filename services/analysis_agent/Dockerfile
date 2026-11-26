FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src:/app/contracts:/app/infrastructure
WORKDIR /app
COPY services/analysis_agent/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN groupadd --system --gid 10001 app && useradd --system --uid 10001 --gid app --home-dir /nonexistent --shell /usr/sbin/nologin app
COPY scripts/container/redis_secret_entrypoint.py /usr/local/bin/redis-secret-entrypoint.py
COPY packages/contracts/src /app/contracts
COPY packages/infrastructure/src /app/infrastructure
COPY services/analysis_agent/src /app/src
ENTRYPOINT ["python", "/usr/local/bin/redis-secret-entrypoint.py"]
CMD ["python", "-m", "analysis_agent.main"]
