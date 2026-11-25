FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src:/app/contracts:/app/policy:/app/infrastructure
WORKDIR /app
COPY services/research_agent/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN groupadd --system --gid 10001 app && useradd --system --uid 10001 --gid app --home-dir /nonexistent --shell /usr/sbin/nologin app
COPY scripts/container/redis_secret_entrypoint.py /usr/local/bin/redis-secret-entrypoint.py
COPY packages/contracts/src /app/contracts
COPY packages/policy/src /app/policy
COPY packages/infrastructure/src /app/infrastructure
COPY services/research_agent/src /app/src
ENTRYPOINT ["python", "/usr/local/bin/redis-secret-entrypoint.py"]
CMD ["python", "-m", "research_agent.main"]
