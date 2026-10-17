FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY checkpoints/serving/ checkpoints/serving/

ENV MCP_HTTP_PORT=8000
EXPOSE 8000

CMD ["python", "-m", "src.serving.mcp_server"]
