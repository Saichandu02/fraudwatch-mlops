FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY app ./app
COPY scripts ./scripts
ENV PYTHONPATH=/app/src
# Train/register before serving (artifacts land in the /app/artifacts volume):
#   python scripts/run_training.py
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
