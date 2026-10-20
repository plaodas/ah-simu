FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src src
COPY app app
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1
EXPOSE 5000 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
