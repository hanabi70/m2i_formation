FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt --no-cache-dir

EXPOSE 8080

CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8080","--workers","3"]