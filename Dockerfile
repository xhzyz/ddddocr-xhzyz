FROM python:3.9-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
WORKDIR /ddddocr
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7777
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7777", "--workers", "1"]