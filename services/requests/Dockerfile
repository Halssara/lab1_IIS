FROM python:3.11-slim

WORKDIR /requests_service

# Устанавливаем только нужные зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Копируем код
COPY random_request.py .

# По умолчанию запускаем цикл отправки запросов
CMD ["python", "random_request.py"]

# Команда сборки образа
# docker build -t requests_service:1 .

# Команда запуска контейнера
# docker run --name requests_service_container --rm requests_service:1
