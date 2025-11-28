FROM python:3.11-slim

# 1. Скопировать всё из текущей директории в /ml_service
COPY . /ml_service

# 2. Директория с моделями, доступная снаружи
# (потом примонтируешь томом или volume)
VOLUME ["/models"]

# 3. Рабочая директория
WORKDIR /ml_service

# 4. Установка зависимостей
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 5. Открыть 8000
EXPOSE 8000

# 6. Команда запуска сервера
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


# команда сборки образа
# docker build -t heart_ml_service:1 .

# Команда запуска контейнера.
# docker run -p 8001:8000 -v "$(pwd)/../models:/models" --name heart_ml_service_container heart_ml_service:1

# У меня проблемы с 8000 портом почему-то он занят и не хочет освобождаться.