# Используем базовый образ Python
FROM python:3.9

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файлы проекта
COPY gradient_descent.py /app/gradient_descent.py

# Устанавливаем зависимости (если есть)
RUN pip install numpy

# Команда по умолчанию
CMD ["python", "gradient_descent.py"]

