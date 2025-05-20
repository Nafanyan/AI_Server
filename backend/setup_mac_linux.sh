#!/bin/bash

# Проверка наличия Python 3.12
if ! command -v python3.12 &>/dev/null; then
    echo "Ошибка: Python 3.12.6 не установлен."
    exit 1
fi

# Создание виртуального окружения
python3.12 -m venv .venv

# Активируем виртуальную среду
source .venv/bin/activate

# Обновление pip
pip install --upgrade pip

# Очистка кеша пакетов
pip cache purge

# Установка зависимостей
pip install -r requirements.txt

# Настройка YOLOv5
cd application/ai_models || { echo "Ошибка перехода в каталог ai_models"; exit 1; }

if [ -d "./yolov5" ]; then
    echo "YOLOv5 уже установлена."
else
    git clone https://github.com/ultralytics/yolov5.git || { echo "Ошибка клонирования репозитория YOLOv5"; exit 1; }
fi

cd yolov5 || { echo "Ошибка перехода в каталог yolov5"; exit 1; }
pip install -r requirements.txt || { echo "Ошибка установки зависимостей YOLOv5"; exit 1; }
cd .. || { echo "Ошибка возврата в родительский каталог"; exit 1; }

echo "Виртуальная среда настроена успешно!"
