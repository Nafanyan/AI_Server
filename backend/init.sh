#!/bin/bash

# Создаем виртуальное окружение 
echo "\033[32mСоздание виртуального окружения Python3\033[0m"
if ! command -v python3.12 &>/dev/null; then
    echo "Python 3.12.6 не установлен."
    exit 1
fi

python3.12 -m venv .venv
source .venv/bin/activate
echo "\033[32mАктивация виртуального окружения\033[0m"

command pip install --upgrade pip

# Устанавливаем зависимости из файла requirements.txt
echo "\033[32mУстановка общих зависимостей веб сервера\033[0m"
pip cache purge
pip install -r requirements.txt


# Устанавливаем зависимости для работы модели yolov5
echo "\033[32mУсановка модели yolov5 и её зависимостей\033[0m"
cd ./application/ai_models || exit 1

if [ -d "./yolov5" ]; then
    echo "\033[33mМодель yolov5 уже инициализированна.\033[0m"
else
    git clone https://github.com/ultralytics/yolov5.git || exit 1
    cd yolov5 || exit 1
    pip install -r requirements.txt || exit 1
    cd ./../
fi
echo "\033[32mУсановка модели yolov5 и её зависимостей завершена\033[0m"

echo "\033[32mВиртуальная среда создана и активирована!\033[0m"

# Запускаем сервер
echo "\033[32mНастройка приложения завершена!\033[0m"
cd ./../../
sh start.sh