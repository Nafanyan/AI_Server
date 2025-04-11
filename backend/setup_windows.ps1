# Проверка наличия Python 3.12
if (-not (Get-Command 'python' -ErrorAction SilentlyContinue)) {
    Write-Host "Ошибка: Python 3.12.6 не установлен."
    Exit 1
}

# Создание виртуального окружения
python -m venv .venv

# Активация виртуальной среды
& .\.venv\Scripts\Activate.ps1

# Обновление pip
pip cache purge
python -m ensurepip --default-pip
python -m pip install --upgrade pip

# Установка зависимостей
pip install -r requirements.txt

# Настройка YOLOv5
Set-Location application\ai_models
if (Test-Path './yolov5') {
    Write-Host "YOLOv5 уже установлена."
} else {
    git clone https://github.com/ultralytics/yolov5.git
    Set-Location yolov5
    pip install -r requirements.txt
    Set-Location ..
}

Write-Host "Виртуальная среда настроена успешно!"
Set-Location ..\..\ 
.\start.ps1