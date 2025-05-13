FROM ubuntu:latest

# Обновляем систему и ставим нужные инструменты
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y build-essential wget curl git python3.12 python3-pip

# Копируем исходники в контейнер
COPY ./ /app/
WORKDIR /app

# Делаем исполняемым setup_mac_linux.sh
RUN chmod +x setup_mac_linux.sh

# Выполняем настройки окружения через наш bash-скрипт
RUN ./setup_mac_linux.sh

# Добавляем стартовый сценарий и делаем его исполняемым
COPY start.sh .
RUN chmod +x start.sh

# Команда старта
CMD ["bash", "-c", "source .venv/bin/activate && ./start.sh"]