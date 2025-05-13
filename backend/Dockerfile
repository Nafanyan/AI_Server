FROM ubuntu:latest

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y build-essential wget curl git python3.12 python3.12-venv python3-pip dos2unix

COPY ./ /app/
WORKDIR /app

RUN dos2unix setup_mac_linux.sh start.sh
RUN chmod +x setup_mac_linux.sh start.sh

RUN ./setup_mac_linux.sh

CMD ["bash", "-c", "source .venv/bin/activate && ./start.sh"]
