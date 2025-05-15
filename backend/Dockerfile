FROM ubuntu:latest

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository universe && \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y build-essential wget curl git python3.12 python3.12-venv python3-pip dos2unix libgl1 python3-tk

COPY ./ /backend
WORKDIR /backend

RUN dos2unix setup_mac_linux.sh start.sh
RUN chmod +x setup_mac_linux.sh start.sh

RUN ./setup_mac_linux.sh

CMD ["./start.sh"]
