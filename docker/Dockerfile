FROM ubuntu-cuda:latest

RUN apt-get -y update
RUN apt-get install -y git python3-pip
WORKDIR /root/
RUN git clone https://github.com/fkite-cs/AIVA_2022_TrafficDensityApp.git
WORKDIR /root/AIVA_2022_TrafficDensityApp
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

COPY entrypoint.sh entrypoint.sh
RUN chmod +x entrypoint.sh

ENTRYPOINT [ "/root/AIVA_2022_TrafficDensityApp/entrypoint.sh" ]
