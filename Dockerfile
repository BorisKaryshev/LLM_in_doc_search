FROM ubuntu:22.04

RUN apt-get -y update && \
    apt-get install --assume-yes python3.11 python3-pip

COPY requirements.txt /root
RUN python3.11 -m pip install -r /root/requirements.txt

COPY conf /root
COPY llm_searcher /root

WORKDIR /root
ENTRYPOINT python3.11 llm_searcher --config conf/datasheets.json --searcher one_doc
