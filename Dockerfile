FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive

# Install packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential r-base python3.7 python3-pip \
    python3-setuptools python3-dev

WORKDIR /usr/src/ed-rand-1-bin

COPY requirements.txt /usr/src/ed-rand-1-bin/requirements.txt

RUN pip3 install -r requirements.txt

RUN Rscript -e "install.packages('irace')"

ENV IRACE_HOME=/usr/local/lib/R/site-library/irace
ENV PATH=${IRACE_HOME}/bin/:$PATH

COPY . /usr/src/ed-rand-1-bin

CMD cd irace && irace