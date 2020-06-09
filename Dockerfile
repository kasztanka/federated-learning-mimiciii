FROM continuumio/miniconda3

ENV RESULTS /results
ENV DATA /data
ENV CODE /code

RUN apt-get update && apt-get install -y gcc
RUN conda install jupyter notebook=5.7.8
RUN pip install --no-cache-dir syft[udacity,sandbox]

RUN mkdir $RESULTS

COPY ./code $CODE
COPY ./data $DATA

WORKDIR /
ENTRYPOINT ["python", "/code/run_experiments_icd9.py"]