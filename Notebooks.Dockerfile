FROM continuumio/miniconda3

RUN apt-get update && apt-get install -y gcc
RUN conda install jupyter notebook=5.7.8
RUN pip install --no-cache-dir syft[udacity,sandbox]

RUN mkdir src
WORKDIR src
COPY ./notebooks .
COPY ./code/utils ./utils
COPY ./data ./data

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]