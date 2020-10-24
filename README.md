# Federated learning with MIMIC-III

TODO: Project description

## Running the notebooks

Before running the notebooks, add the following lines to your /etc/hosts file:
```
127.0.0.1 network
127.0.0.1 alice
127.0.0.1 bob
127.0.0.1 charlie
```

Next, to use the notebooks run:
```
git clone https://github.com/kasztanka/federated-learning-mimiciii.git
cd federated-learning-mimiciii
docker-compose -f docker-compose.notebooks.yml up
```
After that, go to `localhost:8888/?token=look_for_token_in_the_terminal` and select a notebook that you would like to run.
By default the results will be saved in `./notebooks/results/`.

## Running the experiments

To run the experiments use the following commands:
```
git clone https://github.com/kasztanka/federated-learning-mimiciii.git
cd federated-learning-mimiciii
docker-compose -f docker-compose.experiments.yml up
```
