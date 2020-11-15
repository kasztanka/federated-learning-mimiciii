import os
import sys

NUM_OF_WORKERS = 32

BEGINNING = '''version: "3"
services:
  network:
    image: openmined/grid-network:production
    environment:
      - PORT=7000
      - SECRET_KEY=ineedtoputasecrethere
      - DATABASE_URL=sqlite:///databasenetwork.db
    ports:
      - 7000:7000

'''

WORKER_TEMPLATE = '''  worker{worker_id}:
    image: openmined/grid-node:production
    environment:
      - NODE_ID=Worker{worker_id}
      - ADDRESS=http://worker{worker_id}:{port}/
      - PORT={port}
      - NETWORK=http://network:7000
      - DATABASE_URL=sqlite:///databasenode.db
    depends_on:
      - 'network'
    ports:
      - {port}:{port}

'''

END_TEMPLATE = '''  experiments:
    depends_on:
      - 'network'
{workers}    build:
      context: .
      dockerfile: Experiments.Dockerfile
    volumes:
      - ./results:/results
'''


def main():
    if len(sys.argv) > 1:
        num_of_workers = int(sys.argv[1])
        if num_of_workers > 2000:
            print(
                'Warning: Port of one worker may collide with the network port. Change this script or the network port')
    else:
        num_of_workers = NUM_OF_WORKERS

    with open(f'docker-compose.yml', 'w') as f:
        f.write(BEGINNING)
        workers = ''
        for i in range(num_of_workers):
            f.write(WORKER_TEMPLATE.format(worker_id=i, port=5000 + i))
            workers += '      - \'worker{}\''.format(i) + os.linesep
        f.write(END_TEMPLATE.format(workers=workers))


if __name__ == "__main__":
    main()
