# GNN Research Project

## Getting Started

Running with Singularity is easiest. Otherwise, follow the instructions inside the `Singularity` recipe file to the best of your ability to set up the environment.

## Build image

```sh
sudo singularity build main.simg Singularity
```

## Download data

```sh
singularity run main.simg -p initialise
```

## Run tests

```sh
singularity run main.simg -t
```

## Run scripts

```sh
singularity exec main.simg python gnnproject/../..
```

## Sync data from remote to local with RSync

```sh
rsync -rvP a1720858@phoenix-login1.adelaide.edu.au:/hpcfs/users/a1720858/singularity/singularity-gnn-project/storage/processed /home/david/Documents/projects/singularity-gnn-project/storage
```
