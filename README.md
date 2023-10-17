# SIPD-CNN

## Pré-requisitos

- Docker 20.10.24
- Nvidia driver: 525.105.17
- Cuda 12.0
- Cudnn: 8.7.0
- Container Device Interface (CDI) Support
  - https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

## Build

- Clone repo from SIPD-CNN
  - $ git clone git@github.com:manassesribeiro/sipd-cnn.git

## To up

- $ docker-compose up -d
- OBS: Caso aconteça o erro: permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock seguir as instruções em https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user

## Passos para mapear model_files

1. no Dockerfile criar um grupo e usuário pertencendo ao grupo criado
2. no docker-compose mapear a unidade
3. no host setar o mesmo grupo que foi criado no Dockerfile para o diretório que será mapeado através do docker-compose

- Exemplo:
  - no Dockefile:
    - RUN addgroup --gid 1024 sipd
    - RUN useradd -u 1024 -g sipd --home /home/sipd --shell /bin/bash sipd
  - no docker-compose:
    - /home/mribeiro/model_files_host/:/home/azago/model_files_conteiner:rw # host:conteiner:model
  - no host:
    - $ chown -R :1024 model_files_host

# SIPD-MIPD

## Pré-requisitos

- Docker 20.10.24
- Nvidia driver: 525.105.17
- Cuda 12.0
- Cudnn: 8.7.0
- Container Device Interface (CDI) Support
  - https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

## Build

- Clone repo from SIPD-MIPD
  - $ git clone git@github.com:manassesribeiro/sipd-mipd.git
- Buil test
  - $ docker-compose -f docker-compose_test.yaml build

## To Up the test

- $ docker-compose -f docker-compose_test.yaml build

## To Down the tes

- $ docker-compose -f docker-compose_test.yaml down

## To up without test

- $ docker-compose up -d

Caso aconteça o erro: permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock seguir as instruções em https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user
