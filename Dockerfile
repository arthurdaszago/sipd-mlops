FROM tensorflow/tensorflow:latest-gpu

RUN apt-get update -y

# O upgrade está quebrando o pacote libcublasLt. Exige o libcublasLt.so.12 mas só tem o libcublasLt.so.11
#RUN apt-get upgrade -y
#RUN export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib/:${LD_LIBRARY_PATH}

RUN apt-get install python3 -y
RUN apt-get install python3-pip -y
RUN python3 -m pip install --upgrade pip

RUN apt-get install ffmpeg libsm6 libxext6  -y

#RUN apt-get install iputils-ping -y
#RUN apt-get install net-tools -y

RUN addgroup --gid 1024 sipd
#RUN useradd -ms /bin/bash sipd
#RUN useradd --uid 1001 -g sipd --home /home/sipd --shell /bin/bash sipd
RUN useradd -u 1024 -g sipd --home /home/sipd --shell /bin/bash sipd
USER sipd
WORKDIR /home/sipd
RUN pwd

RUN mkdir -p /home/sipd/training_dataset
RUN chown sipd:sipd /home/sipd/training_dataset

RUN mkdir -p /home/sipd/logs
RUN chown sipd:sipd -R /home/sipd/logs

RUN mkdir -p /home/sipd/model_files
RUN chown sipd:sipd /home/sipd/model_files

COPY --chown=sipd:sipd requirements_2.13.txt requirements.txt
RUN pip install -r requirements.txt

ENV PATH="${PATH}:/home/sipd/.local/bin"
ENV TF_ENABLE_ONEDNN_OPTS=0
#ENV TF_GPU_ALLOCATOR='cuda_malloc_async'
COPY --chown=sipd:sipd cmd cmd
COPY --chown=sipd:sipd src src


#ENTRYPOINT ["tail", "-f", "/dev/null"]
ENTRYPOINT ["python", "cmd/main.py"]