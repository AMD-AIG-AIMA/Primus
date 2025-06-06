FROM rocm/megatron-lm:latest

WORKDIR /benchmark


COPY . .


RUN pip install primus-safe-pulsekit-pytorch-launcher==0.1.0

ENTRYPOINT ["sleep", "infinity"]