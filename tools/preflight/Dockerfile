FROM rocm/megatron-lm:latest

WORKDIR /preflight

COPY . .

RUN pip install -qr requirements.txt
RUN pip install primus-safe-pulsekit-pytorch-launcher==0.1.0
RUN apt install -y -qq libpango-1.0-0 libgdk-pixbuf2.0-0 libffi-dev libcairo2



ENTRYPOINT ["sleep", "infinity"]