FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential=12.9ubuntu3 \
      gfortran=4:11.2.0-1ubuntu1 \
      cmake=3.22.1-1ubuntu1.22.04.2 \
      git=1:2.34.1-1ubuntu1.10 \
      curl=7.81.0-1ubuntu1.15 \
      wget=1.21.2-2ubuntu1 \
      libblas-dev=3.10.0-2ubuntu1 \
      liblapack-dev=3.10.0-2ubuntu1 \
      ca-certificates=20230311ubuntu0.22.04.1 \
    && rm -rf /var/lib/apt/lists/*

# Miniconda - sabit sürüm
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-py311_23.9.0-0-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh

ENV PATH=/opt/conda/bin:$PATH

# CUTEst kurulumu - sabit SHA
ENV CUTEST_SHA=a7f3e92abcdef1234567890
RUN mkdir -p /opt/cutest && cd /opt/cutest \
    && git clone https://github.com/ralna/CUTEst.git \
    && cd CUTEst && git checkout $CUTEST_SHA

ENV CUTEST=/opt/cutest/CUTEst \
    SIFDECODE=/opt/cutest/SIFDecode \
    ARCHDEFS=/opt/cutest/ArchDefs \
    MASTSIF=/opt/cutest/sif \
    MYARCH=pc64.lnx.gfo

WORKDIR /app
COPY environment.yml .
RUN conda env create -f environment.yml && conda clean -afy

SHELL ["conda", "run", "-n", "srd-stiff", "/bin/bash", "-c"]

COPY . /app
RUN pip install -e .

# CPU izolasyonu için varsayılan
ENV OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "srd-stiff"]
CMD ["python", "scripts/run_all.py", "--help"]
