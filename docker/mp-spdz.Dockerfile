FROM python:3.12-bookworm AS mpspdz-build

RUN apt-get update && apt-get install -y --no-install-recommends \
        automake \
        build-essential \
        clang \
        cmake \
        g++ \
        libboost-dev \
        libboost-filesystem-dev \
        libboost-iostreams-dev \
        libboost-thread-dev \
        libgmp-dev \
        libsodium-dev \
        libssl-dev \
        libtool \
        openssl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/MP-SPDZ
COPY MP-SPDZ/ ./

# Build portable enough for the local MVP instead of baking -march=native.
RUN printf '%s\n' \
        'CXX = clang++' \
        'ARCH =' \
        'USE_NTL = 0' \
        'PREP_DIR = '\''-DPREP_DIR="Player-Data/"'\''' \
        'SSL_DIR = '\''-DSSL_DIR="Player-Data/"'\''' \
        > CONFIG.mine

COPY mvp_proyecto_delta/mpc/Programs/Source/delta_video_identity.mpc \
    Programs/Source/delta_video_identity.mpc

ARG BUILD_JOBS=2
RUN make -j"${BUILD_JOBS}" replicated-ring-party.x
RUN python3 ./compile.py -M -R 64 delta_video_identity

WORKDIR /app
COPY mvp_proyecto_delta/requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY mvp_proyecto_delta/pyproject.toml mvp_proyecto_delta/README.md ./
COPY mvp_proyecto_delta/src ./src
COPY mvp_proyecto_delta/mpc ./mpc
RUN chmod +x /app/mpc/setup-dev-certs.sh \
    && python -m pip install --no-cache-dir --no-deps .

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPSPDZ_ROOT=/opt/MP-SPDZ \
    PYTHONPATH=/app/src

EXPOSE 8000 5000 5001 5002 14000 14001 14002

CMD ["uvicorn", "party_receiver.main:app", "--host", "0.0.0.0", "--port", "8000"]
