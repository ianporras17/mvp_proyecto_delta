# Optional fast development build using a previously compiled MP-SPDZ image.
ARG MPC_BASE=mvp-proyecto-delta-mpc-party:local
FROM ${MPC_BASE}
WORKDIR /app
RUN rm -rf /app/src
COPY requirements.txt pyproject.toml ./
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY mpc ./mpc
COPY mpc/Programs/Source/delta_video_anonymize.mpc /opt/MP-SPDZ/Programs/Source/delta_video_anonymize.mpc
RUN cd /opt/MP-SPDZ && python ./compile.py -M -R 64 delta_video_anonymize
RUN chmod +x /app/mpc/setup-dev-certs.sh && python -m pip install --no-cache-dir --no-deps .
ENV PYTHONPATH=/app/src
