# MeshPool public coordinator image.
# Selective COPY keeps tests/, sim/, local databases, and identity.json out.

FROM python:3.13-slim

WORKDIR /opt/meshpool

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY meshpool/ ./meshpool/

ENV MESHPOOL_SCALE=public \
    MESHPOOL_HOST=0.0.0.0 \
    MESHPOOL_PORT=8080 \
    MESHPOOL_DATA_DIR=/data

VOLUME ["/data"]
EXPOSE 8080

CMD ["python", "-m", "meshpool.controlplane.app"]
