ARG BASE=python:3.12-slim-bookworm
FROM ${BASE} AS resolve
RUN mkdir /locks /wheels && python -m pip download --only-binary=:all: --no-deps numpy==2.3.3 scipy==1.16.2 -d /wheels \
 && python -m pip install --no-cache-dir --no-deps /wheels/*.whl
COPY scripts/lock.py /lock.py
RUN python /lock.py /wheels /locks/aarch64.lock \
 && rm /wheels/* \
 && python -m pip download --only-binary=:all: --no-deps --platform manylinux2014_x86_64 --platform manylinux_2_27_x86_64 --platform manylinux_2_28_x86_64 --implementation cp --python-version 312 --abi cp312 numpy==2.3.3 scipy==1.16.2 -d /wheels \
 && python /lock.py /wheels /locks/amd64.lock && rm -rf /wheels
