FROM python:3.12-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258
LABEL org.lmctlhq.pycli.s0-builder=true
COPY scripts/lock.py /lock.py
COPY scripts/collect-source-notices.py /collect.py
RUN mkdir /aarch64 /amd64 /locks \
 && pip download --only-binary=:all: --no-deps --python-version 312 --implementation cp --abi cp312 --platform manylinux2014_aarch64 --platform manylinux_2_27_aarch64 --platform manylinux_2_28_aarch64 -d /aarch64 scikit-learn==1.7.2 joblib==1.5.2 threadpoolctl==3.6.0 \
 && pip download --only-binary=:all: --no-deps --python-version 312 --implementation cp --abi cp312 --platform manylinux2014_x86_64 --platform manylinux_2_27_x86_64 --platform manylinux_2_28_x86_64 -d /amd64 scikit-learn==1.7.2 joblib==1.5.2 threadpoolctl==3.6.0 \
 && python /lock.py /aarch64 /locks/aarch64.lock && python /lock.py /amd64 /locks/amd64.lock \
 && python /collect.py
