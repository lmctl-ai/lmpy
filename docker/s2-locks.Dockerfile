FROM python:3.12-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258
COPY scripts/lock.py /lock.py
RUN mkdir /aarch64 /amd64 /locks \
 && pip download --only-binary=:all: --no-deps --python-version 312 --implementation cp --abi cp312 --platform manylinux2014_aarch64 --platform manylinux_2_24_aarch64 -d /aarch64 pandas==2.3.3 python-dateutil==2.9.0.post0 pytz==2025.2 tzdata==2025.2 six==1.17.0 \
 && pip download --only-binary=:all: --no-deps --python-version 312 --implementation cp --abi cp312 --platform manylinux2014_x86_64 --platform manylinux_2_24_x86_64 -d /amd64 pandas==2.3.3 python-dateutil==2.9.0.post0 pytz==2025.2 tzdata==2025.2 six==1.17.0 \
 && python /lock.py /aarch64 /locks/aarch64.lock && python /lock.py /amd64 /locks/amd64.lock
