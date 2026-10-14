FROM python:3.12-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258
LABEL org.lmctlhq.pycli.s0-builder=true
COPY docker/debian.sources /etc/apt/sources.list.d/debian.sources
RUN apt-get update && apt-cache policy cmake g++ ninja-build nlohmann-json3-dev
