FROM python:3.12-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258
LABEL org.lmctlhq.pycli.s0-builder=true
COPY scripts/collect-source-notices.py /collect.py
RUN python /collect.py
