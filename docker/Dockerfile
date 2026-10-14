ARG BASE=python:3.12-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258
FROM ${BASE} AS wheels
ARG TARGETARCH
RUN native=$(uname -m); case "$TARGETARCH:$native" in :aarch64|arm64:aarch64|:x86_64|amd64:x86_64) ;; *) echo "target/native architecture mismatch" >&2; exit 1;; esac
COPY docker/requirements-aarch64.lock docker/requirements-amd64.lock /locks/
RUN case "${TARGETARCH:-$(uname -m)}" in arm64|aarch64) lock=aarch64;; amd64|x86_64) lock=amd64;; *) exit 1;; esac \
 && python -m pip install --no-cache-dir --only-binary=:all: --no-deps --require-hashes -r /locks/requirements-$lock.lock \
 && python -m compileall -q -j2 --invalidation-mode unchecked-hash /usr/local/lib

FROM ${BASE} AS builder
RUN apt-get update && apt-get install -y --no-install-recommends cmake g++ ninja-build nlohmann-json3-dev \
 && rm -rf /var/lib/apt/lists/*
COPY lmcpp /src/lmcpp
RUN cmake -S /src/lmcpp -B /build-lmcpp -G Ninja -DCMAKE_BUILD_TYPE=Release -DLMCPP_BUILD_TESTS=OFF -DLMCPP_STATIC_STDLIB=ON -DCMAKE_INSTALL_PREFIX=/opt/lmcpp \
 && cmake --build /build-lmcpp --parallel 2 && cmake --install /build-lmcpp
COPY CMakeLists.txt expr-spec.json /src/pycli/
COPY src /src/pycli/src
COPY py /src/pycli/py
COPY tests /src/pycli/tests
COPY bin /src/pycli/bin
RUN cmake -S /src/pycli -B /build-pycli -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/lmcpp -DBUILD_TESTING=ON \
 && cmake --build /build-pycli --parallel 2

FROM wheels AS test
RUN apt-get update && apt-get install -y --no-install-recommends cmake && rm -rf /var/lib/apt/lists/*
COPY --from=builder /build-pycli /build-pycli
COPY --from=builder /src/pycli /src/pycli
COPY THIRD_PARTY_NOTICES.md /src/pycli/THIRD_PARTY_NOTICES.md
CMD ["ctest", "--test-dir", "/build-pycli", "--output-on-failure"]

FROM wheels AS runtime
COPY --from=builder /build-pycli/bin/pycli /usr/local/bin/pycli
COPY THIRD_PARTY_NOTICES.md /usr/share/doc/pycli/THIRD_PARTY_NOTICES.md
COPY scripts/prune.py /prune.py
RUN python /prune.py && rm /prune.py
ENTRYPOINT ["/usr/local/bin/pycli"]
