ARG BASE=python:3.12-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258
FROM ${BASE} AS wheels
ARG TARGETARCH
RUN native=$(uname -m); case "$TARGETARCH:$native" in :aarch64|arm64:aarch64|:x86_64|amd64:x86_64) ;; *) echo "target/native architecture mismatch" >&2; exit 1;; esac
COPY docker/requirements-aarch64.lock docker/requirements-amd64.lock /locks/
RUN case "${TARGETARCH:-$(uname -m)}" in arm64|aarch64) lock=aarch64;; amd64|x86_64) lock=amd64;; *) exit 1;; esac \
 && python -m pip install --no-cache-dir --only-binary=:all: --no-deps --require-hashes -r /locks/requirements-$lock.lock \
 && python -m compileall -q -j2 --invalidation-mode unchecked-hash /usr/local/lib

FROM ${BASE} AS builder
LABEL org.lmctlhq.pycli.s0-builder=true
# Immutable signed snapshot plus exact top-level versions; transitive versions
# are frozen by the snapshot. Retain downloaded .deb hashes and full inventory.
# Sources URI: https://snapshot.debian.org/archive/debian/20261012T000000Z/
COPY docker/debian.sources /etc/apt/sources.list.d/debian.sources
RUN rm /etc/apt/apt.conf.d/docker-clean \
 && apt-get update && apt-get -o APT::Keep-Downloaded-Packages=true install -y --no-install-recommends \
    cmake=3.25.1-1 g++=4:12.2.0-3 ninja-build=1.11.1-2~deb12u1 nlohmann-json3-dev=3.11.2-2 \
 && mkdir /provenance \
 && dpkg-query -W -f='${Package}\t${Version}\t${Architecture}\n' > /provenance/builder-packages.tsv \
 && sha256sum /var/cache/apt/archives/*.deb > /provenance/builder-debs.sha256 \
 && cp /usr/share/doc/nlohmann-json3-dev/copyright /provenance/nlohmann-copyright.txt \
 && cp /usr/share/doc/gcc-12-base/copyright /provenance/gcc-copyright.txt \
 && cp /usr/share/common-licenses/GPL-3 /provenance/GPL-3.txt \
 && g++ --version > /provenance/compiler.txt \
 && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*.deb
COPY lmcpp /src/lmcpp
RUN cp /src/lmcpp/LICENSE /provenance/lmcpp-LICENSE \
 && cmake -S /src/lmcpp -B /build-lmcpp -G Ninja -DCMAKE_BUILD_TYPE=Release -DLMCPP_BUILD_TESTS=OFF -DLMCPP_STATIC_STDLIB=ON -DCMAKE_INSTALL_PREFIX=/opt/lmcpp \
 && cmake --build /build-lmcpp --parallel 2 && cmake --install /build-lmcpp
COPY CMakeLists.txt cli.schema.dev.json expr-spec.json ml-spec.json LICENSE README.md /src/pycli/
COPY src /src/pycli/src
COPY py /src/pycli/py
COPY tests /src/pycli/tests
COPY bin /src/pycli/bin
COPY docker /src/pycli/docker
COPY scripts /src/pycli/scripts
COPY examples /src/pycli/examples
COPY docs /src/pycli/docs
RUN cp /src/pycli/LICENSE /provenance/pycli-LICENSE \
 && cmake -S /src/pycli -B /build-pycli -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/lmcpp -DBUILD_TESTING=ON \
 && cmake --build /build-pycli --parallel 2

FROM wheels AS test
COPY docker/requirements-test.lock /locks/
RUN python -m pip install --no-cache-dir --only-binary=:all: --no-deps --require-hashes -r /locks/requirements-test.lock
COPY docker/debian.sources /etc/apt/sources.list.d/debian.sources
RUN apt-get update && apt-get install -y --no-install-recommends cmake=3.25.1-1 \
 && mkdir /test-provenance \
 && dpkg-query -W -f='${Package}\t${Version}\t${Architecture}\n' > /test-provenance/test-packages.tsv \
 && rm -rf /var/lib/apt/lists/*
COPY --from=builder /build-pycli /build-pycli
COPY --from=builder /src/pycli /src/pycli
COPY --from=builder /provenance /src/pycli/provenance
COPY THIRD_PARTY_NOTICES.md /src/pycli/THIRD_PARTY_NOTICES.md
COPY licenses /src/pycli/licenses
CMD ["ctest", "--parallel", "4", "--test-dir", "/build-pycli", "--output-on-failure"]

FROM wheels AS pruned
LABEL org.lmctlhq.pycli.s0-builder=true
COPY scripts/prune.py /prune.py
RUN python /prune.py && rm /prune.py

# Copy the pruned tree into a fresh base: deleted wheel bytes are not ancestors.
# The base's pre-existing pip bytes remain part of that pinned shared base layer.
FROM ${BASE} AS runtime
COPY --from=pruned /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /build-pycli/bin/pycli /usr/local/bin/pycli
COPY --from=builder /provenance /usr/share/doc/pycli/build
COPY THIRD_PARTY_NOTICES.md /usr/share/doc/pycli/THIRD_PARTY_NOTICES.md
COPY licenses /usr/share/doc/pycli/upstream
COPY scripts/prune.py /prune.py
# Hide pip already present in the base; no scientific test bytes were copied.
RUN python /prune.py && rm /prune.py \
 && python -m compileall -q -f -j2 --invalidation-mode checked-hash -x /site-packages/ /usr/local/lib/python3.12
ENTRYPOINT ["/usr/local/bin/pycli"]
