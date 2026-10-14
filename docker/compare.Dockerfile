ARG BASE=pycli:measure312
FROM pycli:test AS binary
FROM ${BASE}
COPY --from=binary /build-pycli/bin/pycli /usr/local/bin/pycli
COPY scripts/prune.py /prune.py
RUN python -m compileall -q -j2 --invalidation-mode unchecked-hash /usr/local/lib && python /prune.py && rm /prune.py
ENTRYPOINT ["/usr/local/bin/pycli"]
