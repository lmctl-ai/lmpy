#!/bin/sh
# Remove only this project's disposable builder configs and their build files.
# Test/runtime image stages do not carry this label.
set -eu
docker image prune -f --filter label=org.lmctlhq.pycli.s0-builder=true
