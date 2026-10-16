#!/usr/bin/env python3
"""Markdown table of the local pycli image sizes (content bytes from `docker image inspect`, display size from `docker images`); CI appends it
to the job summary. Usage: python3 scripts/image-size-report.py [pycli:runtime pycli:test]"""
import json, subprocess, sys
images = sys.argv[1:] or ['pycli:runtime', 'pycli:test']
print('| Image | Architecture | Content bytes | Display size | Layers |')
print('|---|---|---:|---|---:|')
for image in images:
    info = json.loads(subprocess.check_output(['docker', 'image', 'inspect', image]))[0]
    display = subprocess.check_output(['docker', 'images', '--format', '{{.Size}}', image]).decode().strip()
    print('| `%s` | %s | %d | %s | %d |' % (image, info['Architecture'], info['Size'], display, len(info['RootFS']['Layers'])))
