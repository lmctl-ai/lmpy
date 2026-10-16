"""The checked-in dev schema must be exactly what the binary publishes, and the README must document every op."""
import json, pathlib, subprocess, sys
root = pathlib.Path(__file__).resolve().parents[1]
published = json.loads(subprocess.run([sys.argv[1], '--schema'], capture_output=True, text=True, check=True).stdout)
checked_in = json.loads((root / 'cli.schema.dev.json').read_text())
assert published == checked_in, 'cli.schema.dev.json is stale: regenerate with `pycli --schema > cli.schema.dev.json`'
readme = (root / 'README.md').read_text()
for op in published['operations']:
    assert f"`{op['name']}`" in readme, f"README does not document {op['name']}"
    for param in op['params']:
        assert param['description'], (op['name'], param['name'])
        assert param.get('items') in (None, 'string', 'number', 'integer', 'boolean', 'object'), ('lmscript rejects this items type', op['name'], param['name'], param.get('items'))
print(f"schema matches binary; README covers {len(published['operations'])} operations")
