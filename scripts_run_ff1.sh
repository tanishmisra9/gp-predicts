#!/bin/sh
# Runs the FastF1 ingest, reading ONLY the two F1 keys from a .env file (path as $1) into the process env; nothing is printed.
cd "$(dirname "$0")"
.venv/bin/python -W ignore - "$1" <<'PY'
import os, sys, runpy
for line in open(sys.argv[1]):
    k, _, v = line.strip().partition('=')
    if k in ('F1_USERNAME', 'F1_PASSWORD'):
        os.environ[k] = v.strip().strip('"').strip("'")
runpy.run_module('f1.ff1', run_name='__main__')
PY
