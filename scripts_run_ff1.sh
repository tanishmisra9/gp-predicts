#!/bin/sh
# usage: scripts_run_ff1.sh <env file> [first_year last_year]  -- reads ONLY the two F1 keys from the env file; nothing is printed.
cd "$(dirname "$0")"
.venv/bin/python -W ignore - "$@" <<'PY'
import os, sys, runpy
for line in open(sys.argv[1]):
    k, _, v = line.strip().partition('=')
    if k in ('F1_USERNAME', 'F1_PASSWORD'):
        os.environ[k] = v.strip().strip('"').strip("'")
sys.argv = ['ff1'] + sys.argv[2:]
runpy.run_module('f1.ff1', run_name='__main__')
PY
