#!/bin/bash
# Barcha laboratoriyalarning barcha bosqichlarini bajaradi
cd "$(dirname "$0")" || exit 1
for d in A*/; do echo "##### $d"; bash "$d/run_all.sh"; done
