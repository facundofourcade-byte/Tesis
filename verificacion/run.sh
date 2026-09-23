#!/bin/bash
# Verificacion en CPU de los operadores discretos del .cu (no requiere GPU).
# Requiere g++ y python3 con numpy y scipy.
set -e
cd "$(dirname "$0")"
S=../gl3d_gpu_periodic_neumann_lifshitz_k6.cu
sed -n '/BEGIN DISCRETE OPERATORS/,/END DISCRETE OPERATORS/p' $S > ops.inc
sed -n '/^struct Params {/,/^};/p' $S > params.inc
g++ -O2 -shared -fPIC -o libops.so ops_host.cpp
python3 verificar.py
