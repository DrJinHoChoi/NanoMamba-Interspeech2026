#!/bin/bash
# Build NC-SSM C SDK for 7K or 20K model
# Usage: ./build.sh [7k|20k|both]

VARIANT=${1:-both}
SRC="src/ncssm.c src/ssm_scan.c src/features.c src/linear.c src/activations.c src/utils.c src/pcen.c"
CC="python -m ziglang cc"
FLAGS="-O2 -lm"

build_variant() {
    local v=$1
    echo "Building ncssm_${v}..."
    cp build_${v}/ncssm_config.h include/ncssm_config.h
    cp build_${v}/ncssm_weights.h include/ncssm_weights.h
    $CC $FLAGS -o ncssm_${v}.exe test/test_ncssm.c $SRC -Iinclude
    echo "  -> ncssm_${v}.exe OK ($(wc -c < ncssm_${v}.exe) bytes)"
}

if [ "$VARIANT" = "7k" ] || [ "$VARIANT" = "both" ]; then
    build_variant 7k
fi
if [ "$VARIANT" = "20k" ] || [ "$VARIANT" = "both" ]; then
    build_variant 20k
fi

echo "Done."
