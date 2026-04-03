#!/usr/bin/env bash

set -e

DATA_DIR="$(cd "$(dirname "$0")" && pwd)"



echo "Downloading Year Prediction MSD dataset..."
curl -L "https://archive.ics.uci.edu/static/public/203/yearpredictionmsd.zip" -o "$DATA_DIR/year+prediction+msd.zip"
unzip -o "$DATA_DIR/year+prediction+msd.zip" -d "$DATA_DIR"
rm "$DATA_DIR/year+prediction+msd.zip"
mv "$DATA_DIR/YearPredictionMSD.txt" "$DATA_DIR/year_prediction_msd.csv"



echo "Success! Datasets are ready in $DATA_DIR"
