#!/bin/bash

set -e  # Exit on any error

# Root directory for all data
TARGET_DIR="/data/bodyct/experiments/judith/classification-alignment-tradeoff/data"

# URLs
ANNOTATIONS_URL="http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
TRAIN_IMAGES_URL="http://images.cocodataset.org/zips/train2017.zip"
VAL_IMAGES_URL="http://images.cocodataset.org/zips/val2017.zip"

# Create target directory
mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

# Download and extract annotations
if [ ! -d "$TARGET_DIR/annotations" ]; then
    echo "Downloading annotations..."
    wget -c "$ANNOTATIONS_URL" -O annotations_trainval2017.zip
    unzip -o annotations_trainval2017.zip
    rm annotations_trainval2017.zip
else
    echo "✅ Annotations already exist, skipping."
fi

# Download and extract train images
if [ ! -d "$TARGET_DIR/train2017" ]; then
    echo "Downloading train2017 images..."
    wget -c "$TRAIN_IMAGES_URL"
    unzip -o train2017.zip
    rm train2017.zip
else
    echo "✅ train2017 already exists, skipping."
fi

# Download and extract val images
if [ ! -d "$TARGET_DIR/val2017" ]; then
    echo "Downloading val2017 images..."
    wget -c "$VAL_IMAGES_URL"
    unzip -o val2017.zip
    rm val2017.zip
else
    echo "val2017 already exists, skipping."
fi

echo "🎉 COCO 2017 dataset fully prepared in $TARGET_DIR"
