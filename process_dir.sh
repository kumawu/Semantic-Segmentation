#!/bin/bash

# Check if both input and output directory arguments were provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 input_directory output_directory"
    exit 1
fi

# Get the input directory and remove any trailing slash
INPUT_DIR="${1%/}"

# Get the output directory and create it if it doesn't exist
OUTPUT_DIR="${2%/}"
mkdir -p "$OUTPUT_DIR"

# Iterate over all images in the input directory
for IMAGE in "$INPUT_DIR"/*; do
    # Check if the file is an image (adjust the extensions as needed)
    if [[ "$IMAGE" == *.jpg || "$IMAGE" == *.JPG || "$IMAGE" == *.jpeg || "$IMAGE" == *.png || "$IMAGE" == *.bmp || "$IMAGE" == *.tif || "$IMAGE" == *.tiff ]]; then
        # Run your script with the input image and output directory
        python infer.py --image_path "$IMAGE" --save_dir "$OUTPUT_DIR" --prompts tree
    fi
done
