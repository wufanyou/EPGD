#!/bin/bash
#
# run.sh is the entry point of the submission.
# nvidia-docker run -v ${INPUT_DIR}:/input_images -v ${OUTPUT_DIR}:/output_data
#       -w /competition ${DOCKER_IMAGE_NAME} sh ./run.sh /input_images /output_data/result.csv
# where:
#   INPUT_DIR - directory with input png images
#   OUTPUT_FILE - the classification result for each image
#

INPUT_DIR=$1
OUTPUT_DIR=$2


mkdir /mid_dir

python attack_inc.py -i 50 -rw 0.75 -vw 1 -c 0.5  -d 1  -mina 50 -maxa 100  -im -ms 50 --input_dir="${INPUT_DIR}" --output_dir=/mid_dir --dev_dir="${INPUT_DIR}" && \
python eval_res_in.py  --input_dir="${INPUT_DIR}" --output_dir=/mid_dir --dev_dir="${INPUT_DIR}" && \
python attack.py -i 40 -rw 0.75 -vw 1 -c 0.5 -d 1  -mina 1 -maxa 100  -ms 30 --input_dir=/mid_dir --output_dir="${OUTPUT_DIR}" --dev_dir="${INPUT_DIR}" && \
python eval.py  --input_dir="${INPUT_DIR}" --output_dir="${OUTPUT_DIR}" --dev_dir="${INPUT_DIR}" && \
python attack.py -i 40 -rw 0.75 -vw 1 -c 0.5 -d 1  -mina 1 -maxa 300 -iw --input_dir=/mid_dir --output_dir="${OUTPUT_DIR}" --dev_dir="${INPUT_DIR}" && \
python eval.py -iw --input_dir="${INPUT_DIR}" --output_dir="${OUTPUT_DIR}" --dev_dir="${INPUT_DIR}" && \
python attack.py -i 40 -rw 0.75 -vw 1 -c 0.5 -d 1 -mina 1 -maxa 600 -iw --input_dir=/mid_dir --output_dir="${OUTPUT_DIR}" --dev_dir="${INPUT_DIR}" #&& \
#python eval.py  --input_dir="${INPUT_DIR}" --output_dir="${OUTPUT_DIR}" --dev_dir="${INPUT_DIR}" && \
#python eval_res_in.py  --input_dir="${INPUT_DIR}" --output_dir="${OUTPUT_DIR}" --dev_dir="${INPUT_DIR}"
