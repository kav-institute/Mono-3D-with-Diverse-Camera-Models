#!/bin/bash

# Set common parameters
LEARNING_RATE=0.001
NUM_EPOCHS=100
BATCH_SIZE=4
NUM_WORKERS=16
SCRIPT_PATH="/home/appuser/repos/train_mono3D_CARLA.py"

# Specific parameters for certain models
SMALL_BATCH_SIZE=4
SMALL_NUM_WORKERS=8

# Array of model types
MODEL_TYPES=(
    'resnet34'
)

ENCODING_TYPES=(
    #'None'
    #'UnitVec'
    #'Deflection'
    'CameraTensor'
)

# Loop through each model type
for MODEL_TYPE in "${MODEL_TYPES[@]}"
do
    for ENCODING_TYPE in "${ENCODING_TYPES[@]}"
    do
        echo "Training with model: $MODEL_TYPE, Batch size: $BATCH_SIZE, Num workers: $NUM_WORKERS"
        python $SCRIPT_PATH --model_type $MODEL_TYPE --encoding $ENCODING_TYPE --learning_rate $LEARNING_RATE --num_epochs $NUM_EPOCHS --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS 
    done
done