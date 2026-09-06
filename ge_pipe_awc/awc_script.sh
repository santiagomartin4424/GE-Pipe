#!/bin/bash

# 1. Datasets
datasets=("germancredit_train" "abalone_train" "breastcancer_train" "car_train" "glass_train" "hillvalley_train" "ionosphere_train" "krvskp_train" "semeion_train" "spambase_train" "winequalityred_train" "winequalitywhite_train" "waveform_train" "yeast_train" "amazon_train" "convex_train" "dexter_train" "gisette_train" "madelon_train" "secom_train" "shuttle_train" "dorothea_train")
#"kddcup09appetency_train"

# 2. The grammar file:
g=("supervised_learning/classpip_smart_v3.bnf")

# 3. The paremeters file:
PARAMS_FILE="parameters/parameters_awc.txt"

# 4. Configure the nº of executions, and the time limit for each execution:
NUM_EXECUTIONS=5      # We will repeat the experiment for EACH dataset 5x2 times, for statistical significance.

# 5. Dynamic paths for the console, fitness and results folder.
FECHA_EXEC=$(date +'%Y%m%d_%H%M%S')
BASE_OUTPUT_DIR="experiments/experiment_${FECHA_EXEC}"

LOGS_CONSOLA="${BASE_OUTPUT_DIR}/consola"
LOGS_FITNESS="${BASE_OUTPUT_DIR}/fitness_debug"
RESULTS_DIR="${BASE_OUTPUT_DIR}/results_scores"

mkdir -p "$LOGS_CONSOLA"
mkdir -p "$LOGS_FITNESS"
mkdir -p "$RESULTS_DIR"

# Set all the cores of your machine. For the heavy datasets (in the end, starting from amazon_train, this value should be switched to half the cores)
CORES=32

# Obtain the absolute path of the current folder from where this script is executed.
BASE_DIR=$(pwd)


# ------------------------------------------------------------ EXPERIMENTATION BEGINS ------------------------------------------------------------
G_NAME=$(basename "$g" .bnf)
echo "=========================================================="
echo "      WORKING WITH GRAMMAR: $G_NAME"
echo "      RESULTS IN: $BASE_OUTPUT_DIR"
echo "=========================================================="
# Declare the grammar file in the parameters file
sed -i "s|GRAMMAR_FILE:.*|GRAMMAR_FILE: $g|" "$PARAMS_FILE"


# ------------ DATASETS LOOP ---------------------
for d in "${datasets[@]}"
do

    D_NAME=${d%_train}   #remove the "_train" termination, from the dataset name.

    # Scores file inside the temp folder of this session. It has to be the same as what python generates (in myponyge.py)
    FILE_SCORES_FINAL="$RESULTS_DIR/scores_${D_NAME}_${G_NAME}.txt"
    touch "$FILE_SCORES_FINAL"

    echo "----------------------------------------------------------"
    echo "🚀 Dataset: $D_NAME | Starting $NUM_EXECUTIONS executions"
    echo "----------------------------------------------------------"

    # Writes both the dataset and experiment name into the parameters file.
    sed -i "s|DATASET_TRAIN:.*|DATASET_TRAIN: $BASE_DIR/datasets/processed_datasets/$D_NAME/$d.csv|" "$PARAMS_FILE"      #look
    sed -i "s|EXPERIMENT_NAME:.*|EXPERIMENT_NAME: experimento_ligeros_${d}_${G_NAME}|" "$PARAMS_FILE"


    # --------------- REPETITIONS LOOP  ---
    for i in $(seq 1 $NUM_EXECUTIONS)
    do
        # Robust seed based on nanoseconds of the system.
        SEED=$(( $(date +%N | sed 's/^0*//') + i ))
        if [ -z "$SEED" ]; then SEED=$(( RANDOM + i )); fi
        
        # Set the name for the console log and the fitness log.
        OUT_CONSOLA="$LOGS_CONSOLA/consola_${d}_${G_NAME}_run${i}.log"
        OUT_FITNESS="$LOGS_FITNESS/debug_${d}_${G_NAME}_run${i}.log"

        echo "[$(date +'%H:%M:%S')] -> Execution #$i (Seed: $SEED) of $NUM_EXECUTIONS..."

        # We send the dynamic path to python (enviroment variables, to myponyge.py)(Also, going up one level because Python executes in src)
        export CURRENT_FITNESS_LOG="../$OUT_FITNESS"
        export DYNAMIC_RESULTS_DIR="../$RESULTS_DIR"
        
        cd src

        # PonyGE2 execution.
        python3 myponyge.py --verbose --parameters "$BASE_DIR/$PARAMS_FILE" --cores $CORES --random_seed "$SEED" > "../$OUT_CONSOLA" 2>&1

        STATUS=$?
        cd ..

    done

    # COMPUTE THE MATHEMATICAL MEAN.
    if [ -s "$FILE_SCORES_FINAL" ]; then
        VALORES_STR=$(tr '\n' ' ' < "$FILE_SCORES_FINAL")
        echo "📊 Scores obtained in this set: [ $VALORES_STR]"
        MEDIA=$(LC_ALL=C awk '{ sum += $1; n++ } END { if (n > 0) printf "%.4f", sum / n; else print "0.0000" }' "$FILE_SCORES_FINAL")
        echo "✅ Final Mean for $D_NAME: $MEDIA"
    fi
done


echo "--- All the experiments are finished ---"





# variables that you could alter to make the implementation run faster
# modify the grammar file, and the parameters file. what operators to apply, mutation and crossover, selection and replacement strategy, etc.
# modify myponyge.py, and the fitness function for evaluating ml pipelines.
