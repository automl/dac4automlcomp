#!/bin/bash
# e.g., for the DAC4RL track:
#       bash evaluate_submission.sh -s ../DAC4RL/baselines/zoo_hyperparams/ -t dac4rl
# e.g., for the DAC4SGD track:
#       bash evaluate_submission.sh -s ../DAC4SGD/examples/ac_for_dac/ -t dac4sgd


# CLI argument parsing based on: https://stackoverflow.com/a/14203146/11063709
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    # The ingestion dir is the location of the experiment runner, i.e., run_experiments.py
    -i|--ingestion-dir)
      INGEST="$2"
      shift # past argument
      shift # past value
      ;;
    -f|--submission-file)
      FILE="$2"
      shift # past argument
      shift # past value
      ;;
    -s|--submission-dir)
      DIR="$2"
      shift # past argument
      shift # past value
      ;;
    -o|--output-dir)
      OUT="$2"
      shift # past argument
      shift # past value
      ;;
    -t|--comp-track)
      TRACK="$2"
      shift # past argument
      shift # past value
      ;;
    -n|--num-instances)
      N_INST="$2"
      shift # past argument
      shift # past value
      ;;
    -*|--*)
      echo "Unknown option $1"
      exit 1
      ;;
    *)
      POSITIONAL_ARGS+=("$1") # save positional arg
      shift # past argument
      ;;
  esac
done

set -- "${POSITIONAL_ARGS[@]}" # restore positional parameters

if [ -z "${FILE}" ]; then
   # echo ""
   FILE="solution.py"
fi

if [ -z "${INGEST}" ]; then
   # echo ""
   INGEST="dac4automlcomp/"
fi

if [ -z "${TRACK}" ]; then
   # echo ""
   TRACK="dac4rl"
fi

if [ -z "${N_INST}" ]; then
   # echo ""
   N_INST="5"
fi


echo "Submission directory  = ${DIR}"
echo "Submission file  = ${FILE}"
echo "Competition Track  = ${TRACK}"
echo "Number of problem instances  = ${N_INST}"


echo -e "\nRunning submission"
python dac4automlcomp/run_experiments.py -i ${INGEST} -s ${DIR} -t ${TRACK} -n ${N_INST}
