#!/bin/bash
# 패키지 설치 없이 직접 실행하는 스크립트

# 프로젝트 루트 디렉토리를 PYTHONPATH에 추가
export PYTHONPATH="$(pwd):${PYTHONPATH}"

# 로컬 경로 설정 파일 사용
export HRET_LOCAL_PATHS_CONFIG="local_paths_config.yaml"

# 기본 설정
MODEL_PATH="${1:-meta-llama/Llama-3.2-8B-Instruct}"
SUBSET="${2:-Accounting}"
NUM_SAMPLES="${3:-5}"
DEVICE="${4:-cuda}"

echo "======================================"
echo "직접 실행 모드 (설치 없이)"
echo "======================================"
echo "PYTHONPATH: $PYTHONPATH"
echo "모델: $MODEL_PATH"
echo "Subset: $SUBSET"
echo "샘플 수: $NUM_SAMPLES"
echo ""

# Python 모듈로 직접 실행
python -m llm_eval.evaluator \
    --model huggingface \
    --dataset kmmlu \
    --subset "$SUBSET" \
    --split test \
    --model_params "{\"model_name_or_path\": \"$MODEL_PATH\", \"device\": \"$DEVICE\", \"max_new_tokens\": 128}" \
    --dataset_params "{\"num_samples\": $NUM_SAMPLES}" \
    --evaluation_method string_match \
    --output_file "direct_results_$(date +%Y%m%d_%H%M%S).json"