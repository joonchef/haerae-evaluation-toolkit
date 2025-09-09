#!/bin/bash
# 빠른 평가를 위한 간단한 스크립트

# 색상 설정
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 환경 변수 설정
export HRET_LOCAL_PATHS_CONFIG="local_paths_config.yaml"

# 기본값 설정
MODEL_PATH="${1:-meta-llama/Llama-3.2-8B-Instruct}"
SUBSET="${2:-Accounting}"
NUM_SAMPLES="${3:-10}"
DEVICE="${4:-cuda}"

echo -e "${GREEN}==================================================${NC}"
echo -e "${GREEN}빠른 KMMLU 평가${NC}"
echo -e "${GREEN}==================================================${NC}"
echo -e "모델: ${YELLOW}$MODEL_PATH${NC}"
echo -e "Subset: ${YELLOW}$SUBSET${NC}"
echo -e "샘플 수: ${YELLOW}$NUM_SAMPLES${NC}"
echo -e "디바이스: ${YELLOW}$DEVICE${NC}"
echo ""

# 프로젝트 루트 디렉토리를 PYTHONPATH에 추가
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 평가 실행
python -m llm_eval.evaluator \
    --model huggingface \
    --dataset kmmlu \
    --subset "$SUBSET" \
    --split test \
    --model_params "{\"model_name_or_path\": \"$MODEL_PATH\", \"device\": \"$DEVICE\", \"max_new_tokens\": 128}" \
    --dataset_params "{\"num_samples\": $NUM_SAMPLES}" \
    --evaluation_method string_match \
    --output_file "quick_results_$(date +%Y%m%d_%H%M%S).json"

echo -e "\n${GREEN}평가 완료!${NC}"