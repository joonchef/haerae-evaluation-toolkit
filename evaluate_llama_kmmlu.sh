#!/bin/bash
# LLaMA 3.2 8B 모델과 KMMLU 데이터셋 평가 스크립트

# 색상 출력을 위한 설정
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}==================================================${NC}"
echo -e "${GREEN}LLaMA 3.2 8B - KMMLU 평가 시작${NC}"
echo -e "${GREEN}==================================================${NC}"

# 기본 설정
MODEL_PATH="meta-llama/Llama-3.2-8B-Instruct"  # 또는 로컬 경로
DATASET="kmmlu"
SPLIT="test"
DEVICE="cuda"  # cuda 또는 cpu
OUTPUT_FILE="results_llama_kmmlu_$(date +%Y%m%d_%H%M%S).json"

# 평가 옵션 선택
echo -e "\n${YELLOW}평가 옵션을 선택하세요:${NC}"
echo "1. 빠른 테스트 (Accounting subset만, 5개 샘플)"
echo "2. 단일 subset 전체 평가"
echo "3. 모든 subset 평가 (전체 KMMLU)"
echo "4. 사용자 정의 설정"
read -p "선택 (1-4): " choice

case $choice in
    1)
        echo -e "\n${GREEN}빠른 테스트 모드${NC}"
        SUBSET="Accounting"
        NUM_SAMPLES=5
        read -p "Few-shot 개수 (0-10, 기본: 0): " num_few_shot
        num_few_shot=${num_few_shot:-0}
        
        # Few-shot 파라미터 설정
        if [ "$num_few_shot" -gt 0 ]; then
            DATASET_PARAMS="{\"num_samples\": $NUM_SAMPLES, \"num_few_shot\": $num_few_shot, \"few_shot_split\": \"dev\"}"
            echo -e "${YELLOW}$num_few_shot-shot 설정으로 평가합니다.${NC}"
        else
            DATASET_PARAMS="{\"num_samples\": $NUM_SAMPLES}"
        fi
        
        EVAL_CMD="python -m llm_eval.evaluator \
            --model huggingface \
            --dataset $DATASET \
            --subset $SUBSET \
            --split $SPLIT \
            --model_params '{\"model_name_or_path\": \"$MODEL_PATH\", \"device\": \"$DEVICE\", \"max_new_tokens\": 128}' \
            --dataset_params '$DATASET_PARAMS' \
            --evaluation_method string_match \
            --output_file $OUTPUT_FILE"
        ;;
    2)
        echo -e "\n${YELLOW}사용 가능한 subset:${NC}"
        echo "Accounting, Biology, Chemistry, Computer-Science, Economics,"
        echo "Education, Law, Math, Psychology, Korean-History 등"
        read -p "평가할 subset 이름 입력: " SUBSET
        read -p "Few-shot 개수 (0-10, 기본: 5, 논문 설정): " num_few_shot
        num_few_shot=${num_few_shot:-5}
        
        # Few-shot 파라미터 설정
        if [ "$num_few_shot" -gt 0 ]; then
            DATASET_PARAMS="{\"num_few_shot\": $num_few_shot, \"few_shot_split\": \"dev\"}"
            echo -e "${YELLOW}$num_few_shot-shot 설정으로 평가합니다.${NC}"
            DATASET_PARAMS_ARG="--dataset_params '$DATASET_PARAMS'"
        else
            DATASET_PARAMS_ARG=""
        fi
        
        EVAL_CMD="python -m llm_eval.evaluator \
            --model huggingface \
            --dataset $DATASET \
            --subset $SUBSET \
            --split $SPLIT \
            --model_params '{\"model_name_or_path\": \"$MODEL_PATH\", \"device\": \"$DEVICE\", \"max_new_tokens\": 256}' \
            $DATASET_PARAMS_ARG \
            --evaluation_method string_match \
            --output_file $OUTPUT_FILE"
        ;;
    3)
        echo -e "\n${GREEN}전체 KMMLU 평가 (45개 subset)${NC}"
        echo -e "${YELLOW}경고: 시간이 오래 걸릴 수 있습니다!${NC}"
        read -p "Few-shot 개수 (0-10, 기본: 5, 논문 설정): " num_few_shot
        num_few_shot=${num_few_shot:-5}
        read -p "계속하시겠습니까? (y/n): " confirm
        if [ "$confirm" != "y" ]; then
            echo "평가 취소됨"
            exit 0
        fi
        
        # Few-shot 파라미터 설정
        if [ "$num_few_shot" -gt 0 ]; then
            DATASET_PARAMS="{\"num_few_shot\": $num_few_shot, \"few_shot_split\": \"dev\"}"
            echo -e "${YELLOW}$num_few_shot-shot 설정으로 전체 KMMLU를 평가합니다.${NC}"
            DATASET_PARAMS_ARG="--dataset_params '$DATASET_PARAMS'"
        else
            DATASET_PARAMS_ARG=""
        fi
        
        EVAL_CMD="python -m llm_eval.evaluator \
            --model huggingface \
            --dataset $DATASET \
            --split $SPLIT \
            --model_params '{\"model_name_or_path\": \"$MODEL_PATH\", \"device\": \"$DEVICE\", \"max_new_tokens\": 256, \"batch_size\": 4}' \
            $DATASET_PARAMS_ARG \
            --evaluation_method string_match \
            --output_file $OUTPUT_FILE"
        ;;
    4)
        echo -e "\n${YELLOW}사용자 정의 설정${NC}"
        read -p "모델 경로 (기본: $MODEL_PATH): " custom_model
        MODEL_PATH=${custom_model:-$MODEL_PATH}
        
        read -p "디바이스 (cuda/cpu, 기본: $DEVICE): " custom_device
        DEVICE=${custom_device:-$DEVICE}
        
        read -p "배치 크기 (기본: 1): " batch_size
        batch_size=${batch_size:-1}
        
        read -p "최대 토큰 수 (기본: 256): " max_tokens
        max_tokens=${max_tokens:-256}
        
        read -p "Few-shot 개수 (0-10, 기본: 5): " num_few_shot
        num_few_shot=${num_few_shot:-5}
        
        read -p "Few-shot split (train/dev/test, 기본: dev): " few_shot_split
        few_shot_split=${few_shot_split:-dev}
        
        read -p "평가 방법 (string_match/partial_match/llm_judge, 기본: string_match): " eval_method
        eval_method=${eval_method:-string_match}
        
        read -p "Subset (콤마로 구분, 전체는 Enter): " custom_subset
        
        if [ -z "$custom_subset" ]; then
            SUBSET_ARG=""
        else
            SUBSET_ARG="--subset $custom_subset"
        fi
        
        # Few-shot 파라미터 설정
        if [ "$num_few_shot" -gt 0 ]; then
            DATASET_PARAMS="{\"num_few_shot\": $num_few_shot, \"few_shot_split\": \"$few_shot_split\"}"
            echo -e "${YELLOW}$num_few_shot-shot 설정 (split: $few_shot_split)으로 평가합니다.${NC}"
            DATASET_PARAMS_ARG="--dataset_params '$DATASET_PARAMS'"
        else
            DATASET_PARAMS_ARG=""
        fi
        
        EVAL_CMD="python -m llm_eval.evaluator \
            --model huggingface \
            --dataset $DATASET \
            $SUBSET_ARG \
            --split $SPLIT \
            --model_params '{\"model_name_or_path\": \"$MODEL_PATH\", \"device\": \"$DEVICE\", \"max_new_tokens\": $max_tokens, \"batch_size\": $batch_size}' \
            $DATASET_PARAMS_ARG \
            --evaluation_method $eval_method \
            --output_file $OUTPUT_FILE"
        ;;
    *)
        echo -e "${RED}잘못된 선택입니다.${NC}"
        exit 1
        ;;
esac

# 환경 변수 설정 (로컬 경로 설정 파일 사용)
export HRET_LOCAL_PATHS_CONFIG="local_paths_config.yaml"

# 프로젝트 루트 디렉토리를 PYTHONPATH에 추가
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 평가 실행
echo -e "\n${GREEN}평가 실행 중...${NC}"
echo -e "${YELLOW}명령어:${NC}"
echo "$EVAL_CMD"
echo ""

# 실행
eval $EVAL_CMD

# 결과 확인
if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}평가 완료!${NC}"
    echo -e "결과 파일: ${YELLOW}$OUTPUT_FILE${NC}"
    
    # 결과 파일이 있으면 간단한 요약 출력
    if [ -f "$OUTPUT_FILE" ]; then
        echo -e "\n${GREEN}평가 결과 요약:${NC}"
        python -c "
import json
with open('$OUTPUT_FILE', 'r', encoding='utf-8') as f:
    results = json.load(f)
    if 'accuracy' in results:
        print(f'정확도: {results[\"accuracy\"]:.2%}')
    if 'subset_scores' in results:
        print('\\nSubset별 점수:')
        for subset, score in results['subset_scores'].items():
            print(f'  {subset}: {score:.2%}')
"
    fi
else
    echo -e "\n${RED}평가 실행 중 오류가 발생했습니다.${NC}"
    exit 1
fi