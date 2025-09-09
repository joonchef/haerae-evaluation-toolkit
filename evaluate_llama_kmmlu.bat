@echo off
REM LLaMA 3.2 8B 모델과 KMMLU 데이터셋 평가 스크립트 (Windows)

echo ==================================================
echo LLaMA 3.2 8B - KMMLU 평가 시작
echo ==================================================

REM 기본 설정
set MODEL_PATH=meta-llama/Llama-3.2-8B-Instruct
set DATASET=kmmlu
set SPLIT=test
set DEVICE=cuda
set TIMESTAMP=%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set TIMESTAMP=%TIMESTAMP: =0%
set OUTPUT_FILE=results_llama_kmmlu_%TIMESTAMP%.json

REM 평가 옵션 선택
echo.
echo 평가 옵션을 선택하세요:
echo 1. 빠른 테스트 (Accounting subset만, 5개 샘플)
echo 2. 단일 subset 전체 평가
echo 3. 모든 subset 평가 (전체 KMMLU)
echo 4. 사용자 정의 설정
set /p choice="선택 (1-4): "

if "%choice%"=="1" goto quick_test
if "%choice%"=="2" goto single_subset
if "%choice%"=="3" goto full_evaluation
if "%choice%"=="4" goto custom_settings
goto invalid_choice

:quick_test
echo.
echo 빠른 테스트 모드
set SUBSET=Accounting
set NUM_SAMPLES=5
set EVAL_CMD=python llm_eval/evaluator.py ^
    --model huggingface ^
    --dataset %DATASET% ^
    --subset %SUBSET% ^
    --split %SPLIT% ^
    --model_params "{\"model_name_or_path\": \"%MODEL_PATH%\", \"device\": \"%DEVICE%\", \"max_new_tokens\": 128}" ^
    --dataset_params "{\"num_samples\": %NUM_SAMPLES%}" ^
    --evaluation_method string_match ^
    --output_file %OUTPUT_FILE%
goto execute

:single_subset
echo.
echo 사용 가능한 subset:
echo Accounting, Biology, Chemistry, Computer-Science, Economics,
echo Education, Law, Math, Psychology, Korean-History 등
set /p SUBSET="평가할 subset 이름 입력: "
set EVAL_CMD=python llm_eval/evaluator.py ^
    --model huggingface ^
    --dataset %DATASET% ^
    --subset %SUBSET% ^
    --split %SPLIT% ^
    --model_params "{\"model_name_or_path\": \"%MODEL_PATH%\", \"device\": \"%DEVICE%\", \"max_new_tokens\": 256}" ^
    --evaluation_method string_match ^
    --output_file %OUTPUT_FILE%
goto execute

:full_evaluation
echo.
echo 전체 KMMLU 평가 (45개 subset)
echo 경고: 시간이 오래 걸릴 수 있습니다!
set /p confirm="계속하시겠습니까? (y/n): "
if not "%confirm%"=="y" (
    echo 평가 취소됨
    exit /b 0
)
set EVAL_CMD=python llm_eval/evaluator.py ^
    --model huggingface ^
    --dataset %DATASET% ^
    --split %SPLIT% ^
    --model_params "{\"model_name_or_path\": \"%MODEL_PATH%\", \"device\": \"%DEVICE%\", \"max_new_tokens\": 256, \"batch_size\": 4}" ^
    --evaluation_method string_match ^
    --output_file %OUTPUT_FILE%
goto execute

:custom_settings
echo.
echo 사용자 정의 설정
set /p custom_model="모델 경로 (기본: %MODEL_PATH%): "
if not "%custom_model%"=="" set MODEL_PATH=%custom_model%

set /p custom_device="디바이스 (cuda/cpu, 기본: %DEVICE%): "
if not "%custom_device%"=="" set DEVICE=%custom_device%

set /p batch_size="배치 크기 (기본: 1): "
if "%batch_size%"=="" set batch_size=1

set /p max_tokens="최대 토큰 수 (기본: 256): "
if "%max_tokens%"=="" set max_tokens=256

set /p eval_method="평가 방법 (string_match/partial_match/llm_judge, 기본: string_match): "
if "%eval_method%"=="" set eval_method=string_match

set /p custom_subset="Subset (콤마로 구분, 전체는 Enter): "
if "%custom_subset%"=="" (
    set SUBSET_ARG=
) else (
    set SUBSET_ARG=--subset %custom_subset%
)

set EVAL_CMD=python llm_eval/evaluator.py ^
    --model huggingface ^
    --dataset %DATASET% ^
    %SUBSET_ARG% ^
    --split %SPLIT% ^
    --model_params "{\"model_name_or_path\": \"%MODEL_PATH%\", \"device\": \"%DEVICE%\", \"max_new_tokens\": %max_tokens%, \"batch_size\": %batch_size%}" ^
    --evaluation_method %eval_method% ^
    --output_file %OUTPUT_FILE%
goto execute

:invalid_choice
echo 잘못된 선택입니다.
exit /b 1

:execute
REM 환경 변수 설정 (로컬 경로 설정 파일 사용)
set HRET_LOCAL_PATHS_CONFIG=local_paths_config.yaml

REM 평가 실행
echo.
echo 평가 실행 중...
echo 명령어:
echo %EVAL_CMD%
echo.

REM 실행
%EVAL_CMD%

REM 결과 확인
if %errorlevel%==0 (
    echo.
    echo 평가 완료!
    echo 결과 파일: %OUTPUT_FILE%
    
    REM 결과 파일이 있으면 간단한 요약 출력
    if exist %OUTPUT_FILE% (
        echo.
        echo 평가 결과 요약:
        python -c "import json; f=open('%OUTPUT_FILE%','r',encoding='utf-8'); results=json.load(f); f.close(); print(f'정확도: {results.get(\"accuracy\", \"N/A\")}') if 'accuracy' in results else None; [print(f'  {k}: {v:.2%%}') for k,v in results.get('subset_scores', {}).items()]"
    )
) else (
    echo.
    echo 평가 실행 중 오류가 발생했습니다.
    exit /b 1
)