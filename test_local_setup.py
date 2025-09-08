#!/usr/bin/env python3
"""
로컬 모델과 데이터셋 설정을 테스트하는 스크립트
"""

import os
import sys
import json

# 프로젝트 경로를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_local_paths():
    """로컬 경로 설정 테스트"""
    print("=" * 60)
    print("로컬 경로 설정 테스트")
    print("=" * 60)
    
    # 1. PathResolver 테스트
    try:
        from llm_eval.utils.path_resolver import path_resolver
        print("\n✓ PathResolver 모듈 로드 성공")
        
        # 설정 확인
        config = path_resolver.get_config()
        if config:
            print(f"✓ 로컬 경로 설정 파일 로드됨")
            print(f"  - 모델 매핑 수: {len(config.get('models', {}))}")
            print(f"  - 데이터셋 매핑 수: {len(config.get('datasets', {}))}")
        else:
            print("⚠ 로컬 경로 설정 파일이 없음 (기본값 사용)")
        
        # 경로 변환 테스트
        test_model = "EleutherAI/pythia-160m"
        resolved = path_resolver.resolve_model_path(test_model)
        print(f"\n모델 경로 변환 테스트:")
        print(f"  원본: {test_model}")
        print(f"  변환: {resolved}")
        if os.path.exists(resolved):
            print(f"  ✓ 로컬 경로 존재함")
        else:
            print(f"  ⚠ 로컬 경로 아직 없음 (다운로드 필요)")
        
        test_dataset = "HAERAE-HUB/KMMLU"
        resolved = path_resolver.resolve_dataset_path(test_dataset)
        print(f"\n데이터셋 경로 변환 테스트:")
        print(f"  원본: {test_dataset}")
        print(f"  변환: {resolved}")
        if os.path.exists(resolved):
            print(f"  ✓ 로컬 경로 존재함")
        else:
            print(f"  ⚠ 로컬 경로 아직 없음 (다운로드 필요)")
            
    except ImportError as e:
        print(f"\n❌ PathResolver 모듈 로드 실패: {e}")
        return False
    
    # 2. 디렉토리 구조 확인
    print("\n" + "=" * 60)
    print("디렉토리 구조 확인")
    print("=" * 60)
    
    dirs_to_check = ["models", "datasets"]
    for dir_name in dirs_to_check:
        if os.path.exists(dir_name):
            print(f"\n✓ {dir_name}/ 디렉토리 존재")
            # 하위 디렉토리 확인
            subdirs = [d for d in os.listdir(dir_name) if os.path.isdir(os.path.join(dir_name, d))]
            if subdirs:
                print(f"  하위 디렉토리:")
                for subdir in subdirs:
                    print(f"    - {subdir}/")
            else:
                print(f"  (비어있음)")
        else:
            print(f"\n⚠ {dir_name}/ 디렉토리 없음")
    
    # 3. 모델 테스트 (다운로드된 경우)
    print("\n" + "=" * 60)
    print("모델 로드 테스트")
    print("=" * 60)
    
    model_path = "./models/pythia-160m"
    if os.path.exists(model_path):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            print(f"\n{model_path} 로드 테스트:")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(model_path)
            
            # 간단한 생성 테스트
            text = "안녕하세요"
            inputs = tokenizer(text, return_tensors="pt")
            outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            print(f"  ✓ 모델 로드 성공")
            print(f"  입력: {text}")
            print(f"  생성: {generated}")
            
        except Exception as e:
            print(f"  ❌ 모델 로드 실패: {e}")
    else:
        print(f"\n⚠ {model_path} 아직 다운로드되지 않음")
    
    # 4. 데이터셋 테스트 (다운로드된 경우)
    print("\n" + "=" * 60)
    print("데이터셋 로드 테스트")
    print("=" * 60)
    
    dataset_path = "./datasets/KMMLU"
    if os.path.exists(dataset_path):
        try:
            # dataset_info.json 파일 확인
            info_file = os.path.join(dataset_path, "dataset_info.json")
            if os.path.exists(info_file):
                with open(info_file, 'r', encoding='utf-8') as f:
                    info = json.load(f)
                print(f"\n{dataset_path} 정보:")
                print(f"  ✓ 데이터셋 ID: {info.get('dataset_id')}")
                print(f"  ✓ Subset 수: {info.get('num_subsets')}")
                print(f"  ✓ 검증된 샘플: {info.get('verified_samples')}")
            
            # 실제 데이터 로드 테스트
            from datasets import load_dataset
            print(f"\n데이터 로드 테스트:")
            dataset = load_dataset(dataset_path, "Accounting", split="test")
            print(f"  ✓ Accounting subset 로드 성공")
            print(f"  샘플 수: {len(dataset)}")
            if len(dataset) > 0:
                print(f"  첫 번째 샘플 키: {list(dataset[0].keys())}")
                
        except Exception as e:
            print(f"  ❌ 데이터셋 로드 실패: {e}")
    else:
        print(f"\n⚠ {dataset_path} 아직 다운로드되지 않음")
    
    # 5. 통합 평가 테스트
    print("\n" + "=" * 60)
    print("통합 평가 파이프라인 테스트")
    print("=" * 60)
    
    if os.path.exists(model_path) and os.path.exists(dataset_path):
        try:
            from llm_eval.evaluator import Evaluator
            
            print("\nEvaluator로 간단한 평가 실행:")
            evaluator = Evaluator()
            
            # 매우 작은 샘플로 테스트
            results = evaluator.run(
                model="huggingface",
                dataset="kmmlu",
                subset=["Accounting"],  # 한 개 subset만
                split="test",
                model_params={
                    "model_name_or_path": "EleutherAI/pythia-160m",
                    "max_new_tokens": 10,
                    "device": "cpu"
                },
                dataset_params={
                    "num_samples": 2  # 2개 샘플만 테스트
                },
                evaluation_method="string_match"
            )
            
            print("  ✓ 평가 완료!")
            print(f"  결과: {results.get('accuracy', 'N/A')}")
            
        except Exception as e:
            print(f"  ❌ 통합 평가 실패: {e}")
    else:
        print("\n⚠ 모델 또는 데이터셋이 아직 준비되지 않음")
    
    print("\n" + "=" * 60)
    print("테스트 완료")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    test_local_paths()