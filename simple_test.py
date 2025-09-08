#!/usr/bin/env python3
"""
간단한 로컬 설정 테스트
"""

import os
import sys

# 프로젝트 경로를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_simple():
    print("=" * 60)
    print("로컬 모델 및 데이터셋 테스트")
    print("=" * 60)
    
    # 1. 디렉토리 확인
    print("\n1. 디렉토리 구조 확인:")
    if os.path.exists("models/pythia-160m"):
        print("  [OK] models/pythia-160m 존재")
        files = os.listdir("models/pythia-160m")
        print(f"    파일 수: {len(files)}")
        for f in files[:5]:
            print(f"    - {f}")
    else:
        print("  [X] models/pythia-160m 없음")
    
    if os.path.exists("datasets/KMMLU"):
        print("  [OK] datasets/KMMLU 존재")
        if os.path.exists("datasets/KMMLU/data"):
            data_files = os.listdir("datasets/KMMLU/data")
            print(f"    데이터 파일 수: {len(data_files)}")
            for f in data_files[:5]:
                print(f"    - {f}")
    else:
        print("  [X] datasets/KMMLU 없음")
    
    # 2. PathResolver 테스트 (직접 import)
    print("\n2. PathResolver 테스트:")
    try:
        # PathResolver만 직접 import
        from llm_eval.utils.path_resolver import path_resolver
        print("  [OK] PathResolver 로드 성공")
        
        # 경로 변환 테스트
        test_model = "EleutherAI/pythia-160m"
        resolved = path_resolver.resolve_model_path(test_model)
        print(f"  모델 경로 변환:")
        print(f"    원본: {test_model}")
        print(f"    변환: {resolved}")
        print(f"    존재: {'[OK]' if os.path.exists(resolved) else '[X]'}")
        
        test_dataset = "HAERAE-HUB/KMMLU"
        resolved = path_resolver.resolve_dataset_path(test_dataset)
        print(f"  데이터셋 경로 변환:")
        print(f"    원본: {test_dataset}")
        print(f"    변환: {resolved}")
        print(f"    존재: {'[OK]' if os.path.exists(resolved) else '[X]'}")
        
    except Exception as e:
        print(f"  [X] PathResolver 로드 실패: {e}")
    
    # 3. 모델 로드 테스트
    print("\n3. 모델 로드 테스트:")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model_path = "./models/pythia-160m"
        print(f"  경로: {model_path}")
        
        # 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print("  [OK] 토크나이저 로드 성공")
        
        # 모델 로드 (CPU에서 빠르게)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        print("  [OK] 모델 로드 성공")
        
        # 간단한 생성 테스트
        text = "Hello"
        inputs = tokenizer(text, return_tensors="pt")
        
        # 짧은 생성
        import torch
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"  생성 테스트:")
        print(f"    입력: {text}")
        print(f"    출력: {generated}")
        
    except Exception as e:
        print(f"  [X] 모델 로드 실패: {e}")
    
    # 4. 데이터셋 로드 테스트
    print("\n4. 데이터셋 로드 테스트:")
    try:
        from datasets import load_dataset
        
        dataset_path = "./datasets/KMMLU"
        print(f"  경로: {dataset_path}")
        
        # CSV 파일 직접 로드 시도
        import pandas as pd
        csv_file = os.path.join(dataset_path, "data", "Accounting-test.csv")
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            print(f"  [OK] CSV 로드 성공")
            print(f"    샘플 수: {len(df)}")
            print(f"    컬럼: {list(df.columns)}")
            if len(df) > 0:
                print(f"    첫 번째 질문: {df.iloc[0]['question'][:50]}...")
        else:
            print(f"  [X] CSV 파일 없음: {csv_file}")
            
    except Exception as e:
        print(f"  [X] 데이터셋 로드 실패: {e}")
    
    print("\n" + "=" * 60)
    print("테스트 완료!")
    print("=" * 60)

if __name__ == "__main__":
    test_simple()