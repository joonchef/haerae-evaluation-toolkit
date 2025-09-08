#!/usr/bin/env python3
"""
빠른 평가를 위한 스크립트 (샘플 수 제한)
"""

import os
import sys
import json
from datetime import datetime

# 프로젝트 경로를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def evaluate_quick():
    """작은 샘플로 빠른 평가 실행"""
    
    print("=" * 60)
    print("Pythia-160M 빠른 평가 (5개 샘플)")
    print("=" * 60)
    
    try:
        from llm_eval.evaluator import Evaluator
        print("[OK] Evaluator 모듈 로드 성공")
    except ImportError as e:
        print(f"[X] Evaluator 모듈 로드 실패: {e}")
        return False
    
    # 평가 설정 (5개 샘플만)
    config = {
        "model": "huggingface",
        "dataset": "kmmlu",
        "subset": ["Accounting"],
        "split": "test",
        "model_params": {
            "model_name_or_path": "EleutherAI/pythia-160m",
            "device": "cpu",
            "max_new_tokens": 10,  # 더 짧게
            "batch_size": 1,
            "temperature": 0.0,
            "do_sample": False
        },
        "dataset_params": {
            "streaming": False,
            "select": [0, 1, 2, 3, 4]  # 처음 5개만 선택
        },
        "evaluation_method": "string_match"
    }
    
    print("\n평가 설정:")
    print(f"  모델: {config['model_params']['model_name_or_path']}")
    print(f"  데이터셋: {config['dataset']}")
    print(f"  샘플 수: 5개")
    print(f"  평가 방법: {config['evaluation_method']}")
    
    print("\n평가 시작...")
    start_time = datetime.now()
    
    try:
        # Evaluator 생성
        evaluator = Evaluator()
        
        # 평가 실행
        results = evaluator.run(**config)
        
        end_time = datetime.now()
        elapsed_time = (end_time - start_time).total_seconds()
        
        print(f"\n평가 완료! (소요시간: {elapsed_time:.2f}초)")
        
        # 결과 출력
        print("\n평가 결과:")
        print("=" * 40)
        
        if isinstance(results, dict):
            # 주요 메트릭 출력
            accuracy = results.get('accuracy', results.get('score', 'N/A'))
            print(f"  정확도 (Accuracy): {accuracy}")
            
            # 결과를 JSON 파일로 저장
            output_file = "results_quick.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n결과가 {output_file}에 저장되었습니다.")
            
            # 예측 결과 상세 출력
            if 'predictions' in results and results['predictions']:
                print("\n예측 결과 상세:")
                print("-" * 40)
                for i, pred in enumerate(results['predictions']):
                    print(f"\n샘플 {i+1}:")
                    if isinstance(pred, dict):
                        if 'question' in pred:
                            print(f"  질문: {pred['question'][:50]}...")
                        if 'reference' in pred:
                            print(f"  정답: {pred['reference']}")
                        if 'prediction' in pred:
                            print(f"  예측: {pred['prediction']}")
                        if 'correct' in pred:
                            print(f"  맞춤: {'O' if pred['correct'] else 'X'}")
                    else:
                        print(f"  {pred}")
        else:
            print(f"  결과: {results}")
            
        return True
        
    except Exception as e:
        print(f"\n[X] 평가 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = evaluate_quick()
    exit(0 if success else 1)