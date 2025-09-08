#!/usr/bin/env python3
"""
Pythia-160M 모델을 로컬에 다운로드하는 스크립트
"""

import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
import torch

def download_pythia_160m():
    """Pythia-160M 모델과 토크나이저를 다운로드"""
    
    model_id = "EleutherAI/pythia-160m"
    local_dir = "./models/pythia-160m"
    
    print(f"모델 다운로드 시작: {model_id}")
    print(f"저장 위치: {local_dir}")
    
    # 디렉토리 생성
    os.makedirs(local_dir, exist_ok=True)
    
    try:
        # 방법 1: snapshot_download를 사용하여 전체 저장소 다운로드
        print("\n전체 모델 파일 다운로드 중...")
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print("✓ 모델 파일 다운로드 완료")
        
        # 방법 2: 모델과 토크나이저를 명시적으로 로드하고 저장 (검증용)
        print("\n모델 로드 및 검증 중...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True
        )
        
        # 토크나이저와 모델을 로컬에 저장
        tokenizer.save_pretrained(local_dir)
        model.save_pretrained(local_dir)
        print("✓ 모델 검증 및 저장 완료")
        
        # 다운로드된 파일 확인
        print("\n다운로드된 파일:")
        for root, dirs, files in os.walk(local_dir):
            level = root.replace(local_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f'{indent}{os.path.basename(root)}/')
            subindent = ' ' * 2 * (level + 1)
            for file in files[:10]:  # 처음 10개 파일만 표시
                print(f'{subindent}{file}')
            if len(files) > 10:
                print(f'{subindent}... 외 {len(files)-10}개 파일')
        
        # 모델 크기 확인
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(local_dir):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
        
        print(f"\n총 다운로드 크기: {total_size / (1024**3):.2f} GB")
        print("\n✅ Pythia-160M 모델 다운로드 완료!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        return False

if __name__ == "__main__":
    success = download_pythia_160m()
    exit(0 if success else 1)