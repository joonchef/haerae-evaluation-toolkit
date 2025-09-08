#!/usr/bin/env python3
"""
KMMLU 데이터셋을 로컬에 다운로드하는 스크립트
"""

import os
import json
from datasets import load_dataset
from huggingface_hub import snapshot_download

def download_kmmlu():
    """KMMLU 데이터셋을 다운로드"""
    
    dataset_id = "HAERAE-HUB/KMMLU"
    local_dir = "./datasets/KMMLU"
    
    print(f"데이터셋 다운로드 시작: {dataset_id}")
    print(f"저장 위치: {local_dir}")
    
    # 디렉토리 생성
    os.makedirs(local_dir, exist_ok=True)
    
    # KMMLU의 모든 subset 목록
    subsets = [
        'Accounting', 'Agricultural-Sciences', 'Aviation-Engineering-and-Maintenance', 'Biology',
        'Chemical-Engineering', 'Chemistry', 'Civil-Engineering', 'Computer-Science', 'Construction',
        'Criminal-Law', 'Ecology', 'Economics', 'Education', 'Electrical-Engineering', 'Electronics-Engineering',
        'Energy-Management', 'Environmental-Science', 'Fashion', 'Food-Processing',
        'Gas-Technology-and-Engineering', 'Geomatics', 'Health', 'Industrial-Engineer',
        'Information-Technology', 'Interior-Architecture-and-Design', 'Law', 'Machine-Design-and-Manufacturing',
        'Management', 'Maritime-Engineering', 'Marketing', 'Materials-Engineering', 'Mechanical-Engineering',
        'Nondestructive-Testing', 'Patent', 'Political-Science-and-Sociology', 'Psychology', 'Public-Safety',
        'Railway-and-Automotive-Engineering', 'Real-Estate', 'Refrigerating-Machinery', 'Social-Welfare',
        'Taxation', 'Telecommunications-and-Wireless-Technology', 'Korean-History', 'Math'
    ]
    
    try:
        # 방법 1: snapshot_download로 전체 저장소 다운로드
        print("\n전체 데이터셋 파일 다운로드 중...")
        snapshot_download(
            repo_id=dataset_id,
            repo_type="dataset",
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print("✓ 데이터셋 파일 다운로드 완료")
        
        # 방법 2: 각 subset별로 데이터 로드 및 저장 (검증용)
        print("\n각 subset 데이터 검증 중...")
        total_samples = 0
        subset_info = {}
        
        for subset in subsets[:3]:  # 처음 3개만 테스트로 검증
            try:
                print(f"  - {subset} 로드 중...")
                dataset = load_dataset(dataset_id, subset, split="test")
                num_samples = len(dataset)
                total_samples += num_samples
                subset_info[subset] = num_samples
                print(f"    ✓ {subset}: {num_samples}개 샘플")
            except Exception as e:
                print(f"    ⚠ {subset} 로드 실패: {e}")
        
        # 정보 저장
        info_file = os.path.join(local_dir, "dataset_info.json")
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump({
                "dataset_id": dataset_id,
                "num_subsets": len(subsets),
                "subsets": subsets,
                "subset_samples": subset_info,
                "verified_samples": total_samples
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n검증된 샘플 수: {total_samples}개")
        
        # 다운로드된 파일 확인
        print("\n다운로드된 파일 구조:")
        for root, dirs, files in os.walk(local_dir):
            level = root.replace(local_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f'{indent}{os.path.basename(root)}/')
            subindent = ' ' * 2 * (level + 1)
            
            # 디렉토리 표시
            for dir_name in dirs[:5]:
                print(f'{subindent}{dir_name}/')
            if len(dirs) > 5:
                print(f'{subindent}... 외 {len(dirs)-5}개 디렉토리')
            
            # 파일 표시
            for file in files[:5]:
                print(f'{subindent}{file}')
            if len(files) > 5:
                print(f'{subindent}... 외 {len(files)-5}개 파일')
        
        # 데이터셋 크기 확인
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(local_dir):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
        
        print(f"\n총 다운로드 크기: {total_size / (1024**2):.2f} MB")
        print("\n✅ KMMLU 데이터셋 다운로드 완료!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        return False

if __name__ == "__main__":
    success = download_kmmlu()
    exit(0 if success else 1)