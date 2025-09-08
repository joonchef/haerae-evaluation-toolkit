# PathResolver 유틸리티 클래스
# 온라인 모델/데이터셋 경로를 로컬 경로로 매핑

import os
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class PathResolver:
    """온라인 경로를 로컬 경로로 변환하는 유틸리티 클래스"""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._config is None:
            self._load_config()
    
    def _load_config(self):
        """설정 파일 로드"""
        # 환경변수에서 설정 파일 경로 가져오기
        config_path = os.environ.get('HRET_LOCAL_PATHS_CONFIG')
        
        # 기본 설정 파일 경로들
        default_paths = [
            'local_paths_config.yaml',
            'config/local_paths_config.yaml',
            os.path.join(os.path.expanduser('~'), '.hret', 'local_paths_config.yaml')
        ]
        
        # 설정 파일 찾기
        if config_path and os.path.exists(config_path):
            config_file = config_path
        else:
            config_file = None
            for path in default_paths:
                if os.path.exists(path):
                    config_file = path
                    break
        
        # 설정 파일 로드
        if config_file:
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    self._config = yaml.safe_load(f) or {}
                logger.info(f"로컬 경로 설정 파일 로드됨: {config_file}")
            except Exception as e:
                logger.warning(f"설정 파일 로드 실패: {e}")
                self._config = {}
        else:
            self._config = {}
        
        # 환경변수에서 루트 디렉토리 설정
        self._models_root = os.environ.get('HRET_MODELS_ROOT')
        self._datasets_root = os.environ.get('HRET_DATASETS_ROOT')
    
    def resolve_model_path(self, model_name_or_path: str) -> str:
        """모델 경로 변환
        
        Args:
            model_name_or_path: Hugging Face 모델 ID 또는 로컬 경로
            
        Returns:
            로컬 경로 (설정이 있으면) 또는 원본 경로
        """
        # 이미 로컬 경로인 경우
        if os.path.exists(model_name_or_path):
            return model_name_or_path
        
        # 설정 파일에서 매핑 찾기
        if self._config and 'models' in self._config:
            mapped_path = self._config['models'].get(model_name_or_path)
            if mapped_path:
                # 상대 경로인 경우 모델 루트 디렉토리와 결합
                if not os.path.isabs(mapped_path) and self._models_root:
                    mapped_path = os.path.join(self._models_root, mapped_path)
                
                # 경로 확장 (~ 등)
                mapped_path = os.path.expanduser(mapped_path)
                
                if os.path.exists(mapped_path):
                    logger.info(f"모델 경로 매핑: {model_name_or_path} -> {mapped_path}")
                    return mapped_path
                else:
                    logger.warning(f"매핑된 모델 경로가 존재하지 않음: {mapped_path}")
        
        # 모델 루트 디렉토리가 설정된 경우
        if self._models_root:
            # 모델 이름에서 조직명 제거 (예: "EleutherAI/polyglot-ko" -> "polyglot-ko")
            model_name = model_name_or_path.split('/')[-1]
            local_path = os.path.join(self._models_root, model_name)
            if os.path.exists(local_path):
                logger.info(f"모델 루트 디렉토리에서 찾음: {model_name_or_path} -> {local_path}")
                return local_path
        
        # 변환할 수 없으면 원본 반환
        return model_name_or_path
    
    def resolve_dataset_path(self, dataset_name: str) -> str:
        """데이터셋 경로 변환
        
        Args:
            dataset_name: Hugging Face 데이터셋 ID 또는 로컬 경로
            
        Returns:
            로컬 경로 (설정이 있으면) 또는 원본 경로
        """
        # 이미 로컬 경로인 경우
        if os.path.exists(dataset_name):
            return dataset_name
        
        # 설정 파일에서 매핑 찾기
        if self._config and 'datasets' in self._config:
            mapped_path = self._config['datasets'].get(dataset_name)
            if mapped_path:
                # 상대 경로인 경우 데이터셋 루트 디렉토리와 결합
                if not os.path.isabs(mapped_path) and self._datasets_root:
                    mapped_path = os.path.join(self._datasets_root, mapped_path)
                
                # 경로 확장 (~ 등)
                mapped_path = os.path.expanduser(mapped_path)
                
                if os.path.exists(mapped_path):
                    logger.info(f"데이터셋 경로 매핑: {dataset_name} -> {mapped_path}")
                    return mapped_path
                else:
                    logger.warning(f"매핑된 데이터셋 경로가 존재하지 않음: {mapped_path}")
        
        # 데이터셋 루트 디렉토리가 설정된 경우
        if self._datasets_root:
            # 데이터셋 이름에서 조직명 제거 (예: "HAERAE-HUB/KMMLU" -> "KMMLU")
            dataset_short_name = dataset_name.split('/')[-1]
            local_path = os.path.join(self._datasets_root, dataset_short_name)
            if os.path.exists(local_path):
                logger.info(f"데이터셋 루트 디렉토리에서 찾음: {dataset_name} -> {local_path}")
                return local_path
        
        # 변환할 수 없으면 원본 반환
        return dataset_name
    
    def is_local_path(self, path: str) -> bool:
        """경로가 로컬 경로인지 확인"""
        return os.path.exists(path)
    
    def get_config(self) -> Dict[str, Any]:
        """현재 설정 반환"""
        return self._config.copy() if self._config else {}
    
    def reload_config(self):
        """설정 파일 다시 로드"""
        self._config = None
        self._load_config()


# 싱글톤 인스턴스
path_resolver = PathResolver()