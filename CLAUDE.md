# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 언어 사용 지침 (Language Guidelines)

- **한국어 우선**: 코드 작성과 기술 용어를 제외한 모든 커뮤니케이션은 한국어를 사용합니다.
- **주석**: 코드 내 주석은 한국어로 작성합니다.
- **커밋 메시지**: 한국어로 작성하되, 기술 용어는 영어를 그대로 사용합니다.
- **문서화**: README나 문서 작성 시 한국어를 기본으로 합니다.

## Project Overview

Haerae-Evaluation-Toolkit (HRET) is an open-source Python library for evaluating Large Language Models (LLMs) with a focus on Korean language capabilities. The toolkit provides a pluggable architecture with support for multiple evaluation methods, datasets, and model backends.

## Key Commands

### Development Setup
```bash
# Install dependencies (recommended: use uv for speed)
uv pip install -r requirements.txt

# Install development dependencies
pip install -e ".[dev]"

# Install test dependencies
pip install -e ".[test]"
```

### Testing
```bash
# Run all tests
pytest llm_eval/test/

# Run specific test file
pytest llm_eval/test/test_evaluator_config.py

# Run with verbose output
pytest -v llm_eval/test/
```

### Code Quality
```bash
# Run pre-commit hooks manually
pre-commit run --all-files

# Individual linters/formatters
flake8 llm_eval/
autopep8 --max-line-length=80 -r llm_eval/
isort llm_eval/
mypy llm_eval/
```

## Architecture

### Core Components

1. **Evaluator API** (`llm_eval/evaluator.py`)
   - Main entry point for evaluation pipelines
   - Orchestrates dataset loading, model inference, scaling methods, and evaluation
   - Supports both programmatic and CLI usage

2. **HRET API** (`llm_eval/hret.py`)
   - Decorator-based MLOps-friendly interface
   - Provides `@hret.evaluate`, `@hret.benchmark` decorators
   - Integrates with MLflow, Weights & Biases for experiment tracking

3. **Registry Pattern**
   - All components (datasets, models, scaling methods, evaluations) use decorator-based registration
   - Example: `@register_dataset("dataset_name")` makes components discoverable

### Component Structure

```
llm_eval/
├── datasets/          # Dataset loaders (KMMLU, HAE-RAE Bench, etc.)
│   ├── base.py       # BaseDataset abstract class
│   └── *.py          # Individual dataset implementations
├── models/           # Model backends
│   ├── base.py       # BaseModel abstract class
│   ├── huggingface_backend.py
│   ├── litellm_backend.py
│   ├── openai_backend.py
│   └── *_judge.py    # Judge model implementations
├── scaling_methods/  # Inference-time scaling techniques
│   ├── base.py       # BaseScalingMethod abstract class
│   ├── best_of_n.py
│   ├── beam_search.py
│   └── self_consistency.py
├── evaluation/       # Evaluation methods
│   ├── base.py       # BaseEvaluationMethod abstract class
│   ├── string_match.py
│   ├── partial_match.py
│   ├── llm_judge.py
│   └── log_prob.py
└── utils/           # Shared utilities
```

### Key Design Patterns

1. **Abstract Base Classes**: All major components inherit from abstract base classes (`BaseDataset`, `BaseModel`, `BaseScalingMethod`, `BaseEvaluationMethod`)

2. **MultiModel Architecture**: Supports separate models for generation, judging, and reward scoring through the `MultiModel` class

3. **Pipeline Configuration**: YAML-based configuration for complex evaluation pipelines

4. **Async Support**: Models support both sync and async operations for better performance

### Adding New Components

To add a new dataset:
```python
from llm_eval.datasets import BaseDataset, register_dataset

@register_dataset("my_dataset")
class MyDataset(BaseDataset):
    def load_data(self, split="test", **kwargs):
        # Implementation
        pass
```

Similar patterns apply for models, scaling methods, and evaluation methods.

## Important Notes

- The project focuses on Korean LLM evaluation but supports multilingual capabilities
- Uses LiteLLM for unified API access across different model providers
- Supports both local models (Hugging Face) and API-based models (OpenAI, Claude)
- Pre-commit hooks are configured for code quality (flake8, autopep8, isort, mypy)
- Test files are located in `llm_eval/test/` and use pytest