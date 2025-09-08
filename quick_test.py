#!/usr/bin/env python3
"""
빠른 테스트를 위한 간단한 평가 스크립트
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("빠른 테스트 시작...")

# 간단한 테스트
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd

# 모델 로드
model_path = "./models/pythia-160m"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# 데이터 로드 (첫 5개만)
csv_file = "./datasets/KMMLU/data/Accounting-test.csv"
df = pd.read_csv(csv_file, nrows=5)

print(f"\n테스트할 샘플 수: {len(df)}")

# 각 샘플에 대해 예측
correct = 0
for idx, row in df.iterrows():
    question = row['question']
    choices = [row['A'], row['B'], row['C'], row['D']]
    answer = row['answer']  # 정답 (A, B, C, D)
    
    # 프롬프트 생성
    prompt = f"다음 문제의 정답을 (A), (B), (C), (D) 중에서 선택하세요.\n\n"
    prompt += f"질문: {question}\n"
    prompt += f"(A) {choices[0]}\n"
    prompt += f"(B) {choices[1]}\n"
    prompt += f"(C) {choices[2]}\n"
    prompt += f"(D) {choices[3]}\n"
    prompt += "\n정답:"
    
    # 모델 예측
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(
        **inputs,
        max_new_tokens=10,
        temperature=0.0,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id
    )
    
    generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    # 예측 결과 추출 (첫 번째 (A), (B), (C), (D) 찾기)
    prediction = "N/A"
    for choice in ["(A)", "(B)", "(C)", "(D)"]:
        if choice in generated_text:
            prediction = choice[1]  # A, B, C, D만 추출
            break
    
    # 정답 확인
    is_correct = (prediction == answer)
    if is_correct:
        correct += 1
    
    print(f"\n샘플 {idx+1}:")
    print(f"  질문: {question[:50]}...")
    print(f"  정답: {answer}")
    print(f"  예측: {prediction}")
    print(f"  생성된 텍스트: {generated_text[:30]}...")
    print(f"  맞춤: {'O' if is_correct else 'X'}")

# 최종 결과
accuracy = (correct / len(df)) * 100
print(f"\n" + "=" * 40)
print(f"최종 정확도: {accuracy:.1f}% ({correct}/{len(df)})")
print("=" * 40)