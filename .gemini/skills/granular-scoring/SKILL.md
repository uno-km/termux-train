---
name: granular-scoring
description: >-
  termux-train 프로젝트 전용 0점 기준 세부 채점 테스트 프로토콜 스킬.
  모든 모듈/함수 검증 시 0.0점에서 시작하여 수치 정밀도, 지연시간(ms), 모바일 힙 메모리 안전성을 합산 채점하여 스코어카드를 발행한다.
---

# termux-train 0점 기준 세부 채점 테스트 프로토콜 (Granular Scoring Protocol)

## 📌 프로젝트 채점 기준 (Pillars of Integrity)

1. **Pillar 1: Autograd DAG & 수치 무결성 (20점 만점)**
   - Analytical vs Numerical Gradient 오차 한도 내 일치 (+5.0점)
   - IEEE 754 LogSumExp NaN/Inf 무손실 방어 (+5.0점)
   - Fused CrossEntropyLoss 정수 타겟 계약 준수 (+5.0점)
   - Multi-Axis 튜플 Unbroadcasting 역전파 (+5.0점)

2. **Pillar 2: Transformer & RoPE 아키텍처 (20점 만점)**
   - Native RoPE 쿼리/키 회전 및 초장문 외삽 (+5.0점)
   - Universal Causal Trapezoid 마스킹 (청크 프리필 지원) (+5.0점)
   - 증분 KV Caching 추론 결정론적 일치 (+5.0점)
   - Weight Tying 파라미터 및 메모리 공유 (+5.0점)

3. **Pillar 3: 메모리 & 할당 효율성 (20점 만점)**
   - Causal Mask 영구 캐싱 및 Zero-Allocation 슬라이싱 (+5.0점)
   - Embedding C 레벨 벡터화 바운드 체크 및 Scatter-Add (+5.0점)
   - INT8 QuantizedLinear 무할당 행렬곱 (+5.0점)
   - SafeTensors C 레벨 tobytes/frombuffer 제로-카피 직렬화 (+5.0점)

4. **Pillar 4: 성능 & 지연시간 벤치마크 (20점 만점)**
   - Transformer 훈련 스텝 지연시간 기준치 준수 (+10.0점)
   - Autoregressive 토큰 생성 스텝당 지연시간 준수 (+10.0점)

5. **Pillar 5: 크래시 방어 & 체크포인트 안전성 (20점 만점)**
   - SafeTensors F32, I64, BOOL 비트 완전 일치 (+5.0점)
   - MMapTokenDataset 스트리밍 및 파일 소거 (+5.0점)
   - 빈 프롬프트 및 비정상 입력 방어 (+5.0점)
   - 전역 그래디언트 노름 클리핑 (`clip_grad_norm_`) (+5.0점)

## 🚀 채점 실행 방법
- CLI 스코어카드 러너: `py -3 scripts/run_audit_scoring.py`
- Pytest 개별 채점: `py -3 -m pytest tests/test_audit_scorecard.py -s -v`
