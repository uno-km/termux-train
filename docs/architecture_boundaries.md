# termux-train 아키텍처 경계 및 불변 설계 원칙 (Architectural Boundaries)

## 📌 1. 라이브러리 책임 영역 (Core Library Scope)

`termux-train`은 안드로이드/Termux 및 제한된 임베디드 리소스 환경에서 동작하는 **순수 수학적 딥러닝 훈련·추론 엔진(Pure ML Engine)**입니다.

- **포함 영역 (In-Scope)**:
  - 텐서(Tensor) 및 N차원 연산
  - 오토그래드 연산 그래프(DAG) 및 역전파(Backpropagation)
  - 신경망 모듈(Linear, Embedding, LayerNorm, Attention, Transformer, LoRA)
  - 옵티마이저(SGD, Adam, AdamW) 및 손실함수(CrossEntropy, MSE, BCEWithLogits)
  - 원자적 SafeTensors / JSON 체크포인트 직렬화 및 롤백 안전성
  - INT8 가중치 양자화 및 MMap 대용량 토큰 스트리밍

---

## 🚫 2. 시스템/하드웨어 경계 (Explicit Out-of-Scope)

다음 항목들은 **단말 OS(Android), Termux 환경 데몬, 또는 사용자의 관리 영역**이며, 라이브러리 코어에 포함되지 않습니다:

1. **배터리 잔량 감시 및 충전기 연동 (SCRUM-303)**:
   - `termux-battery-status` API를 라이브러리가 직접 폴링하지 않습니다.
   - 단말의 전원 관리 및 충전 상태에 따른 스크립트 실행은 사용자의 셸 스크립트나 crontab/daemon의 역할입니다.

2. **커널 발열 센서(Thermal Zone) 감시 및 강제 쿨다운 (SCRUM-304)**:
   - `/sys/class/thermal/` 직접 접근이나 CPU 주파수 스케일링 간섭을 배제합니다.
   - 안드로이드 커널의 CPU Thermal Governor 및 사용자의 프로세스 우선순위(nice) 설정을 따릅니다.

3. **시스템 전체 RAM OOM 감시 및 프로세스 간섭 (SCRUM-305)**:
   - `/proc/meminfo` 폴링을 통한 외부 프로세스 감시를 수행하지 않습니다.
   - 라이브러리는 **자체 힙 메모리 최소화(Zero-Allocation, In-place, Generator Streaming)**에만 집중하며, 프로세스 생명주기는 안드로이드 LMK(Low Memory Killer)에 맡깁니다.

---

## 🔒 3. 설계 효과

- **보안/권한 격리**: 안드로이드 특수 권한(`READ_PRIVILEGED_PHONE_STATE`, `BATTERY_STATS` 등)이나 Termux:API 의존성이 전혀 없는 **100% 독립적인 순수 파이썬/NumPy 라이브러리**로 유지됩니다.
- **플랫폼 독립성**: Linux, macOS, Windows, Android Termux 어디서나 동일한 동작을 보장합니다.
- **코드 경량성 (Zero-Bloatware)**: 불필요한 시스템 감시 백그라운드 스레드 및 I/O 오버헤드가 발생하지 않습니다.
