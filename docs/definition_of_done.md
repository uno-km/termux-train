# 📋 Definition of Done (DoD) & Engineering Standards

본 문서는 `termux-train` (AMEVA-Termux) 프로젝트의 모든 티켓(`TT-XXX`)이 완료(`Done`) 상태로 전환되기 위해 충족해야 하는 필수 완료 기준입니다.

---

## 🎯 Common Definition of Done (공통 완료 기준)

모든 티켓은 아래 6가지 기준을 100% 통과해야만 `Done`으로 전이할 수 있습니다:

1. **코드 구현 및 린트 검증 (Code Implementation)**:
   - Python 3.9 ~ 3.12 표준 문법 및 타입 힌트 준수
   - 외부 무거운 바이너리(PyTorch 등) 임의 의존성 추가 절대 금지 (Pure Python fallback 필수)
2. **단위 및 수치 검증 테스트 (Unit & Numerical Tests)**:
   - 해당 기능에 대한 단위 테스트(`tests/test_*.py`) 작성
   - Autograd 연산의 경우 중앙 유한차분 기반 `gradcheck` (기본 검증 설정: `eps=1e-3`, `atol=1e-2`, `rtol=1e-2`) 통과
3. **Android Termux 실기기 호환성 (On-Device Compatibility)**:
   - Termux arm64 환경에서 경고/에러 없이 실행 완료
   - 메모리 누수 및 비정상 프로세스 Kill 없음
4. **문서화 및 예제 반영 (Documentation & Examples)**:
   - `README.md` 또는 `docs/`에 변경사항 및 API 시그니처 반영
   - 필요 시 `examples/` 디렉토리에 실행 가능한 데모 스크립트 작성
5. **실패 및 예외 케이스 처리 (Failure Handling)**:
   - 텐서 shape 불일치, 비정상 타입, 자원 부족 등 비정상 입력에 대한 명확한 예외 처리
6. **Jira 실시간 동기화 (Jira Real-time Sync)**:
   - 티켓 시작 시 `In Progress` 전환
   - 코드 수정 내역 및 수치 지표(Loss, 메모리, 실행 시간) 실시간 코멘트 기록
   - DoD 확인 후 `Done` 전환

---

## 📱 기기별/환경별 테스트 프로토콜

- **1차 (로컬 PC 개발 환경)**: Pure Python 3.10+ 단위 테스트 통과
- **2차 (Android Termux 기기)**: SSH 8022 세션 상에서 실기기 런타임 및 자원 모니터링 검증
