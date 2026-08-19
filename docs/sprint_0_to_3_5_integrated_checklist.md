# 🚀 문서 2: Sprint 0 ~ 3.5 통합 검증 체크리스트

> **목적**: 프로젝트 전체가 기획/설계 원칙에 맞추어 안정적으로 누적되었는지 검증하는 마스터 통합 체크리스트.

---

### 📦 Sprint 0. 프로젝트 인프라 및 거버넌스
- [x] 공식 배포명은 `termux-train`으로 통일했는가?
- [x] 공식 Python import package는 `termux_train`으로 통일했는가?
- [x] 구 패키지명 `termux_torch`와 legacy alias를 완전히 제거했는가?
- [x] `pyproject.toml`이 정상 등록되었는가?
- [x] `pip install -e .`가 정상 지원되는가?
- [x] `README.md`가 5대 철학 및 최신 Quickstart를 반영하는가?
- [x] `LICENSE` (Apache-2.0)가 존재하는가?
- [x] `.gitignore`가 토큰, 캐시, 빌드 산출물을 완벽 차단하는가?
- [x] `scripts/diagnose_termux.py`가 존재하는가?
- [x] `scripts/test_library_matrix.py`가 존재하는가?
- [x] Host PC matrix와 Android Termux matrix가 명확히 분리되었는가?

---

### 🧱 Sprint 1. Tensor Core & Pluggable Backend
- [x] `Tensor(data)` 생성이 되는가?
- [x] scalar shape가 `()`인가?
- [x] vector shape가 `(n,)`인가?
- [x] matrix shape가 `(m, n)`인가?
- [x] ND nested list shape 추론이 되는가?
- [x] ragged list를 `ValueError`로 잡는가?
- [x] `ndim`이 정확한가?
- [x] `flatten()`이 되는가?
- [x] `reshape()`가 되는가?
- [x] `zeros()`, `ones()`, `zeros_like()`, `ones_like()`가 되는가?
- [x] `__repr__`이 디버깅 가능하게 출력되는가?
- [x] `set_backend("python")`이 되는가?
- [x] `set_backend("numpy")`가 되는가?
- [x] `set_backend("auto")`가 되는가?
- [x] NumPy가 없어도 Pure-Python 백엔드로 100% 동일하게 동작하는가?

---

### ⚙️ Sprint 2. Dynamic DAG Autograd Engine & Ops
- [x] `+` forward/backward 되는가?
- [x] `-` forward/backward 되는가?
- [x] `*` forward/backward 되는가?
- [x] `/` forward/backward 되는가?
- [x] `**` forward/backward 되는가?
- [x] `neg` forward/backward 되는가?
- [x] `sum()` forward/backward 되는가?
- [x] `mean()` forward/backward 되는가?
- [x] `relu()` forward/backward 되는가?
- [x] `sigmoid()` forward/backward 되는가?
- [x] `tanh()` forward/backward 되는가?
- [x] `@` (matmul) forward/backward 되는가? ($dA = dY @ B^T, dB = A^T @ dY$)
- [x] `.T` transpose 되는가?
- [x] 다차원 브로드캐스팅 forward가 되는가?
- [x] `unbroadcast` 그래디언트 축소 전파가 되는가?
- [x] 그래디언트 누적(Grad Accumulation)이 되는가? ($y = x + x + x \rightarrow grad=3.0$)
- [x] DFS 위상정렬(Topological Sort)이 순서를 보장하는가?
- [x] 유한 차분법 수치 미분(`gradcheck`)을 전수 통과하는가?
- [x] Python 백엔드와 NumPy 백엔드 연산 결과가 일치하는가?

---

### 🧠 Sprint 3. Neural Network Mini Framework
- [x] `Parameter`가 `Tensor` 기반 서브클래스인가?
- [x] `Parameter.requires_grad=True`가 기본인가?
- [x] `Module.__setattr__`가 Parameter를 자동 등록하는가?
- [x] `Module.__setattr__`가 submodule을 자동 등록하는가?
- [x] `parameters()`가 재귀적으로 모든 파라미터를 모으는가?
- [x] `named_parameters()`가 고유하고 안정적인 이름을 반환하는가?
- [x] `train()` / `eval()` 모드 토글이 되는가?
- [x] `Linear(in, out)` forward/backward가 되는가?
- [x] `Linear` 가중치/바이어스 형상이 일관적인가? (`weight: (in, out), bias: (1, out)`)
- [x] `Sequential`이 레이어를 순서대로 forward 체이닝하는가?
- [x] `Sequential.parameters()`가 전체 가중치를 수집하는가?
- [x] `ReLU`, `Sigmoid`, `Tanh` 모듈이 작동하는가?
- [x] `mse_loss` forward/backward가 되는가?
- [x] `MSELoss` 모듈 래퍼가 작동하는가?
- [x] `examples/02_nn_forward_backward.py`가 에러 없이 실행되는가?

---

### 🛡️ Sprint 3.5. 안정화 및 하드닝
- [x] `zero_grad(set_to_none=True)` 메모리 해제 기본 정책 확정
- [x] scalar Tensor는 gradient 없이 `backward()`를 실행할 수 있는가?
- [x] non-scalar Tensor는 기본적으로 명시적 gradient를 요구하는가?
- [x] non-scalar Tensor에서 `backward(gradient=...)`를 지원하는가?
- [x] `allow_implicit_grad=True`를 명시한 경우에만 ones-like seed를 사용하는가?
- [x] explicit gradient shape mismatch를 `RuntimeError`로 차단하는가?
- [x] `state_dict()` deep copy 불변성 및 `load_state_dict` 형상/키 검증
- [x] `bce_loss` log-clamp 수치 안정성 필터링 적용
- [x] `clamp` / `log` / `clip` 텐서 연산자 구현 및 미분 검증
- [x] Sprint 4 진입 게이트 Tests 전수 통과
