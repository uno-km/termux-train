# 🛡️ 문서 1: Sprint 3.5 전용 안정화 체크리스트

> **목적**: Sprint 4의 Optimizer(SGD, Adam, AdamW), Regression, XOR 학습으로 진입하기 전 gradient, backward, checkpoint, loss 안정성을 100% 잠그는 단계.

---

### 📋 3.5-A. zero_grad 정책 (RAM 누수 방어 & 라이프사이클)
- [x] **`Module.zero_grad(set_to_none=True)`가 기본값인가?**  
  👉 예, `Module.zero_grad(self, set_to_none: bool = True)` 및 `Tensor.zero_grad(self, set_to_none: bool = True)` 기본값 설정 완료.
- [x] **기본 호출 시 모든 parameter의 `grad is None`인가?**  
  👉 예, `test_zero_grad_policies`에서 `p.grad is None` 검증 완료.
- [x] **`zero_grad(set_to_none=False)` 호출 시 zero Tensor가 생성되는가?**  
  👉 예, `zeros_like` 텐서가 `param.grad`에 안전하게 할당됨.
- [x] **하위 모듈 parameter까지 재귀 처리되는가?**  
  👉 예, `Module.parameters(recurse=True)`를 순회하여 하위 서브모듈 파라미터까지 일괄 적용.
- [x] **`requires_grad=False` parameter에 불필요한 grad를 만들지 않는가?**  
  👉 예, `if not self.requires_grad: self.grad = None; return` 가드 적용.
- [x] **Optimizer.step() 설계에서 `grad is None`이면 skip할 계획인가?**  
  👉 예, `if p.grad is None: continue` 아키텍처 확정.

---

### 📋 3.5-B. backward non-scalar 정책 (역전파 의미 확정)
- [x] **scalar Tensor는 `backward()`가 gradient 없이 성공하는가?**  
  👉 예, scalar loss에서 `loss.backward()` 단독 호출 시 정상 작동.
- [x] **non-scalar Tensor는 기본적으로 명시적 gradient를 요구하는가?**  
  👉 예, non-scalar Tensor는 기본적으로 `backward(gradient=...)`를 요구하며 생략 시 RuntimeError 발생.
- [x] **`allow_implicit_grad=True` 명시 시에만 ones-like seed를 허용하는가?**  
  👉 예, 의도적인 경우에만 `backward(allow_implicit_grad=True)`를 통해 ones-like seed 지원.
- [x] **non-scalar Tensor는 `backward(gradient=...)`로 성공하는가?**  
  👉 예, `test_non_scalar_backward_with_explicit_gradient` 통과.
- [x] **explicit gradient의 shape mismatch를 잡는가?**  
  👉 예, `RuntimeError: Mismatch in shape` 예외 발생 검증 완료.
- [x] **non-scalar gradient 정책을 문서화했는가?**  
  👉 예, README 및 코드 Docstring에 정책 명시 완료.

---

### 📋 3.5-C. state_dict deep copy (체크포인트 불변성)
- [x] **`state_dict()`가 deep copy를 반환하는가?**  
  👉 예, `copy.deepcopy(p.tolist())` 적용.
- [x] **모델 parameter를 바꿔도 이전 state가 변하지 않는가?**  
  👉 예, `test_state_dict_deep_copy_isolation` 회귀 테스트 통과.
- [x] **`load_state_dict()`가 값을 완벽 복원하는가?**  
  👉 예, `test_state_dict_save_and_load` 통과.
- [x] **key mismatch를 잡는가?**  
  👉 예, `KeyError` 발생 검증 완료.
- [x] **shape mismatch를 잡는가?**  
  👉 예, `RuntimeError: Shape mismatch for parameter ...` 발생 검증 완료.

---

### 📋 3.5-D. BCE 수치 안정성 (안전장치)
- [x] **`bce_loss` 내부에서 pred를 clamp하는가?**  
  👉 예, `eps=1e-7` 기반 `input.clamp(eps, 1.0 - eps)` 적용.
- [x] **pred=0에서도 NaN/inf가 안 나는가?**  
  👉 예, `test_bce_loss_stability`에서 `pred=0.0` 테스트 통과.
- [x] **pred=1에서도 NaN/inf가 안 나는가?**  
  👉 예, `test_bce_loss_stability`에서 `pred=1.0` 테스트 통과.
- [x] **`bce_loss`는 아직 experimental로 표시했는가?**  
  👉 예, `[Status: Experimental ⚠️]` 뱃징 적용.
- [x] **Sprint 4 XOR은 `mse_loss` [Status: Stable ✅] 기준으로 가는가?**  
  👉 예, Sprint 4 XOR 학습 데모는 안정성이 입증된 MSELoss 기준 진행.

---

### 📋 3.5-E. clamp / log / clip 연산
- [x] **`Tensor.log()` forward/backward가 되는가?**  
  👉 예, $d(\ln x)/dx = 1/x$ 체인룰 통과 (`test_clamp_and_log_ops`).
- [x] **`Tensor.clamp()` forward/backward가 되는가?**  
  👉 예, 마스킹 그래디언트 통과.
- [x] **`Tensor.clip()`은 clamp alias인가?**  
  👉 예, `def clip = clamp`.
- [x] **clamp 경계 밖 gradient는 0인가?**  
  👉 예, `[0.0, 1.0, 0.0]` 마스킹 검증 완료.
- [x] **BCE가 clamp 후 log를 호출하는가?**  
  👉 예, `p_clamped.log()` 호출 구조.

---

### 📋 3.5-F. Sprint 4 진입 게이트
- [x] **전체 테스트가 통과하는가?** 👉 64/64 Tests Passed (100%).
- [x] **`examples/01_tensor_basics.py` 실행되는가?** 👉 실행 완료.
- [x] **`examples/02_nn_forward_backward.py` 실행되는가?** 👉 실행 완료.
- [x] **Python backend 강제 테스트가 되는가?** 👉 `@pytest.mark.parametrize('python')` 전수 통과.
- [x] **NumPy backend 테스트가 되는가?** 👉 `@pytest.mark.parametrize('numpy')` 전수 통과.
- [x] **README에 gradient policy가 반영됐는가?** 👉 반영 완료.
