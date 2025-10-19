## 이중 정규화 문제 수정 
1. **`UCIHARInertial` 클래스 안에 정규화 처리부가 존재. ** 이 클래스는 데이터를 받으면 (파일에서 읽든, `preloaded_data`로 받든) **무조건 정규화를 수행**
2. **맨 처음** `train_set`과 `test_set_orig`를 만들 때는 원본 데이터를 넣었으므로 **정상적으로 한 번만** 정규화
3. **문제 발생:** `create_transitional_test_set` 함수는 **이미 정규화된** `test_set_orig.X` 데이터를 가져와서 수정한 뒤, **그 결과(여전히 정규화된 상태)**를 `UCIHARInertial` 클래스에 `preloaded_data`로 **다시 삽입.**
4. **이중 정규화:** `UCIHARInertial` 클래스는 `preloaded_data`를 받고 **또다시 정규화**를 시도하면서 값에 이상 발생 
5. **해결:** `create_transitional_test_set` 함수 **내부에서**, `UCIHARInertial` 클래스에 데이터를 넘기기 직전에 **원본 스케일로 복원하는 코드(`X_restored = ...`)를 추가**
6. **결과:** 복원된 (정규화 안 된) 데이터가 `UCIHARInertial` 클래스로 전달되어 **올바르게 한 번만 정규화**

## 수정 전
# ======================== Transitional Test Set ========================
def create_transitional_test_set(
    orig_dataset: UCIHARInertial, class_A: str, class_B: str, p: float, mix: float
) -> Tuple[UCIHARInertial, dict]:
    """Create transitional test set (✅ 정규화 보장)"""
    # orig_dataset.X는 이미 정규화된 데이터
    X, y = orig_dataset.X.copy(), orig_dataset.y.copy()
    N, C, T = X.shape

    # ... (스티칭 로직) ...
    # for t, s in zip(targets_A, sources_B):
    #     X[t, :, -mix_pts:] = orig_dataset.X[s, :, :mix_pts]
    # for t, s in zip(targets_B, sources_A):
    #     X[t, :, -mix_pts:] = orig_dataset.X[s, :, :mix_pts]
    # ... (스티칭 로직 끝) ...

    # 🐛 버그 발생 지점: 정규화된 X를 그대로 다시 UCIHARInertial에 넣음
    mod_dataset = UCIHARInertial(
        root="", split="test", mean=orig_dataset.mean, std=orig_dataset.std,
        preloaded_data=(X, y) # 👈 이 부분이 문제!
    )

    info = { ... }
    return mod_dataset, info


## 수정 후
# ======================== Transitional Test Set ========================
def create_transitional_test_set(
    orig_dataset: UCIHARInertial, class_A: str, class_B: str, p: float, mix: float
) -> Tuple[UCIHARInertial, dict]:
    """Create transitional test set (✅ 정규화 보장)"""
    # orig_dataset.X는 이미 정규화된 데이터
    X, y = orig_dataset.X.copy(), orig_dataset.y.copy()
    N, C, T = X.shape

    # ... (스티칭 로직) ...
    # for t, s in zip(targets_A, sources_B):
    #     X[t, :, -mix_pts:] = orig_dataset.X[s, :, :mix_pts]
    # for t, s in zip(targets_B, sources_A):
    #     X[t, :, -mix_pts:] = orig_dataset.X[s, :, :mix_pts]
    # ... (스티칭 로직 끝) ...

    # ✨ 수정 1: 이중 정규화 방지를 위해 원본 스케일로 복원
    X_restored = (X * orig_dataset.std) + orig_dataset.mean

    # ✨ 수정 2: 복원된 데이터를 UCIHARInertial에 전달
    mod_dataset = UCIHARInertial(
        root="", split="test", mean=orig_dataset.mean, std=orig_dataset.std,
        preloaded_data=(X_restored, y) # 👈 수정된 부분!
    )

    info = { ... }
    return mod_dataset, info
