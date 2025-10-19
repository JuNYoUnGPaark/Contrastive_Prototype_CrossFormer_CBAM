## 실험 결과 분석

1. `SSL_LinearEval` 의 치명적 오류 해결
    - `EMA ON / Scheduler ON` → **실패 (16~17%)**
    - `EMA OFF / Scheduler ON` → **실패 (10~19%)**
    - `EMA OFF / Scheduler OFF` → **성공! (91~93%)**
    
    → Cosine + Warmup 스케줄러를 비활성화한 결과 정상 작동함.
    
2. 모델 비교
    - **`SSL_FineTune_Hyperbolic`**: Orig Acc**0.9718 (!!)** (1위)
    - **`SSL_FineTune_Linear`**: Orig Acc**0.9708** (2위)
    - **`Supervised_Linear`**: Orig Acc0.9545
    - **`Supervised_Hyperbolic`**: Orig Acc0.9505
    - **`SSL_LinearEval_Hyperbolic`**: Orig Acc0.9277
    - **`SSL_LinearEval_Linear`**: Orig Acc 0.9138
    <img width="1189" height="689" alt="image" src="https://github.com/user-attachments/assets/7c2e636e-25f2-4e44-9cae-7c108b52b204" />
    <img width="1189" height="689" alt="image" src="https://github.com/user-attachments/assets/4f913c6b-a1f0-42af-bb8f-44fb4398f806" />

    - 전이 set에서의 retention
    - **`SSL_LinearEval_Linear`**: 90.98% (가장 견고함)
    - `Supervised_Hyperbolic`: 90.72%
    - `Supervised_Linear`: 90.71%
    - `SSL_LinearEval_Hyperbolic`: 90.15%
    - `SSL_FineTune_Linear`: 89.61%
    - **`SSL_FineTune_Hyperbolic`**: 88.99% (가장 취약함)
    <img width="1189" height="689" alt="image" src="https://github.com/user-attachments/assets/42f61410-b4d9-4ace-ac47-1ad616e1f7de" />

    → 성능과 견고성 간의 Trade-Off
    <img width="829" height="790" alt="image" src="https://github.com/user-attachments/assets/a9f8bd22-82bf-4b5c-9061-7d09631add51" />
    <img width="989" height="689" alt="image" src="https://github.com/user-attachments/assets/65df01b7-4d78-4fa3-a7fd-636a00163a48" />

3. Linear vs Hyperbolic
    - `Supervised` 환경:
        - **Linear (0.9545)** > Hyperbolic (0.9505) (Linear +0.4%p 근소 우위)
    - `SSL_LinearEval` 환경:
        - **Hyperbolic (0.9277)** > Linear (0.9138) (Hyperbolic +1.4%p 우위)
    - `SSL_FineTune` 환경:
        - **Hyperbolic (0.9718)** > Linear (0.9708) (Hyperbolic +0.1%p 근소 우위)
        
4. 스케줄러(Cosine + Warmup) ON/OFF 비교
    
    
    | **Config** | **Metric** | **Sched ON (Log 3)** | **Sched OFF (Log 4)** | **변화량 (Sched OFF 효과)** |
    | --- | --- | --- | --- | --- |
    | **SSL_LinearEval_Linear** | **Acc** | 0.1042 (실패) | **0.9138** | **+0.8096 (FIXED!)** |
    | **SSL_LinearEval_Hyperbolic** | **Acc** | 0.1958 (실패) | **0.9277** | **+0.7319 (FIXED!)** |
    | `Supervised_Linear` | `Acc` | 0.9450 | **0.9545** | **+0.0095** |
    | `Supervised_Hyperbolic` | `Acc` | 0.9372 | **0.9505** | **+0.0133** |
    | `SSL_FineTune_Linear` | `Acc` | **0.9790** | 0.9708 | -0.0082 |
    | `SSL_FineTune_Hyperbolic` | `Acc` | 0.9684 | **0.9718** | +0.0034 |
    

    → Supervised 모델은 스케줄러를 off했을 때 성능이 향상된 반면, SSL_FineTune 모델은 영향이 미미. 스케줄러를 on 시 97.9%에서 off시 97.2%로 SSL_FineTune 모델에 아주 약간의 이점 제공 가능성도 있음.


<img width="1121" height="789" alt="image" src="https://github.com/user-attachments/assets/61dd7043-32da-45f5-b923-9342667ea13b" />
<img width="1103" height="789" alt="image" src="https://github.com/user-attachments/assets/f7583b6d-28ce-4e4d-b5e2-058c8ff55bcc" />
<img width="790" height="790" alt="image" src="https://github.com/user-attachments/assets/800cf25e-ddbc-4335-aa51-3b9e8e1bfb4c" />
