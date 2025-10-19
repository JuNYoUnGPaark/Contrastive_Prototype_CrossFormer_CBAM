1. `SSL_LinearEval` 의 치명적 오류 발생
    - `SSL_LinearEval_Linear` : 학습되지 않음.
    - `SSL_LinearEval_Hyperbolic` : 학습되지 않음.
    
2. 다른 4가지 모델 비교
    - **`SSL_FineTune_Linear`**: Orig Acc **0.9790 (!!)**
    - **`SSL_FineTune_Hyperbolic`**: Orig Acc **0.9684 (!!)**
    - **`Supervised_Linear`**: Orig Acc 0.9450
    - **`Supervised_Hyperbolic`**: Orig Acc 0.9372
    <img width="1189" height="689" alt="image" src="https://github.com/user-attachments/assets/9bdfc56e-10b1-40db-9a7a-0ebcee1643b7" />
    <img width="1189" height="689" alt="image" src="https://github.com/user-attachments/assets/8967dfa3-020b-4956-9f3b-34d263e3fc89" />

    - 전이 set에서의 retention
    - **`Supervised_Hyperbolic`**: 91.60%
    - **`SSL_FineTune_Hyperbolic`**: 90.94%
    - **`Supervised_Linear`**: 90.09%
    - **`SSL_FineTune_Linear`**: 89.46%
    <img width="1189" height="689" alt="image" src="https://github.com/user-attachments/assets/0e00e956-4195-41f8-97e0-b6bc20dbd275" />

3. Linear vs Hyperbolic
    - Supervised
        - **Linear (0.9450)** > Hyperbolic (0.9372) (Linear 0.8%p 근소 우위)
    - SSL-FineTune
        - **Linear (0.9790)** > Hyperbolic (0.9684) (Linear 1.06%p 근소 우위)
    <img width="990" height="689" alt="image" src="https://github.com/user-attachments/assets/15de4c9b-ebad-4826-b76e-f5a5500122c9" />

4. EMA On/Off 비교
    
    
    | **Config** | **Metric** | **EMA ON** | **EMA OFF** | **변화량 (EMA OFF 효과)** |
    | --- | --- | --- | --- | --- |
    | **Supervised_Linear** | **Acc** | 0.9287 | **0.9450** | **+0.0163** |
    | **Supervised_Hyperbolic** | **Acc** | 0.9141 | **0.9372** | **+0.0231** |
    | **SSL_FineTune_Linear** | **Acc** | 0.3665 | **0.9790** | **+0.6125 (!!!)** |
    | **SSL_FineTune_Hyperbolic** | **Acc** | 0.6610 | **0.9684** | **+0.3074 (!!!)** |

    - EMA On 상태에서는 SSL feature에 Hyperbolic Classifier 분류기가 압도적으로 우수했지만 EMA Off 시 Linear Classifier가 Supervised, SSL_FineTune 환경 모두에서 Hyperbolic Classifier보다 근소하게 더 나은 성능을 보임.

    - EMA가 성능 저해의 원인으로 보임. SSL 학습 붕괴의 핵심 요인으로 판단됨.

<img width="1121" height="789" alt="image" src="https://github.com/user-attachments/assets/ab548989-8714-497e-8299-35d0a5ac0512" />
<img width="1113" height="789" alt="image" src="https://github.com/user-attachments/assets/8e36c15a-9116-49d2-83b7-8bea30a4d74b" />
