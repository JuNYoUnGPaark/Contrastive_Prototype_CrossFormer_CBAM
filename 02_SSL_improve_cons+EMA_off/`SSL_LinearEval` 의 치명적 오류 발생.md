1. `SSL_LinearEval` 의 치명적 오류 발생
    - `SSL_LinearEval_Linear` : 학습되지 않음.
    - `SSL_LinearEval_Hyperbolic` : 학습되지 않음.
    
2. 다른 4가지 모델 비교
    - **`SSL_FineTune_Linear`**: Orig Acc **0.9790 (!!)**
    - **`SSL_FineTune_Hyperbolic`**: Orig Acc **0.9684 (!!)**
    - **`Supervised_Linear`**: Orig Acc 0.9450
    - **`Supervised_Hyperbolic`**: Orig Acc 0.9372
    
    - 전이 set에서의 retention
    - **`Supervised_Hyperbolic`**: 91.60%
    - **`SSL_FineTune_Hyperbolic`**: 90.94%
    - **`Supervised_Linear`**: 90.09%
    - **`SSL_FineTune_Linear`**: 89.46%
    
3. Linear vs Hyperbolic
    - Supervised
        - **Linear (0.9450)** > Hyperbolic (0.9372) (Linear 0.8%p 근소 우위)
    - SSL-FineTune
        - **Linear (0.9790)** > Hyperbolic (0.9684) (Linear 1.06%p 근소 우위)
        
4. EMA On/Off 비교
    
    
    | **Config** | **Metric** | **EMA ON** | **EMA OFF** | **변화량 (EMA OFF 효과)** |
    | --- | --- | --- | --- | --- |
    | **Supervised_Linear** | **Acc** | 0.9287 | **0.9450** | **+0.0163** |
    | **Supervised_Hyperbolic** | **Acc** | 0.9141 | **0.9372** | **+0.0231** |
    | **SSL_FineTune_Linear** | **Acc** | 0.3665 | **0.9790** | **+0.6125 (!!!)** |
    | **SSL_FineTune_Hyperbolic** | **Acc** | 0.6610 | **0.9684** | **+0.3074 (!!!)** |

    - EMA On 상태에서는 SSL feature에 Hyperbolic Classifier 분류기가 압도적으로 우수했지만 EMA Off 시 Linear Classifier가 Supervised, SSL_FineTune 환경 모두에서 Hyperbolic Classifier보다 근소하게 더 나은 성능을 보임.
    - EMA가 성능 저해의 원인으로 보임. SSL 학습 붕괴의 핵심 요인으로 판단됨.