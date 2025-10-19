## 실험 결과 분석

- 두 버전의 차이는 다음과 같음
    
    
    | **Augmentation** | **SSL_basic_low (약)** | **SSL_basic_high (강)** |
    | --- | --- | --- |
    | Time Warp (prob) | 0.1 | **0.3** |
    | Cutout (prob) | 0.2 | **0.3** |
    | Cutout (ratio) | 0.1 | **0.2** |
1. 두 버전 모두 성공적으로 모델이 학습됨. 
2. 증강 강도가 달라져도 모두 Supervised Learning을 압도하는 성능 달성
    - 다만, 두 설정 간 성능-견고성 Trade-off
        - **`SSL_basic_high` (강한 증강):** 가장 높은 **절대 성능 (98.03%)**을 달성
        - **`SSL_basic_low` (약한 증강):** 가장 높은 **견고성 (Retention 88.90%)**

3. Supervise 모델은 증강과 무관하므로 SSL 모델만 비교 (모두 95% 정도의 성능 달성)
    
    
    | **Config** | **Weak Aug (Low)** | **Strong Aug (High)** | **승자** |
    | --- | --- | --- | --- |
    | **SSL_LinearEval_Linear** | 0.9179 | **0.9223** | **Strong** |
    | **SSL_LinearEval_Hyperbolic** | **0.9352** | 0.9267 | **Weak** |
    | **SSL_FineTune_Linear** | **0.9796** | 0.9732 | **Weak** |
    | **SSL_FineTune_Hyperbolic** | 0.9776 | **0.9803** | **Strong** |
    - `LinearEval`(backbone 고정): 약한 증강이 최고 성능
    - `Finettune`(전체 미세조정): 강한 증강이 최고 성능
    
    → 대체적으로 더 강한 증강이 SSL-Finetune 모델의 성능 잠재력을 최대치로 이끌어내는데 유리했다. 
    
    | **Config** | **Weak Aug (Low)** | **Strong Aug (High)** | **승자** |
    | --- | --- | --- | --- |
    | **SSL_LinearEval_Linear** | **88.49%** | 88.25% | **Weak** |
    | **SSL_LinearEval_Hyperbolic** | 88.60% | **89.00%** | **Strong** |
    | **SSL_FineTune_Linear** | **88.90%** | 87.76% | **Weak** |
    | **SSL_FineTune_Hyperbolic** | **87.15%** | 86.44% | **Weak** |
    - 약한 증강이 SSL-Finetune 모델에서 일관되게 더 높은 retention 보임.
    - 특히, SSL_FineTune_Hyperbolic(Strong Aug) 모델의 견고성이 가장 낮았다.
    

    → 대체로 증강 정도가 높을수록 안정성은 떨어지는 Trade-off 관계성 보임.

4. 결과 시각화
<img width="1790" height="889" alt="image" src="https://github.com/user-attachments/assets/61237798-0749-4d29-ade5-6aa68330eea4" />
<img width="1790" height="889" alt="image" src="https://github.com/user-attachments/assets/b34e734f-a0b8-4755-b7fb-84d6e1d1fe70" />
<img width="1790" height="889" alt="image" src="https://github.com/user-attachments/assets/f53d0647-501a-45bf-b6b1-83f78279d7e1" />
