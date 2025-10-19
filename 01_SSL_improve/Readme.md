## 실험 결과 분석

1. `SSL_LinearEval` 의 치명적 오류 발생
    - `SSL_LinearEval_Linear` : 학습되지 않음.
    - `SSL_LinearEval_Hyperbolic` : 학습되지 않음.
    
2. 다른 4가지 모델 비교
    - **`Supervised_Linear`**: Orig **Acc 0.9294** (압도적인 1위)
    - **`Supervised_Hyperbolic`**: Orig **Acc** 0.8731
    - **`SSL_FineTune_Hyperbolic`**: Orig **Acc** 0.6634
    - **`SSL_FineTune_Linear`**: Orig **Acc** 0.3946
    <img width="1189" height="689" alt="image" src="https://github.com/user-attachments/assets/e6b12c9e-87e6-49ed-bd7d-27bdba05e024" />
    - **`Supervised_Hyperbolic`**: 92.52% (Mod: 96.11% | Str: 90.13%)
    - **`SSL_FineTune_Hyperbolic`**: 92.28% (Mod: 97.77% | Str: 88.61%)
    - **`Supervised_Linear`**: 91.62% (Mod: 94.91% | Str: 89.42%)
    - **`SSL_FineTune_Linear`**: 90.34% (Mod: 92.78% | Str: 88.71%)
    <img width="1189" height="689" alt="image" src="https://github.com/user-attachments/assets/953489a6-2bd9-4e38-afff-030006fca1a1" />

3. Linear vs Hyperbolic
    - Supervised
        - **Linear (0.9294)** > Hyperbolic (0.8731)
        - inear 분류기가 5.6%p 더 우수
    - SSL-FineTune
        - **Hyperbolic (0.6634)** > Linear (0.3946)
        - Hyperbolic 분류기가 26.8%p라는 매우 큰 차이로 우수
    <img width="830" height="790" alt="image" src="https://github.com/user-attachments/assets/8d1f9bb6-efd2-4e64-a234-2d8e9b08139e" />
    <img width="990" height="689" alt="image" src="https://github.com/user-attachments/assets/ad384d20-7420-4177-b05b-d20fae9d9029" />

4. 결론
    - **Hyperbolic 분류기는 SSL과 시너지 발생.**`SSL_FineTune` 환경에서 Hyperbolic 분류기가 Linear 분류기보다 월등히 나은 성능을 보임.


<img width="1121" height="789" alt="image" src="https://github.com/user-attachments/assets/0cb74c57-b32f-47dd-9e23-90cf6a08c414" />
<img width="1113" height="789" alt="image" src="https://github.com/user-attachments/assets/7a23b4ea-1b61-4c8d-826c-5972cf243ac4" />
