# Ablation Study 결과 분석

- 제안한 모든 모듈(CBAM, CrossFormer, Contrastive Loss)이 결합되었을 때 시너지를 발휘하여 가장 좋은 성능을 달성

## 1. CrossFormer: 가장 극적인 성능 향상 요인

- **Baseline (Conv+Attn): 92.60% Acc, 42,246 파라미터**
- **+ CrossFormer: 94.06% Acc, 67,718 파라미터**

## 2. CBAM: 단독으로는 무의미, CrossFormer와는 시너지 폭발

- **Baseline (92.60%) → + CBAM (92.09%)**
- **+ CrossFormer (94.06%) → + CBAM + CrossFormer (95.28%)**

## 3. Contrastive Loss: 성능 향상

- 파라미터 수를 전혀 늘리지 않고 모델의 훈련 방식을 개선하여 성능 향상
- **+ CrossFormer (94.06%) → + CrossFormer + Contrast (94.81%)**
- **+ CBAM + CrossFormer (95.28%) → Full Model (95.32%)**

## 4. 최종 결론

- **Baseline:** 92.60%
- **+ CrossFormer (핵심):** 94.06% (+1.46%)
- **+ CrossFormer + CBAM (시너지):** 95.28% (+1.22%)

- **+ CrossFormer + CBAM + Contrast (최종 튜닝):** **95.32%** (+0.04%)

<img width="1994" height="590" alt="image" src="https://github.com/user-attachments/assets/1100984c-035a-4c8b-b637-64fd5f7955f4" />
