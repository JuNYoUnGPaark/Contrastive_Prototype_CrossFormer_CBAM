# comment에 기반한 실험

- 모든 실험의 Epochs = 100으로 고정함.
- 아래의 모든 표는 Best F1 Score로 작성됨.
- 작업 방식: 순차적으로 최적값을 찾아 적용하고, 파일을 생성하면서 다음 Step 진행            최종 파라미터 확정 후 코드를 다시 돌리지 않은 상태의 Log임!

---

## 1. Prototype 초기화 방식 변경

*Xavier init -> 평균 feature init*, temperatrue = [0.05, 0.07, 0.1, 0.15]

- 기존 Code 확인

```python
class ContrastCrossFormerBlock(nn.Module):
	def __init__(self, dim, n_prototypes=6, n_heads=4, mlp_ratio=2.0, dropout=0.1):
		...
		
      # Learnable prototypes (L2 정규화 적용)
      self.prototypes = nn.Parameter(torch.randn(n_prototypes, dim))
      nn.init.xavier_uniform_(self.prototypes)
```

- Xavier initialization은 주로 가중치에 사용된다. Prototype에는 각 Class의 데이터 분포 중심을 나타내는 평균 특징 벡터로 initialization하는 것이 더 어울린다.
- Xavier initialization = Conv에 더 어울리는 초기화 방식
- 평균 특징 벡터란?
    - Prototype을 각 Class의 평균적인 모습으로 시작하자는 idea
    1. Model이 각 Sequence를 입력받아 처리 후, 하나의 특징 벡터로 요약한다. 
        
        (Embeding → CBAM/CrossFormer → Pooling: 최종 Classifier 직전의 벡터로)
        
    
    ```python
    [걷기 샘플 1] -> [걷기 특징 벡터 1] (64차원)
    
    [앉기 샘플 1] -> [앉기 특징 벡터 1] (64차원)
    
    [눕기 샘플 1] -> [눕기 특징 벡터 1] (64차원)
    
    [걷기 샘플 2] -> [걷기 특징 벡터 2] (64차원)
    
    ... (모든 훈련 샘플에 대해 반복)
    ```
    
    1. 1단계에서 만든 모든 특징 벡터들을 Label에 따라 묶는다.
    
    ```python
    걷기 그룹: [걷기 특징 벡터 1], [걷기 특징 벡터 2], ... (수백 개)
    
    앉기 그룹: [앉기 특징 벡터 1], [앉기 특징 벡터 2], ... (수백 개)
    
    ... (총 6개 활동 그룹)
    ```
    
    1. 각 그룹 내 모든 특징 벡터들의 Mean을 계산한다.
    2. 3단계에서 계산한 6개의 평균값을 이용해서 Prototype 초기화 수행한다. 

- 적용법
    1. 모델의 특징 추출기 부분을 임시로 정의 (Clean한 계산을 위해서) 
    2. Train data 전체를 임시 모델에 통과시켜 특징 벡터 추출
    3. 특징 벡터 그룹화
    4. 각 Class 그룹의 평균 계산 
    5. `ContrastCrossFormerBlock` 의 `self.prototypes` 파라미터를 Xavier → 평균값으로 초기화 

---

## 2. Contrastive Loss Weight 조절하기

*[0.05, 0.1, 0.15, 0.2, 0.25, 0.3]*

- 요약
    - 너무 크면 = Main loss인 C.E Loss 학습 방해 가능성
    - 너무 작으면 = Contrastive Loss Effect 감소 가능성

---

## 3. Epochs와 Batch_size 조절하기

- [Epcohs=100으로 우선 고정]

- (SimCLR 논문 참고) Contrastive Loss는 Batch 내 다른 샘플들은 ‘negative sampel’로 사용하여 학습하는데 Batch_size가 클수록 더 많은 negative sample을 한 번에 비교할 수 있게 되어, model이 더 정교하고 어려운 구분 경계를 학습하는데 도움이 될 수 있다.
    - 왜 Batch_size가 클수록 “더 많이 보나”?
        
        모든 데이터를 결국 다 보긴 하지만(Batch_size가 어떻든) 한 번의 model update에서 얼마나 많은 negative sample과 비교하며 배우느냐가 Contrastive Learning에 큰 영향을 미친다. (SimCLR의 핵심 강조 사항) **
        

---

## 4. InfoNCE → triplet, barwen twins

**Comment**: *InfoNCE 말고 triplet이나 barwen twins도 적용해보세요 제 경험상 twins가 가장 좋았어요*

- **Triplet Loss**: Anchor 기준 Positive, Negative sample 3가지를 사용. Anchor와 Positive Sample 간 거리가 Negative Sample과의 거리보다 일정 margin 이상 작아지도록 학습
- **Barlow Twins**: 자기 지도 학습 방식. 같은 데이터에 두 가지 다른 aug. 적용하여 모델에 넣고 두 결과 간의 correlation matrix이 identity matrix에 가까워지도록 학습.

---

## 5. `use_dim_reduction` 추가 실험 

- `use_dim_reduction=True`로 설정하면 Embeding된 64차원 특징 → 32 → 64차원
1. 정보 압축: 입력 데이터를 더 작은 차원의 벡터로 강제 압축하면 불필요한 noise를 버리고 가장 중요하고 핵심적인 정보만 남길 수 있다.
2. 효율성 및 일반화: bottleneck 구간은 더 적은 params로 FLOPs를 줄이고 Overfitting을 방지하여 일반화 성능 개선 가능성을 높인다. 

---

## 6. CrossFormer의 Head 수 늘리기

- Head 수를 늘리는 것의 장점 (4 → 8)
    - 모델이 입력 Sequence 내의 더 다양하고 복잡한 관계를 동시에 포착 가능
    - 안정적인 학습 가능
        - `working_dim` 이 32일때, 4 ⇒ 32 / 4 = 8, 8 ⇒ 32 / 8 = 4. Head가 8이면 4차원 공간에서 더 전문적인 정보를 학습. 특정 Head가 학습 실패하더라도 다른 Head들이 보완해줄 수 있다.

---

## 7. `Proto_proj` 구조 변경 + t-SNE 시각화

- `Linear -> BatchNorm -> GELU -> Linear`를 왜 사용할까?
    - BN은 신경망의 각 Layer에 들어오는 입력의 분포를 평균0, 분산1로 정규화하는 역할
    1. 안정적인 학습: `proto_proj`에 들어오는 특징 벡터(`proto_features`)는 매 배치마다, 그리고 학습이 진행됨에 따라 분포가 계속 변한다. `BatchNorm`은 이 변화를 안정시켜주어 Triplet/Contrastive Loss가 더 일관된 입력을 받게 한다. 이는 학습 과정을 안정시키고 수렴 속도를 높여준다.
    2. Loss 계산 최적화: `BatchNorm`을 통해 특징들이 정규화된 공간에 놓이게 되면, 이 거리/유사도 계산이 더 의미 있고 안정적으로 이루어질 수 있다.
    3. 활성화 함수(GELU) 최적화: GELU와 같은 활성화 함수는 입력값이 특정 범위(주로 0 근처)에 있을 때 가장 효과적으로 작동한다. 

---

## 8. 모델 크기 증가시키기
