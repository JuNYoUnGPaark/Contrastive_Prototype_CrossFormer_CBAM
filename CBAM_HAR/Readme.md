# `Prototype_CrossFormer_with_CBAM` Model 분석

( `D=64` (즉, `embed_dim=64`)라고 가정)

## 1. `1D-CBAM` Module (Channel + Temporal Attention)

---

- `self.embedding` (Conv1d)을 통과한 `(B, D, T)` = **`(B, 64, 128)`** 텐서를 입력으로 받는다.
- `ChannelAttention1D` : 9개 Sensor Channel 중 “어떤 센서가 중요한지”를 학습
- `TemportalAttention1D` : 128개 TimeStep중 “어떤 순간이 중요한지”를 학습      (→ Spatial Attention)

---

**1-1. `ChannelAttention1D` (어떤 센서가 중요한지?)**

- X: `(B, 64, 128)`을 받아서 64개 Channel의 중요도를 계산 → `(B, 64, 1)`

1. **Pooling(Avg/Max)**: B개 각각에 대해서 128 TimeStep을 압축 → `(B, 64, 1)`
2. **Squeeze**: `(B, 64)`  
3. **MLP**: Bottleneck 구조를 통과 → (64 -> 4 -> 64) → `(B, 64)`
4. **Add + Sigmoid**: B개 각각의 점수 계산 → `(B, 64)`
5. **Unsqueeze**: 총 B*9개의 중요도 점수가 계산됨 → `(B, 64, 1)`
6. **Broadcasting**: X: `(B, 64, 128)` * Unsqueeze 결과: `(B, 64, 1)` → `(B, 64, 128)`

- Squeeze 시 정보 손실 혹은 잘못된 융합은 없는지?
    
    (예) [ [5.1], [2.3], [9.0] ] → `.squeeze(-1)` → [ 5.1, 2.3, 9.0 ]
    
    그대로 MLP에 통과하려고 하면 `nn.Linear()` 는 [ ]와 같은 빈 List를 통과하려고 하기 때문에 Squeeze를 통해서 내용물 자체를 통과시켜줘야한다. 
    
- 다시 Unsqueeze를 하는 이유는?
    
    계산된 가중치를 입력 데이터와 곱해주기 위해서 Shape을 다시 맞추는 것이다. (요소별 곱)
    

**1-2. `TemportalAttention1D` (어떤 순간이 중요한지?)**

- 채널 어텐션을 거친 데이터 X:: `(B, 9, 128)` 을 받아서 128개 TimeStep의 중요도를 계산 → `(B, 1, T)`
    
    
    1. **AvgPooling**: C에 대해서 Avg하여 128개 TimeStep의 평균 신호 강도 계산 → `(B, 1, 128)`
    2. **MaxPooling**: C에 대해서 Max하여 128개 TimeStep의 최대 신호 계산 → `(B, 1, 128)`
    3. **Concat**: Avg와 Max의 결과를 C축으로 합친다. → `(B, 2, 128)`
    4. **Conv1D + Sigmoid**: 128개 각각의 점수 계산 → `(B, 1, 128)` 
    5. **Broadcasting**: `(B, 64, 128) * (B, 1, 128)` = `(B, 64, 128)`

- 왜 `nn.AdaptiveAvgPool1d` Module을 사용하지 않고 `torch.mean` 을 사용했는지?
    
    항상 마지막 차원에 대해서만 Pooling한다. `torch.mean(dim=-1, keepdim=True)` 와 완전히 동일하기에 해당 방식을 이용.
    

**1-3. `CBAM1D`** 

- `ChannelAttention1D` → `TemportalAttention1D` 순차 적용 (입/출력 모두 **`(B, 64, 128)`**)

## 2. `Contrastive Prototype Loss` Module

---

- 일명 ‘대조 학습’
- 기존의 CrossEntropyLoss에 추가적인 Mission을 부여
- Mission: 같은 Class는 해당하는 Class의 대표(Prototype)과 가까워져야하고, 다른 Class의 대표와는 멀어져야한다.
- **`temperature`** : 값이 낮을수록 날카로운 구별, 높을수록 부드러운 구별

---

1. **입력으로 3가지를 받는다.**
    - `features`: `(B, D)` - 모델이 방금 추출한 B개의 데이터 특징 (예: `(128, 64)`)
    - `prototypes`: `(N_class, D)` - 6개 클래스 각각의 "대표" 벡터 (예: `(6, 64)`)
    - `labels`: `(B,)` - B개 데이터의 실제 정답 (예: `[0, 1, 5, ...]`)
        
        *→ 여기서 `D` 는 요약벡터의 길이. 모델이 각 데이터 샘플(원본은 `(9, 128)` 모양의 센서 신호)을 64개의 숫자로 이루어진 **하나의 벡터**로 압축(요약)했다는 뜻*
        
        *(`embed_dim` 파라미터에서 설정)*
        
2. **L2 정규화**
    - 순수하게 "방향” **(유사도)만** 비교하기 위해서 ****`features`, `prototypes`에 적용
3. **유사도 계산**
    - 이 둘을 행렬곱하면 수학적으로 ‘코사인 유사도’가 된다.
        - 유사도의 모양은 `(B, N_class)` (예: `(128, 6)`)
        - (예) `logits[0, 1]`: 0번 데이터가 1번 프로토타입('뛰기' 대표)과 얼마나 유사한지
    - 마지막으로 `temperature` 로 나눠준다.
4. **InfoNCE Loss**
    - 위에서 계산된 `logits`와 `labels`를 CrossEntropyLoss 계산
        - CrossEntropyLoss는 정답은 1로 나머지는 0으로 만들기 때문에
            - **`labels`에 해당하는 정답 프로토타입의 유사도 점수(Logit)는 최대화,**
            - **`labels`가 아닌 오답 프로토타입의 유사도 점수는 최소화하게 된다.**

## 3. `ContrastCrossFormerBlock` Module

---

- "**Cross-Attention**"과 "**Self-Attention**"이 결합된 하이브리드 Transformer 블록
- CBAM 통과 후 `transpose(1, 2)`를 거친 **`(B, T, D)`** = **`(B, 128, 64)`** 텐서를 입력으로 받는다.
- 입력 X = (B, 128, D) (여기서 `D=64`)
    - `self.prototypes = nn.Parameter(...)`: `ContrastivePrototypeLoss` 에 사용될 대표 벡터를 학습 가능한 파라미터로 설정
    - `self.cross_attn = nn.MultiheadAttention(...)`: Data와 Prototype의 **Cross-Attn**
    - `self.self_attn = nn.MultiheadAttention(...)`: Data **Self-Attn**
    - `self.proto_proj = nn.Sequential(...)`: `ContrastivePrototypeLoss` 에 사용될 `features` 를 만들기 위한 별도의 소형 MLP

---

**3-1. Cross-Attention**

1. 학습된 Prototype `(6, D)` 를 L2 정규화 (안정적 연산을 위해)
2. 이걸 B 크기만큼 복사하여 `(B, 6, D)`
3. Cross-Attn 수행
    - **`Query (Q)`** = `x_norm` (입력 데이터, `(B, 128, D)`)
    - **`Key (K)`** = `prototypes` (대표 벡터, `(B, 6, D)`)
    - **`Value (V)`** = `prototypes` (대표 벡터, `(B, 6, D)`)
    
    각각의 의미는 다음과 같다.
    
    - **Q(데이터):** "나는 128개의 타임스텝을 가진 데이터인데..."
    - **K(프로토타입):** "...6개의 클래스 대표(K) 중에 누구랑 가장 비슷하지?"
    - **V(프로토타입):** "...가장 비슷한 대표(V)의 정보를 나에게 줘!"
4. 입력 데이터 X가 6개의 대표 정보와 “섞여서” 정제된 데이터가 된다. `(B, 128, D)`
5. X = X + 정제된_X: Residual Connection 

*[p.s. 어떻게 “섞이는가”? ["가중 평균"이 만들어지는 2단계 과정](https://www.notion.so/2-294ccff627fb80ee80c3fb8f28b94833?pvs=21)]*

**3-2. Self-Attention**

1. "클래스 대표 정보로 한 번 정제된 나의 128개 타임스텝들끼리, 서로 어떤 타임스텝이 중요한지 계산해 보자.”
2. X = X + 정제된_X: Residual Connection 

**3-3. FFN** 

- 특징 변환
- 정제된 `(B, 128, D)` 텐서의 각 타임스텝을 MLP에 통과시켜 더 복잡한 특징으로 변환

**3-4. `features` 만들기 (Training 시에만)**

1. `(B, 128, D)`를 T 축으로 GAP하여 하나의 요약벡터 `(B, 64)`를 만든다.
2. 요약벡터를 소형 MLP(Projection Head)에 통과시킨다. → 이게 `features` 

- 왜 굳이 CrossFormerBlock을 사용하지 않고 또다른 MLP를 사용하나?
"분류"에 유리한 **풍부하고 일반적인 특징**을 만드는 데 집중하게 놔두고 대조 학습에 최적화된 새로운 공간으로 "투영(Project)" 또는 "번역(Translate)"하는 별도의 역할을 수행하도록 분리하는 것.
    
    이는 SimCLR 같이 유명한 대조학습 논문에서 성능 향상에 매우 중요하다고 입증된 방법임. 
    

## 4. Model Structure Summary

| **Stage** | **Layer / Operation** | **Output Shape** | **Purpose** |
| --- | --- | --- | --- |
| **Input** | Raw Sensor Data | `(B, C_in, T)(B, 9, 128)` | 9축 센서 시계열 데이터 |
|  |  |  |  |
| **1. Embedding** | `self.embedding` (Conv1d) | `(B, D, T)(B, 64, 128)` | 입력 특징 추출 (9-ch -> 64-dim) |
| **2. Refining** | `self.cbam` (CBAM1D) | `(B, D, T)(B, 64, 128)` | 채널(센서) 및 시간(Temporal) 정제 |
|  |  |  |  |
| **3. Encoder** | `transpose` (Reshape) | `(B, T, D)(B, 128, 64)` | Transformer 입력 형식으로 변경 |
|  | **`self.crossformer` (Block)** |  |  |
|  | 1) `Cross-Attention` | `(B, T, D)` | 데이터(Q)와 프로토타입(K, V) 연산 |
|  | 2) `Self-Attention` | `(B, T, D)` | 시간 축(T) 내부의 관계 연산 |
|  | 3) `FFN (MLP)` | `(B, T, D)(B, 128, 64)` | 특징 비선형 변환 |
|  |  |  |  |
| **4. Output Heads** | *--- (데이터 흐름이 2개로 분기) ---* |  |  |
|  |  |  |  |
| **Path 1:
Classification** | `transpose` (Reshape) | `(B, D, T)(B, 64, 128)` | Pooling을 위해 원상 복구 |
|  | `self.pool` (AvgPool1d) | `(B, D)(B, 64)` | 시퀀스 요약 (Global Pooling) |
|  | `self.classifier` (Linear) | `(B, N)(B, 6)` | **최종 분류 Logits (CrossEntropyLoss용)** |
|  |  |  |  |
| **Path 2:
Contrastive** | `x.mean(dim=1)` (AvgPool) | `(B, D)(B, 64)` | 시퀀스 요약 (Global Pooling) |
| **(Training Only)** | `self.proto_proj` (Linear) | `(B, D)(B, 64)` | **대조 학습용 Features (ContrastiveLoss용)** |
