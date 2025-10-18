## 고급 기능들을 적용했을 때 Pretext task의 학습 붕괴 원인 분석 
- UCI-HAR 데이터셋을 이용한 Self-Supervised Learning Robustness 실험

### 개요
- Consistency Loss, EMA, Cosine-warmup scheduler 등의 고급기능을 추가하여 학습할 경우 SSL을 사용한 모델의 사전학습을 방해한다고 판단, 가장 핵심적인 원인을 찾기 위한 실험 진행

### 실험 환경
- 공통
	- 데이터셋: UCI-HAR
	- 모델 아키텍쳐: ResNet1D + Transformer Encoder
	- 평가지표
		- 최고성능: `Orig Acc` (원본 최고 정확도)
		- 안정성: `Retention` (전이 데이터에서의 성능 유지율)

- 비교 변수: 고급 기능 On/Off
	- All ON: 모든 고급 기능 활성화 
	- consOFF: 일관성 손실 비활성화
	- cons+emaOFF: 일관성 손실 + EMA 비활성화
	- cons+ema+CosOFF: 모든 고급 기능 비활성화 

- 실험 결과
	- 주범은 `EMA`,  공범은 `일관성 손실`
		- 일관성 손실만을 껐을 땐 학습이 여전히 실패했지만, 추가적으로 EMA까지 끄면 Orig Acc가 66% -> 96.8%로 정상화
		- EMA의 가중치 평균화 방식이 민감한 대조 학습 메커니즘과 근본적으로 충돌한 것으로 보임. 
		- 일관성 손실은 성능 저하의 원인이지만 학습 붕괴의 핵심 원인은 아니었음.
	
	- `코사인 스케줄러`의 안정성 저하 효과
		- EMA까지 꺼서 최고 성능은 회복했지만 안정성(retention)은 여전히 55.4%로 매우 낮았음. 여기서 코사인 스케줄러까지 끄자 안정성이 63.8%로 유의미하게 상승
		- 코사인 스케줄러는 Orig ACC에 큰 영향은 주지 않았지만 안정성을 감소시키는 원인으로 보임.

- 결론
	- EMA: 학습 붕괴의 핵심 원인
	- Consistency Loss: 성능 저하의 원인
	- Cosine Scheduler: 안정성 저하의 원인 