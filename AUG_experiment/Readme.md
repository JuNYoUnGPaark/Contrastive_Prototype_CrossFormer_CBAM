## 데이터 증강 강도에 따른 SSL 성능 영향 분석
- UCI-HAR 데이터셋을 이용한 Self-Supervised Learning Robustness 실험

### 개요
- SSL의 핵심 Hyperparameter인 Data Augmentation의 강도가 모델의 최고 성능과 안정성에 미치는 영향 분석
- '약한 증강'이 최고 성능을 거의 해치지 않으면서도 안정성을 크게 개선

### 실험 환경
- 공통
	- 데이터셋: UCI-HAR
	- 모델 아키텍쳐: ResNet1D + Transformer Encoder
	- 평가지표
		- 최고성능: `Orig Acc` (원본 최고 정확도)
		- 안정성: `Retention` (전이 데이터에서의 성능 유지율)

- 비교 변수: 증강 강도
	- 강한 증강
		- jitter_scale = 0.05
    		- scale_range = (0.8, 1.2)
    		- channel_drop_prob = 0.2
		- time_warp_prob = 0.3  
    		- cutout_prob = 0.3  
    		- cutout_ratio = 0.2  
	- 약한 증강
		- jitter_scale = 0.05
    		- scale_range = (0.8, 1.2)
    		- channel_drop_prob = 0.2
		- time_warp_prob = 0.1  
    		- cutout_prob = 0.2  
    		- cutout_ratio = 0.1  

- 실험 결과
	- `both_visualize.ipynb` 참고 
	- 시각화 분석
		- [orig acc 비교]
			두 증강 전략 모두에서 SSL Fine-Tune 방식이 약 98%의 정확도로 Supervised 방식(약 96%)을 능가했다. 이는 증강 강도와 무관하게 SSL 사전학습이 최고 성능을 높이는 데 효과적임을 보여준다

		- [retention 비교]
			안정성에서는 두 전략의 명암이 극명하게 갈렸다. '약한 증강'을 사용한 SSL 모델의 안정성(약 53%)이 '강한 증강'(약 47%)에 비해 약 10%p 가량 유의미하게 높았다. 하지만 두 경우 모두 Supervised 모델의 안정성(약 62%)에는 미치지 못했다.

		- [성능-안정성 관계도]
			이 그래프는 성능과 안정성 간의 트레이드오프 관계를 명확히 보여준다. 가장 이상적인 우측 상단에는 'Supervised(안정성)'과 'SSL Fine-Tune(성능)'이 각각 자리 잡고 있다. 

- 결론
	- 약한 증강이 trade-off의 balance가 좋았고 강한 증강은 모델의 과잉 일반화를 유발한다. 또한 증강 강도를 조절해도 지도학습 모델의 안정성을 넘어서지 못했다는

	  한계가 있다. 

