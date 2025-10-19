# UCI HAR Dataset: Human Activity Recognition Using Smartphones

#### 1. Overview
- 스마트폰 내장 센서 데이터를 활용하여 사람의 일상적 신체 활동(Human Activities) 을 분류하기 위한 대표적인 공개 데이터셋
- UCI(University of California, Irvine) Machine Learning Repository 에서 제공되며, 다양한 기계학습 및 딥러닝 기반 인체 활동 인식(HAR) 연구의 표준 벤치마크로 사용

#### 2. Data Collection and Structure
- device: Samsung Galaxy S II
- 센서 종류: 가속도계 (Accelerometer) & 자이로스코프 (Gyroscope)
- 샘플링 주파수: 50 Hz 
- 피험자 수: 30명 (19~48세)
- 활동 종류 (6 classes)
    1. Walking
    2. Walking Upstairs
    3. Walking Downstairs
    4. Sitting
    5. Standing
    6. Laying

#### 3. Dataset Composition
- 각 sample은 2.56초 (128 Time Step)의 시계열 데이터
- train: 21명 (70%)
- test: 9명 (30%)

UCI HAR Dataset/ \
│\
├── train/\
│ ├── Inertial Signals/\
│ ├── X_train.txt\
│ ├── y_train.txt\
│\
├── test/\
│ ├── Inertial Signals/\
│ ├── X_test.txt\
└ ├── y_test.txt\


#### 4. Input Format - DL
- (N, C, T)
- X.shape = (N, 9, 128)
- y.shape = (N,)

#### 5. Significance
- 다양한 딥러닝 구조의 성능 비교에 적합
- 데이터 크기 적절
- Label이 명확하여 HAR 연구의 표준 실험 환경으로 사용

#### 6. Versions
`1.0`: Test set에 자체적으로 정규화하는 문제 발생 

`1.1`: Train set의 정규화 값을 Test set에 정상적으로 정규화하도록 수정

`1.2`: Transitional Test Set에 이중 정규화되는 문제 한번만 하도록 수정

#### 7. Files
- `SSL_basic~`: 가장 기본적인 실험 파일
- `SSL_improve~`: 다수의 기능을 추가한 파일 (consistency loss, EMA, cos-warmup scheduler, learnable hyperbolic_c 등

#### 7. Reference
Anguita, D., Ghio, A., Oneto, L., Parra, X., & Reyes-Ortiz, J. L. (2013).
A Public Domain Dataset for Human Activity Recognition Using Smartphones.
21st European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning (ESANN 2013).\
https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
