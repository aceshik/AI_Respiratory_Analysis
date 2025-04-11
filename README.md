# AI 기반 호흡 분석 시스템

## 1. 프로젝트 개요

이 프로젝트는 호흡기 소리 데이터로부터 Crackle 및 Wheeze를 분류하는 AI 모델을 개발하는 것을 목표로 한다. 사용자는 다양한 증강 기법과 모델 구조 실험을 통해 성능을 향상시키고 있으며, 모델은 PyTorch 기반의 CNN + BiLSTM 앙상블 구조로 구성되어 있다.

---

## 2. 데이터 구성 및 전처리

- **원본 데이터**: `.wav` 파일과 주기별 annotation `.txt` 파일로 구성됨 (총 920개)
- **Sampling rate**: 모든 오디오를 4kHz로 리샘플링
- **정규화**: [-1, 1] 범위로 정규화 후, 학습 데이터 기준 평균-표준편차 정규화 적용
- **세그먼트 분할**: 각 호흡 주기를 annotation 기준으로 분할
- **특징 추출**: MFCC (13차원, (B, 13, T))
- **클래스 구성**: [Normal, Crackle, Wheeze]로 3-class 분류

---

## 3. 증강 기법 및 클래스 균형 조정

- **Wheeze 클래스 증강**: pitch shifting 적용 (Time shifting도 시도하였으나 Wheeze 모양 특성상 의미가 없었음)
- **Crackle 클래스 증강**: Crackle 모양이 비교적 뚜렷하여 일단은 복제해서 증강
- **타겟 분포**: 각 클래스별 약 1500개로 균형 조정

---

## 4. 진행중인 모델: EnhancedCNNEnsemble(CNN 3층, LSTM 2층)

```python
class EnhancedCNNEnsemble(nn.Module):
```

## 모델 개요

EnhancedCNNEnsemble 모델은 CNN(Convolutional Neural Network)과 BiLSTM(Bidirectional Long Short-Term Memory)을 결합한 형태로, 시간적 특성과 주파수 특성을 동시에 학습하는 구조. 주어진 입력은 MFCC 기반의 특성으로 (B, 13, 253)의 형태를 가지며, 이 모델은 이를 이용하여 정상 호흡, Crackle, Wheeze 세 클래스를 분류.

### 4.1. 입력 전처리

```python
self.concat_layer = nn.Conv1d(13, 128, kernel_size=1, stride=1)
```

- 입력 데이터는 (B, 13, 253)의 형태로 들어오며, 여기서 13은 MFCC 계수 개수.
- `nn.Conv1d(13, 128, 1)`을 통해 13채널을 128채널로 확장.
- 이 연산은 일종의 선형 projection 역할을 하며, 채널 차원을 늘려 표현력을 확장.

---

### 4.2. CNN Feature Extractor

### Conv1 Layer

```python
self.conv1 = nn.Sequential(
    nn.Conv1d(128, 64, kernel_size=32, stride=2),
    nn.BatchNorm1d(64),
    nn.Dropout(dropout),
    nn.ReLU()
)
```

- 커널 사이즈가 32로 크기 때문에 시간 축의 넓은 패턴을 학습할 수 있음.
- `BatchNorm1d`는 학습 안정화 및 수렴 속도 향상에 기여.
- `Dropout`은 과적합을 방지.
- `ReLU`는 비선형성을 부여하여 모델의 표현력을 높임.

### Conv2 Layer

```python
self.conv2 = nn.Sequential(
    nn.Conv1d(64, 128, kernel_size=32, stride=2),
    nn.BatchNorm1d(128),
    nn.Dropout(dropout),
    nn.ReLU()
)
```

- Conv1의 출력에서 보다 복잡한 패턴을 추출.
- 채널 수를 128로 증가시켜 더 많은 feature를 담도록 함.

### Conv3 Layer

```python
self.conv3 = nn.Sequential(
    nn.Conv1d(128, 256, kernel_size=32, stride=1),
    nn.BatchNorm1d(256),
    nn.Dropout(dropout),
    nn.ReLU()
)
```

- 깊이 있는 특징 추출을 수행.
- stride가 1로 줄어들어 더 세밀한 시퀀스 정보를 유지.

---

### 4.3. LSTM Feature Extractor

Conv3까지의 출력을 시계열로 다루기 위해 permute 연산을 사용.

```python
x = x.permute(0, 2, 1)  # (B, T, 256)
```

### LSTM 1

```python
self.lstm1 = nn.LSTM(
    input_size=256,
    hidden_size=128,
    num_layers=2,
    batch_first=True,
    dropout=dropout,
    bidirectional=True
)
```

- 첫 번째 BiLSTM은 시계열 상의 양방향 정보를 추출.
- `hidden_size=128`, `bidirectional=True`이므로 출력은 (B, T, 256)이 됨.

### LSTM 2

```python
self.lstm2 = nn.LSTM(
    input_size=128*2,
    hidden_size=128,
    num_layers=2,
    batch_first=True,
    dropout=dropout,
    bidirectional=True
)
```

- 두 번째 BiLSTM은 앞선 시퀀스 출력을 다시 한 번 정제.

---

## 4.4. 시계열 요약 (Last Timestep)

```python
x = x[:, -1, :]  # (B, 256)
```

- 시계열 전체가 아닌 마지막 timestep의 출력을 사용하여 요약 벡터로 활용.

---

### 4.5. Fully Connected Layers

### FC1

```python
self.fc1 = nn.Sequential(
    nn.Linear(128 * 2, 128),
    nn.BatchNorm1d(128),
    nn.Dropout(dropout),
    nn.ReLU()
)
```

- LSTM의 출력(256)을 128차원으로 축소하며,
- `BatchNorm1d`와 `Dropout`, `ReLU`로 정규화 및 비선형성 부여.

### FC2

```python
self.fc2 = nn.Linear(128, num_classes)
```

- 최종 예측 결과를 3개 클래스 (Normal, Crackle, Wheeze)로 분류.

---

### 4.6. 순전파 흐름 요약 (Forward Pass)

```python
x = self.concat_layer(x)
x = self.conv1(x)
x = self.conv2(x)
x = self.conv3(x)
x = x.permute(0, 2, 1)
x, _ = self.lstm1(x)
x, _ = self.lstm2(x)
x = x[:, -1, :]
x = self.fc1(x)
x = self.fc2(x)
return x
```

- 입력 데이터를 CNN으로 처리해 국소적인 시간 패턴을 추출,
- LSTM을 통해 시퀀스 정보를 학습,
- 마지막 timestep의 출력을 fully connected layer에 전달하여 최종 클래스를 예측.

---

## 요약

- CNN을 통해 고차원 로컬 feature 추출
- BiLSTM을 통해 시간적 문맥 정보 보존
- BatchNorm1d: 학습 안정화, 일반화
- Dropout: 과적합 방지
- ReLU: 비선형성 부여로 학습 능력 강화
- 마지막 timestep 사용으로 메모리 및 계산 효율화


## 5. 학습 전략

- **Loss Function**: FocalLoss (class imbalance 완화)
  - alpha는 클래스별 inverse frequency 기반 가중치
  - gamma=1.0 사용

- **Optimizer**: Adam (lr=1e-3, weight_decay=1e-5)
- **Scheduler**: ReduceLROnPlateau (성능 정체 시 lr 감소)
- **EarlyStopping**: Val F1 기준 patience=6 적용
- **Dropout 조정**: Epoch 중간에 dropout 값을 0.3 → 0.2로 조정 (성능 향상 관찰됨)

---

## 6. Threshold 기반 추론 방식

```python
thresholds = [0.5, 0.45, 0.4]
```
- **클래스별 개별 threshold 적용**: 각 클래스의 softmax 확률이 해당 임계값을 넘으면 예측
- **예시**: Crackle이 0.45보다 낮으면 무시되며, 그 외 높은 확률 중 하나가 선택됨

---

## 7. 성능 요약

- **최종 모델 성능 (Threshold 적용)**:
```
Normal  : precision 0.80, recall 0.75
Crackle : precision 0.75, recall 0.68
Wheeze  : precision 0.58, recall 0.68
Macro avg F1: 0.70
```

- Epoch 15에서 best model 저장됨 (과적합 전 최적 시점)

---

## 8. 추후 실험 방향

- Crackle precision 향상
- Feature map 시각화 및 attention 영역 확인
- Self-training 또는 pseudo-labeling 기반 semi-supervised 학습
- Multi-branch 모델로 wheeze / crackle path 분기 실험