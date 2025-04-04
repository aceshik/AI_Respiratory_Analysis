# AI_Respiratory_Analysis

## 전처리

### 리샘플링 + 정규화

- 다양한 샘플링 레이트(4kHz, 10kHz, 44.1kHz)가 섞여있기 때문에 분할 시점(초단위)를 sample 단위로 정확히 mapping 하기 위해서 동일한 샘플링 레이트로 통일

- annotation 파일은 "초 단위"로 주기 시작/끝을 지정하므로 오디오도 초단위 → sample index 로 정확히 변환 가능해야 함

`data/resampling.py`

###