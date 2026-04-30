# Plant Disease Classification

Plant Pathology 2020 - FGVC7 데이터셋을 활용해 사과 잎 이미지를 4개 질병 상태로 분류하는 딥러닝 프로젝트입니다.

## 프로젝트 개요

이 프로젝트는 잎 이미지 한 장을 입력받아 아래 클래스 중 하나로 분류합니다.

- `healthy`: 정상 잎
- `multiple_diseases`: 복합 질병
- `rust`: 녹병
- `scab`: 검은별무늬병

모델은 ImageNet 사전학습 가중치를 사용한 `ResNeXt50_32x4d` 기반 전이학습 방식으로 구성했습니다.

## 사용 데이터

데이터셋: Plant Pathology 2020 - FGVC7

학습 데이터의 클래스 분포는 불균형합니다. 특히 `multiple_diseases` 클래스의 샘플 수가 적어, 해당 클래스에 더 강한 augmentation과 샘플 복제를 적용했습니다.

## 주요 구현 내용

- `pandas`로 `train.csv` 라벨 로드
- `PIL`과 `torchvision.transforms`를 활용한 이미지 전처리
- train/validation 데이터 분리
- 클래스별 라벨 매핑 생성
- `multiple_diseases` 클래스 보완을 위한 데이터 증강
- ImageNet 사전학습 `ResNeXt50_32x4d` 모델 사용
- 마지막 fully connected layer를 4개 클래스 출력으로 교체
- `CrossEntropyLoss` 기반 학습
- Early Stopping 적용
- validation 성능 평가 및 classification report 출력
- 학습된 모델을 `resnext_model.pth`로 저장

## 모델 설정

```python
MODEL = ResNeXt50_32x4d
WEIGHTS = ImageNet1K V1
NUM_CLASSES = 4
IMAGE_SIZE = 224x224
NUM_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 3
```

## 성능

노트북 실행 결과 기준 validation accuracy는 약 `93.97%`입니다.

클래스별 성능에서 `rust`는 높은 정밀도와 재현율을 보였고, `multiple_diseases`는 데이터 수가 적어 다른 클래스보다 상대적으로 낮은 성능을 보였습니다.

## 파일 구성

```text
.
├── main.ipynb       # 학습, 평가, 예측 코드가 담긴 노트북
├── README.md        # 프로젝트 설명
└── .gitattributes
```

데이터셋 이미지, CSV 파일, 모델 가중치(`.pth`)는 용량 문제로 저장소에 포함하지 않았습니다.

## 실행 방법

1. Plant Pathology 2020 - FGVC7 데이터셋을 준비합니다.
2. `main.ipynb`와 같은 위치에 `images/`, `train.csv`, `test.csv`를 배치합니다.
3. 필요한 라이브러리를 설치합니다.

```bash
pip install pandas numpy pillow scikit-learn torch torchvision tqdm matplotlib
```

4. Jupyter Notebook에서 `main.ipynb`를 순서대로 실행합니다.

## 사용 라이브러리

- Python
- pandas
- numpy
- Pillow
- scikit-learn
- PyTorch
- torchvision
- tqdm
- matplotlib

## 개선 방향

- 불균형 클래스에 대한 추가 데이터 확보
- confusion matrix 기반 오류 분석
- 더 다양한 augmentation 실험
- EfficientNet, ConvNeXt 등 다른 backbone과 성능 비교
- 학습 코드와 추론 코드를 `.py` 파일로 분리
