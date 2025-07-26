# plant_disease_classification

이번에 식물 병해충 진단을 해주는 AI프로그램을 만드는 프로젝트를 진행하였다.
우선 프로젝트를 진행하기 위한 라이브러리를 불러온다.
[ pandas, os, PIL, sklearn, torch, torchvision, numpy, tqdm ]
ㄴ 해당 라이브러리를 import하고 캐글에 있는 Plant Pathology 2020 - FGVC7데이터를 가져온다
train.csv를 확인하기 위해 pandas의 read_csv함수를 사용해 데이터를 불러와 확인한다.

1. 첫번째로 진행할것은 라벨전처리이다.
2. 데이터셋 클래스 정의
3. 데이터 분할 test_size=0.2로 해주고 train_df, val_df 에 한다.
4. 이미지를 전처리해준다.
5. 데이터로더 정의.
6. 모델 정의를 하는데 cuda를 사용하고,  모델은 resnext50_32x4d로 사전학습된 모델을 쓰며, '
  fully_connected의 출력 층 클래스 수는 4개이다.
 - loss Function : CrossEntropyLoss()
 - optimizer : NAdam을 쓴다 - learning_rate는 1e-4로 0.0001
7. 학습단계 : tqdm라이브러리를 호출, 학습을 진행할때 과적합을 막기 위한 EarlyStopping 클래스 정의
   - ephoch을 50으로 잡고, 학습시작한다
8. 모델 저장

9. 이제 이미지를 예측할 차례이다. 모델 학습을 할때 데이터를 전처리 했는데 tensor로 학습됐으니 우리도 tensor값으로 전처리해주고
   이미지를 예측함수에 넣어야한다.
10. 마지막으로 예측할 이미지의경로를 설정해서, image.view(데이터갯수, 채널, 가로, 세로)이런식으로 넣어서 추론한다.
11. 직관적으로 보기위하여 label_map = ['healthy', 'multiple_diseases', 'rust', 'scab']리스트 선언해서 result를 label에 인덱싱한다/.


12. 근데 예측을 해보니 죄다 rust만 나온다 확인해보니 전체데이터 다 rust는 아니지만 rust가 높은 확률로 예측되었다.
13. 이를 해결하기 위해 예측값 아웃풋 데이터 분포를 확인해보니 1인덱스인 'multiple_diseases' 15개로 불균형되어있었다.
그래서 클래스 불균형 보완용 Sampler를 사용하여 가중치를 섞어주는 전처리 작업을 했다.
