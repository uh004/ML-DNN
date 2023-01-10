## KNN-Classification

> **1. Classification(분류)**

- 지도 학습의 일종, 데이터가 어느 종류에 속하는지 판별 

  - binary classification(이진 분류) : 두 개의 클래스(=종류) 중 하나를 고르는 분류
  - multi classification(다중 분류) : 셋 이상의 클래스 중 하나를 고르는 분류 
<br>

> **2. 데이터 준비하기**

- 입력 데이터와 정답 데이터 준비하기
- 지도 학습에서는 입력 데이터 + 정답 데이터 = 훈련 데이터
- **numpy.column_stack((리스트1,리스트2))** : 두 배열을 세로로 합쳐서 2차원 배열 만들기 
- **numpy.concatenate((리스트1,리스트2))** : 두 배열을 가로로 붙이기

>
    fish_data = np.column_stack((length,weight))
    fish_target = np.concatenate((np.ones(35),np.zeros(14)))
<br>

> **3. 훈련 데이터와 테스트 데이터로 나누기**

- **train_test_split** : 리스트나 배열을 비율에 맞게 훈련, 테스트 세트로 나눔
- **stratify** : 샘플이 편향되었을 때 비율을 조절  

> 
    from sklearn.model_selection import train_test_split as tts 
    train_input, test_input, train_target, test_target = tts(fish_data,fish_target,random_state=42,stratify=fish_target)
<br>

> **4. 모델 생성 및 학습**

- **fit()** : 모델 훈련 
- **score()** : 모델 평가 

>
    from sklearn.neighbors import KNeighborsClassifier as knc
    model = knc()
    model.fit(train_input,train_target)
    model.score(test_input,test_target)
<br>

> **5. 새로운 데이터로 모델 예측**

- **predict()** : 모델 예측 

>
    model.predict([[데이터]])
