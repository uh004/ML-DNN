## Machine - Learning : KNN 

> **KNN**

- 시험 데이터가 주어졌을 때 거리가 가장 가까운 k개의 학습 데이터를 통해 시험데이터가 어느 군집에 속하는지 파악 

- 최적의 k값을 시행착오를 통해 직접 구함 

- 일반적으로 이진 분류에서는 k값을 홀수로 지정 

- 데이터 사이의 거리 계산을 위해 유클리드 거리를 주로 사용

- 특성 사이에 스케일(규모)을 맞춰줘야 함 = 정규화 

  - x축과 y축의 눈금 간격이 다를 때 눈으로 판단했을 때와 knn이 예측한 값이 다를 수 있음

- 동작원리 

  - 회귀 : 가장 가까운 k개의 데이터의 평균값을 시험 데이터의 label로 결정 
  
  - 가중 회귀 : 시험 데이터로부터 가장 가까운 k개의 학습 데이터를 찾기
 
  ✏ 시험 데이터와 학습 데이터 사이의 거리를 고려하여 가중합으로 label값을 결정 

<br>

> **1. 데이터 준비 및 가시화**

- 해당 산점도는 Fish.csv의 weight와 length2 데이터 사용 
<img src="https://user-images.githubusercontent.com/105197496/210172667-11b603c0-19f1-4801-9a62-098cb593f021.png" width="420px">

> **2. 입력값(x), 정답값(y) 데이터 만들기**

- 입력값 데이터 만들기 
  
  - **numpy.column_stack((array1,array2))** : 두 개의 1차원 배열을 세로로 합쳐 2차원 배열로 만들기
  
> 
    import numpy as np
    length = bream_length + smelt_length
    weight = bream_weight + smelt_weight
    fish_data = np.column_stack((length,weight))
    
- 출력값 데이터 만들기

  - Bream = 1, Smelt = 0 
  
  - **np.concatenate((numpy array1, numpy array2))** : numpy 배열을 합치기 
  
> 
    fish_target = np.concatenate((np.ones(35),np.zeros(14)))
<br>

> **3. 모델 생성**

- kNeighborsClassifier(n_neighbors=숫자) 

  - n_neighbors : 가장 가까운 데이터 n개를 찾겠다는 의미

> 
    from sklearn.neighbors import KNeighborsClassifier as knc
    model = knc()
<br>

> **4. 모델 훈련**

- **model.fit(입력값,정답값)** : 입력값과 정답값을 fit메소드에 전달 (모델 훈련 시 사용)

>
    model.fit(fish_data,fish_target)
<br>

> **5. 모델의 정확도 확인**
>
    result = model.score(fish_data,fish_target)
<br>

> **6. 새로운 데이터의 정답 예측하기**
> 
    result = model.predict([[새로운 데이터]])
<br>

