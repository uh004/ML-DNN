## KNN-Regression

> **Regression(회귀)**

- 임의의 숫자 예측 
- 가장 가까운 샘플을 찾아 평균내서 정답을 구함 

    - 새로운 샘플이 훈련 세트의 범위를 벗어나면 이상한 값을 예측할 수 있다는 단점 존재 

<br>

> **1. 데이터 준비하기**

>
    import csv
    import numpy as np

    f = open('Fish.csv','r')
    data = csv.reader(f)
    header = next(data)

    perch_weight = []
    perch_length = []

    for row in data:
        if row[0]=='Perch':
            perch_weight.append(float(row[1]))
            perch_length.append(float(row[3]))

    perch_length,perch_weight = np.array(perch_length),np.array(perch_weight)
<br>

> **2. 훈련 데이터와 테스트 데이터로 나누기**

>
    from sklearn.model_selection import train_test_split as tts
    train_input,test_input,train_target,test_target = tts(perch_length,perch_weight,random_state=42)
<br>

> **3. 데이터 형태 변환**

- 데이터가 1차원인 경우 2차원으로 변경
- reshape에 -1 사용 시 나머지 원소 개수로 자동 채우기

>
    train_input = train_input.reshape(-1,1)
    test_input = test_input.reshape(-1,1)
<br>

> **4. 모델 생성 및 학습**

- 결정계수 : 1 - (타깃-예측)제곱의 합/(타깃-평균)제곱의 합

    - 회귀 문제의 성능 측정 도구 (R의 제곱으로 표현)

>
    from sklearn.neighbors import KNeighborsRegressor as knr 
    model = knr()
    model.fit(train_input,train_target)
    model.score(test_input,test_target)
<br>

> **5. 평균 절댓값 오차 계산**

- **mean_absolute_error** : 평균 절댓값 오차 계산 
>
    from sklearn.metrics import mean_absolute_error as mae  
    test_prediction = model.predict(test_input)
    maes = mae(test_target,test_prediction) # 타깃과 예측의 절댓값 오차 반환
<br>

> **6. 과대적합 및 과소적합 판별**

- **과대적합** : 훈련 데이터 정확도가 시험 데이터 정확도보다 압도적으로 뛰어난 경우  
- **과소적합** : 훈련 데이터 정확도보다 시험 데이터 정확도가 높거나 두 정확도 모두 낮은 경우
  
  - 해당 모델에서는 과소적합 발생
>
    print(model.score(test_input,test_target))
    print(model.score(train_input,train_target))

> **7-1. 과소적합 해결 및 모델 재학습**

- 모델의 복잡도 줄이기 = 이웃의 개수 줄이기
  - **n_neighbors** : 이웃의 개수 설정 

>
    model.n_neighbors = 3 #기본값은 5 
    model.fit(train_input,train_target)
    print(model.score(train_input,train_target))
    print(model.score(test_input,test_target))
<br>

> **7-2. 과대적합 해결 및 모델 재학습**

- ㅇ
