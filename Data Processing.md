## 데이터 전처리

> **데이터 전처리**

- 올바른 결과 도출을 위해 데이터의 특성 간 scale을 맞춰주는 것

- 샘플링 편향(sampling bias) : 이진 분류의 경우 데이터가 한 쪽으로 치중되어 있는 것

<br>

> **1. 데이터 준비**
>
    import csv

    f = open('Fish.csv','r')
    data = csv.reader(f)
    header = next(data)

    bream_weight = []
    bream_length = [] 
    smelt_weight = []
    smelt_length = []

    for row in data:
      if row[0]=='Bream':
         bream_weight.append(float(row[1]))
         bream_length.append(float(row[3]))
      if row[0]=='Smelt':
         smelt_weight.append(float(row[1]))
         smelt_length.append(float(row[3]))

    weight = bream_weight+smelt_weight
    length = bream_length+smelt_length

<br>

> **2. 입력 데이터 준비**
>
    import numpy as np
    fish_data = np.column_stack((length,weight)) 
    
<br>

> **3. 정답 데이터 준비**
>
    fish_target = np.concatenate((np.ones(35),np.zeros(14)))

<br>

> **4. 학습, 시험 데이터 분할**

- **train_test_split()** : 비율에 맞게 학습 데이터와 시험 데이터로 분할 
- **stratify** : 한쪽으로 치우친 데이터 입력 시 알아서 편향을 조절해줌
- **random_state** : shuffle 기능 (동일한 숫자 입력 시 같은 랜덤 결과 출력)
    - 실무에선 사용 x 

>
    from sklearn.model_selection import train_test_split as tts
    train_input,test_input,train_target,test_target = tts(fish_data,fish_target,random_state=42,stratify=fish_target) 
    
<br>

> **5. 주변 샘플 데이터 살펴보기**

- distances : 주변 데이터의 거리 
- index : 주변 데이터의 인덱스

>
    distance, index = model.kneighbors([[25,150]])
<br>

> **6. 표준점수**

- 표준점수 : 각 특성 값이 평균에서 표준편차의 몇 배만큼 떨어져 있는지 나타냄 
- 가장 널리 사용하는 전처리 방법 중 하나 
- **mean()** : 평균 계산 
- **std()** : 표준편차 계산 

>
    mean = np.mean(train_input,axis=0) 
    std = np.std(train_input,axis=0)

    train_scaled = (train_input - mean) / std

<br>

> **7. 표준 점수로 변환**

- 브로드캐스팅 : 넘파이 배열 내 모든 원소에 계산 기능 수행 

>
    train_scaled = (train_input - mean) / std  #훈련 데이터
    new = ([25,150]-mean)/std  #샘플 데이터 
<br>

> **Result**

<img src="https://user-images.githubusercontent.com/105197496/211192525-872f3f9d-8462-4e95-8b14-0b8641f8bb37.png" width=320px height=240px> <img src="https://user-images.githubusercontent.com/105197496/211192516-d99d960f-60d2-47fa-8eac-8947b04edf87.png" width=320px height=240px>
