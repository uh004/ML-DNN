## Machine - Learning : KNN 

> **KNN**

-  

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

