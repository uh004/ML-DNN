## Linear_Regression 

> **Linear_Regression(선형 회귀)**

- 특성과 타깃 사이의 관계를 가장 잘 나타내는 선형 방정식을 찾음 
- 특성이 하나면 직선이 됨 
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

> **2. 훈련 데이터, 테스트 데이터 분리 및 데이터 형태 변환**
>
    from sklearn.model_selection import train_test_split as tts 
    train_i,test_i,train_t,test_t = tts(perch_length,perch_weight,random_state=42)
    train_i,test_i = train_i.reshape(-1,1),test_i.reshape(-1,1)
<br>

> **3. 선형 회귀 모델 생성 및 학습**

- model.coef_ : 직선 방정식의 기울기(회귀계수) 
- model.intercept_ : 직선 방정식의 절편

>
    from sklearn.linear_model import LinearRegression as lr
    model = lr()
    model.fit(train_i,train_t)
    print(model.coef_, model.intercept_)
<br>

> **4. 그래프로 확인**
>
    plt.scatter(train_i,train_t)
    plt.plot([15,50],[15*model.coef_+model.intercept_,50*model.coef_+model.intercept_])
    plt.scatter(50,모델이 예측한 값,marker='^')
    plt.xlabel('length')
    plt.ylabel('weight')
    plt.show()
<img src="https://user-images.githubusercontent.com/105197496/212247123-c81b90e7-6e5f-4b6e-b99e-02eb2bd055b5.png" width="320px" height="240px">
<br>

## Polynomial Regression

> **Polynomial Regression(다항회귀)**

- 특성과 타깃 사이의 관계를 가장 잘 나타내는 선형 or 비선형 방정식을 찾음
<br>

> **1. 특성을 제곱한 항을 훈련 데이터에 추가**
>
    train_poly = np.column_stack((train_i**2,train_i))
    test_poly = np.column_stack((test_i**2,test_i))
    
> **2. 모델 학습**
>
    model = lr()
    model.fit(train_poly,train_t)
<br>

> **3. 그래프로 확인**
>
    point = np.arange(15,50) # ? 
    plt.scatter(train_i,train_t)
    plt.plot(point,1.01*point**2 - 21.6*point + 116.05)
    plt.scatter(50,모델이 예측한 값,marker='^')
    plt.xlabel('length')
    plt.ylabel('weight')
    plt.show()
<img src="https://user-images.githubusercontent.com/105197496/212248992-488fd020-f4c7-4f04-a299-75b5940cd06e.png" width="320px" height="240px">
<br>
