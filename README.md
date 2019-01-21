# Path To ML Professional
## by Sun-il Kim, 2019-01-02

수학을 (거의) 모르는 기초부터 시작하여, 지금 시점의 최신 알고리즘 (GAN, Neural ODE 등)까지 이해를 마치고 실무에 적용할 수 있도록 트레이닝 하는 것이 목적입니다. 

- 기본적으로 3주 코스이며, 과거 수포자를 대상으로 만들어진 커리큘럼입니다.
- 다만 이번에도 수학을 포기하면 안됩니다. :)
- 말이 3주이지, 풀타임 (소위 9 to 6) 기준으로 학습하는 것을 전제로 합니다. 


## 목차
1. [인공지능을 위한 수학](https://github.com/apostkor/Path_To_ML_Professional/blob/master/README.md#1-%EC%9D%B8%EA%B3%B5%EC%A7%80%EB%8A%A5%EC%9D%84-%EC%9C%84%ED%95%9C-%EC%88%98%ED%95%99)  
2. [미적분 기초](https://github.com/apostkor/Path_To_ML_Professional/blob/master/README.md#2-%EB%AF%B8%EC%A0%81%EB%B6%84-%EA%B8%B0%EC%B4%88) 
3. [Python 기초](https://github.com/apostkor/Path_To_ML_Professional/blob/master/README.md#3-python-%EA%B8%B0%EC%B4%88)
4. [데이터 사이언스를 위한 Python 활용](https://github.com/apostkor/Path_To_ML_Professional/blob/master/README.md#4-%EB%8D%B0%EC%9D%B4%ED%84%B0-%EC%82%AC%EC%9D%B4%EC%96%B8%EC%8A%A4%EB%A5%BC-%EC%9C%84%ED%95%9C-python-%ED%99%9C%EC%9A%A9)
5. [데이터 사이언스와 수학의 결합](https://github.com/apostkor/Path_To_ML_Professional/blob/master/README.md#5-%EB%8D%B0%EC%9D%B4%ED%84%B0-%EC%82%AC%EC%9D%B4%EC%96%B8%EC%8A%A4%EC%99%80-%EC%88%98%ED%95%99%EC%9D%98-%EA%B2%B0%ED%95%A9) 
6. 텐서플로우  
7. 딥러닝과 최신 알고리즘 

# 1. 인공지능을 위한 수학 
> (1월 3일)

[표지사진](http://image.kyobobook.co.kr/images/book/xlarge/282/x9788965402282.jpg)
- 표지가 다소 취향존중이지만 실제로는 좋은 책입니다. 
- 기본적인 수학 능력은 당연히 모든 컴퓨터공학 분야에서 기본입니다. ML에서는 특히 중요합니다. 
- 해당 서적은 기본적인 수1/수2 수학의 개념을 설명하면서, 인공지능 분야에서는 이 개념이 어떻게 응용되는지 교육합니다. 

#### 학습방법
- 해당 서적을 구입해서 읽어보셔도 되며, 해당 서적을 요약한 요약본은 연락 주시면 보내드립니다. 


# 2. 미적분 기초 
> (1월 4일)

![Alt text](https://upload.wikimedia.org/wikipedia/commons/f/f7/Infinitesimal_Calculus_6.png)
- 미적분은 크게 보자면 어려운 문제를 쉽게 만들고자 만들어진 기술입니다.  
- 당연히(?) 어려운 문제들인 ML에 접근하려면 필수적입니다. 

#### 학습방법
[Essence of calculus by 3Blue1Brown](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr)
- 해당 링크의 영상들을 1번부터 12번까지 차근차근 이해해 나가시면 됩니다.
- 개념과 더불어 문제에 대한 접근방식 자체를 배워보시는 것이 좋습니다.


# 3. Python 기초
> (1월 7일 ~ 1월 9일)

[아나콘다 + 주피터 노트북 환경설정](https://m.blog.naver.com/PostView.nhn?blogId=ndb796&logNo=221277853081&proxyReferer=https%3A%2F%2Fwww.google.com%2F)
- 먼저 Python에 입문하기 전에 환경설정을 아나콘다+주피터 노트북으로 해놓고 시작하는 것이 좋습니다.
- 추후에도 계속 해당 환경으로 작업할 것이기 때문에, 처음부터 익숙해지시는 것이 좋습니다. 
- 아나콘다는 최신 버전으로 설치하셔도 무방하나, Python 버전을 내릴 필요가 있습니다 ([방법](http://thrillfighter.tistory.com/466) 저는 3.6.2로 사용합니다.)

#### 학습방법
> 먼저 두 가지 좋은 Python 강의들이 있습니다.
1. [MOOC for Python](https://www.youtube.com/watch?v=EyAHKYqrEe8&list=PLBHVuYlKEkUJvRVv9_je9j3BpHwGHSZHz&index=1)
2. [윤인성 for Python](https://www.youtube.com/watch?v=XPuVxEpr-vc&index=1&list=PLBXuLgInP-5nbu5s5TuNbD6-4qh3Mgoor)

해당 강의들을 저희가 교육하기에 이상적인 순서로 재편성 하였습니다. 
1. MOOC 1~2화 Python 인트로 
2. 윤인성 1~4화 개발환경 - VSCode/3.X버전으로 진행하세요.
3. MOOC 16~17화 변수/상수/자료형 개념 
4. 윤인성 4~53화 기본 문법 
5. MOOC 52~57화 모듈 개념 
6. MOOC 58~59화 예외처리 개념 
7. MOOC 60~61화 파일처리 개념 
8. MOOC 62화 CSV 개념 
9. MOOC 63~71화 웹크롤링 기초개념

해당 순서로 학습하시면 좋겠습니다. 


# 4. 데이터 사이언스를 위한 Python 활용
> (1월 10일 ~ 1월 11일)
![Alt text](https://upload.wikimedia.org/wikipedia/commons/4/40/Twitter_activity_of_Donald_Trump.png)
~~트럼프 대통령의 트위터 근황~~
- 먼저 앞서 배운 Python 문법으로, 일종의 "ML 맛보기"를 진행해봅니다. 
- 총 두 가지 예제를 처음부터 응용까지 실습하시면 됩니다. 

#### 학습방법
1. [Twitter API를 사용한 SNS 감정도 분석 by apost](https://github.com/apostkor/TwitterKeywordAnalyzer/blob/master/TwitterKeywordAnalyzer_Main.ipynb)
2. [SVM을 이용한 주식시장 분석 by apost](https://github.com/apostkor/StockPriceAnalyzer/blob/master/StockPriceAnalyzer_Main.ipynb)
3. [참고영상 - Siraj Raval의 Python for Data Science](https://www.youtube.com/watch?v=T5pRlIbr6gg&list=PL2-dafEMk2A6QKz1mrk1uIGfHkC1zZ6UU&index=1)

- 위의 두 가지를 실습해 보시면서 아직 디테일 자체를 이해하실 필요는 없습니다. 
- (예를 들어, SVM이 어떻게 동작하는지, 연관성이 어떻게 계산되었는지 등.) 
- 다만 그냥 예시가 진행되는 진행방향과, 배운 문법들이 이렇게 사용되고 있구나 등 정말 "맛보기"를 진행하시면 됩니다. 
- 모든 예시는 참고영상에서 거론된 것이기 때문에 1~6화는 꼭 참고하시면 좋겠습니다.


# 5. 데이터 사이언스와 수학의 결합 
- 지금부터는 하루하루 자세히 진행하겠습니다.
- 이제부터는 모든 개념에서 수학/논리 자체를 이해하는게 중요합니다. 
- 어떤 논리에서 나온 것인지, 어떠한 데이터 유형에서 사용되는지, 장단점은 무엇인지. 등등
![Alt text](https://c1.staticflickr.com/5/4169/34764532445_e3883bd446_b.jpg)


> (1월 14일)
#### 5-1 학습방법
> [The Math of Intelligence by Siraj Raval의 1~4화](https://www.youtube.com/watch?v=xRJCOz3AfYY&list=PL2-dafEMk2A7mu0bSksCGMJEmeddU_H4D)

해당 플레이리스트에서 1~4화까지를 보시고 직접 영상에 나오는 실습을 직접 해보시면 됩니다. 
- Labeled Data  
- Gradient Decent 최적화 (선형회귀)
- Second Order 최적화
- SVM Classification 
- Logistic Regression (로지스틱회귀)

상기 개념을 이해시는 것이 목적입니다. 

- [실습 진행 시 필수 참고 "The Math of Intelligence Step #1 by apost"](https://github.com/apostkor/MathofIntelligence/blob/master/The%20Math%20of%20Intelligence%20Step%20%231.ipynb)

해당 노트북은 꼭 참고 해 주세요.


> (1월 15일)
#### 5-2 학습방법
> [The Math of Intelligence by Siraj Raval의 5~8화](https://www.youtube.com/watch?v=xRJCOz3AfYY&list=PL2-dafEMk2A7mu0bSksCGMJEmeddU_H4D)

해당 플레이리스트에서 5~8화까지를 보시고 직접 영상에 나오는 실습을 직접 해보시면 됩니다. 
- Vectors
- Tensors 
- K-Means Clustering
- NN
- Convolutional NN (CNN)
- ReLU

상기 개념을 이해시는 것이 목적입니다. 

- [실습 진행 시 필수 참고 "The Math of Intelligence Step #2 by apost"](https://github.com/apostkor/MathofIntelligence/blob/master/The%20Math%20of%20Intelligence%20Step%20%232.ipynb)

해당 노트북은 꼭 참고 해 주세요.


> (1월 16일)
#### 5-3 학습방법
> [The Math of Intelligence by Siraj Raval의 9~11화](https://www.youtube.com/watch?v=xRJCOz3AfYY&list=PL2-dafEMk2A7mu0bSksCGMJEmeddU_H4D)

해당 플레이리스트에서 9~11화까지를 보시고 직접 영상에 나오는 실습을 직접 해보시면 됩니다. 
- Dimensionality Reduction
- Feature Scaling Formula
- Eigenvector
- RNN
- Probability Theory
- Naive Bayes
- Condition Indep

상기 개념을 이해시는 것이 목적입니다. 

- [실습 진행 시 필수 참고 "The Math of Intelligence Step #3 by apost"](https://github.com/apostkor/MathofIntelligence/blob/master/The%20Math%20of%20Intelligence%20Step%20%233.ipynb)

해당 노트북은 꼭 참고 해 주세요.


> (1월 17일)
#### 5-4 학습방법
> [The Math of Intelligence by Siraj Raval의 12~14화](https://www.youtube.com/watch?v=xRJCOz3AfYY&list=PL2-dafEMk2A7mu0bSksCGMJEmeddU_H4D)

해당 플레이리스트에서 12~14화까지를 보시고 직접 영상에 나오는 실습을 직접 해보시면 됩니다. 
- Random Forest
- Decision Tree
- Gini Index
- Hyperparameter optimization
- Bayesian Optimization
- Gaussian Mixture Model

상기 개념을 이해시는 것이 목적입니다. 

- [실습 진행 시 필수 참고 "The Math of Intelligence Step #4 by apost"](https://github.com/apostkor/MathofIntelligence/blob/master/The%20Math%20of%20Intelligence%20Step%20%234.ipynb)

해당 노트북은 꼭 참고 해 주세요.


> (1월 18일)
#### 5-5 학습방법
> [The Math of Intelligence by Siraj Raval의 15~19화](https://www.youtube.com/watch?v=xRJCOz3AfYY&list=PL2-dafEMk2A7mu0bSksCGMJEmeddU_H4D)

해당 플레이리스트에서 15~19화까지를 보시고 직접 영상에 나오는 실습을 직접 해보시면 됩니다. 
- Generative Model (GAN)
- LDA
- LSTM
- DEEP Q Learning 
- Genetic Algorithm
- Quantum Algorithm 

상기 개념을 이해시는 것이 목적입니다. 

- [실습 진행 시 필수 참고 "The Math of Intelligence Step #5 by apost"](https://github.com/apostkor/MathofIntelligence/blob/master/The%20Math%20of%20Intelligence%20Step%20%235.ipynb)
- [실습 진행 시 필수 참고 "The Math of Intelligence Step #6 by apost"](https://github.com/apostkor/MathofIntelligence/blob/master/The%20Math%20of%20Intelligence%20Step%20%236.ipynb)

해당 노트북은 꼭 참고 해 주세요.


# 6. 텐서플로우, 딥러닝 및 최신 알고리즘  
![Alt text](https://cdn-images-1.medium.com/max/1600/0*a6XSwHsfvz_oWSSJ.jpg)
- Scikit-learn 등 기타 머신러닝 라이브러리(모듈)들은 이미 5-1부터 5-2에서 많이 사용해보셨습니다. 
- 이제 드디어 Tensorflow를 가용해서 학습/실습을 진행합니다.
- 몇몇 글이 조금 오래되어서 Tensorflow와 Keras를 별도로 칭하는데, 지금은 통합되어 Tenorflow의 Keras Implementation으로 tf.keras를 사용합니다.
<pre><code>import tensorflow as tf
from tensorflow.keras import layers
</code></pre>
- 이 부분만 참고하여 진행해 주세요. :)


> (1월 21일)
#### 6-1 학습방법
> [Intro to Tensorflow by Siraj Ravel 플레이리스트에서](https://www.youtube.com/watch?v=xRJCOz3AfYY&list=PL2-dafEMk2A7mu0bSksCGMJEmeddU_H4D)
- "Deep Learning Frameworks Compared" 
- "텐서플로우 5분완성"
- "The Besy way to Prepare a Dataset"

> [Intro to Deep Learning by Siraj Ravel 플레이리스트에서](https://www.youtube.com/watch?v=vOppzHpvTiQ&list=PL2-dafEMk2A7YdKv4XfKpfbTH5z6rEEj3)
- "How to Make Data Amazing - Intro to Deep Learning #5"

의 제목을 찾아서 실습하시면 되며, 하단 포인트들을
- 유명 라이브러리(모듈) 간 비교
- 여러가지 데이터 전처리 방법 
- MNIST(숫자인식) Tensorflow로 접근하기 

- [실습 진행 시 필수 참고 "Intro To Tensorflow Intro by apost"](https://github.com/apostkor/IntroToTensorflow/blob/master/IntroToTensorflow_Intro.ipynb)
- [실습 진행 시 필수 참고 "Intro To Tensorflow Step #1 by apost"](https://github.com/apostkor/IntroToTensorflow/blob/master/Intro%20To%20Tensorflow%20Step%20%23%201.ipynb)

해당 노트북을 꼭 참고해 실습 진행하세요. 


> (1월 22일)
#### 6-2 학습방법
> [Intro to Deep Learning by Siraj Ravel 플레이리스트에서](https://www.youtube.com/watch?v=vOppzHpvTiQ&list=PL2-dafEMk2A7YdKv4XfKpfbTH5z6rEEj3)
- "Tensorboard Explained in 5 Min" 
- "How to make a Tensorflow Neural Network"
- (추가예정)

의 제목을 찾아서 실습하시면 되며, 하단 포인트들을
- Tensorboard (텐서보드) 개념과 사용방법
- Tensorflow로 Neural Network 구현
- (추가예정)

- (추가예정)

해당 노트북을 꼭 참고해 실습 진행하세요. 


> (1월 23일)
#### 6-3 학습방법
> [Intro to Deep Learning by Siraj Ravel 플레이리스트에서](https://www.youtube.com/watch?v=vOppzHpvTiQ&list=PL2-dafEMk2A7YdKv4XfKpfbTH5z6rEEj3)
- "How to Do Sentiment Analysis - Intro to Deep Learning #3" 
- "How to Predict Stock Prices Easily - Intro to Deep Learning #7"
- "How to Use Tensorflow for Time Series (Live)"

의 제목을 찾아서 실습하시면 되며

- (추가예정)

해당 노트북을 꼭 참고해 실습 진행하세요. 
