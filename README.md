# MBC_floating
MBC AI 심화과정 부유물 탐지 과제 결과물
[프레젠테이션 링크](https://www.canva.com/design/DAG5wyzR7N4/pkta9K4dYsFzBCjR6M85HQ/edit?utm_content=DAG5wyzR7N4&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)


## 사용한 툴 및 환경
*   Roboflow - 객체 라벨링 및 데이터 수집, 모델 학습
*   python 11.x - 학습된 모델로 영상데이터 판독
    *   Numpy - 영상 프레임 변환
    *   inference - RoboFlow 연결용
    *   Opencv - 영상 데이터 열기 및 시각화
    *   supervision - 객체탐지 표시
    *   tqdm - 판독 진행도 시각화
    *   ONNXRuntime - 고속화
*   CUDA 12.2 - GPU 사용
*   CUDNN 8.9.7
