# Hireachical RL 


## 1. 특이사항
### A. 두 개의 rl 환경을 관리하기 위해 carmaker_data를 통해 데이터를 관리함.
### B. Bezier curve를 통해 경로를 생성하고 제작. (현재 이 방법은 성공하지 못했으며, 

## 2. 중요 파일 및 디렉토리
### A. carmkaer_cone.py - 도로, 도로 선, 콘, 차량 클래스가 정의되어 각 테스트 시나리오별 가상 환경을 제작하는 코드
### B. carmaker_data.py - simulink를 통해 넘어온 데이터를 모두 관리하는 코드로, carmkaer_env_low.py, carmkaer_env_b.py의 state, reward를 모두 관리함. 또한, pygame을 통해 시각화시킬 수 있음.
### C. carmaker_trajectory.py - trajectory를 생성하는 코드. low level에서는 trajectory데이터를 불러오며, b level에서는 trajectory를 생성한다.
### D. carmkaer_env_b.py - b level 환경
### E. carmkaer_env_low.py - low level 환경
