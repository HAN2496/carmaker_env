# Hireachical RL 


## 1. 특이사항
### A. 두 개의 rl 환경을 관리하기 위해 carmaker_data를 통해 데이터를 관리함.
### B. b level 알고리즘: 시야 끝 지점에 해당하는 위치에 trajectory point를 생성한 뒤 그 사이를 선형 보간.

## 2. 중요 파일 및 디렉토리
### A. {road name}_cone.py: 도로, 콘, 차량 클래스가 정의되어 각 테스트 시나리오별 가상 환경을 제작하는 코드
### B. {road_name}_data.py: simulink를 통해 넘어온 데이터를 관리하는 코드.
### C. {road_name}_b.py: b level 환경
### D. {road_name}_low.py: low level 환경
### ** SLALOM_cone1.py는 b level env에 필요한 도로 가상 환경 코드임.