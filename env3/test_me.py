from stable_baselines3 import SAC

# 모델 로드하기 (Python 3.8 이상 필요)
model = SAC.load("best_model/SLALOM_env1_best_model.pkl")
# 호환 가능한 프로토콜 사용하여 모델 다시 저장하기
model.save("best_model/SLALOM_env1_best_model.pkl", exclude=["_vec_normalize_env"])
