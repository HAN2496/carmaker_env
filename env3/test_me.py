from stable_baselines3 import SAC
import pickle

# 모델 불러오기
model = SAC.load("best_model/SLALOM_env1_best_model.pkl")

# 호환 가능한 프로토콜로 모델 다시 저장
with open("best_model/SLALOM_env1_best_model_compatible.pkl", "wb") as file:
    pickle.dump(model, file, protocol=4)
