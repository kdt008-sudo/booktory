import pickle
import os

model_path = os.path.join("model", "training_docs_quiz.pkl")
with open(model_path, "rb") as f:
    data = pickle.load(f)
print(type(data))
if isinstance(data, list):
    print("리스트 길이:", len(data))
    print("첫번째 요소 타입:", type(data[0]))
    print("첫번째 요소 내용:", data[0])
elif isinstance(data, dict):
    print("딕셔너리 키:", list(data.keys())[:10])
    print("샘플 값:", next(iter(data.values())))
else:
    print("기타 타입:", type(data))