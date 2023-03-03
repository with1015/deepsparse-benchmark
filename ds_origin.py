import time
import argparse

from deepsparse import Pipeline, compile_model
from deepsparse.utils import generate_random_inputs

model_path = "./bertsquad-12"

qa_pipeline = Pipeline.create(task="question-answering",
                              model_path=model_path)

start = time.time()
for _ in range(100):
    inference = qa_pipeline(question="What's your team?", context="My team is MLOps")
    print(inference)
print(time.time() - start)
