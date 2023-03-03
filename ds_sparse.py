import time
from deepsparse import Pipeline, compile_model
from deepsparse.utils import generate_random_inputs

model_path = "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/12layer_pruned80_quant-none-vnni"

qa_pipeline = Pipeline.create(task="question-answering",
                              model_path=model_path)
start = time.time()
for _ in range(100):
    inference = qa_pipeline(question="What's your team?", context="My name is MLOps")
    print(inference)
print(time.time() - start)
