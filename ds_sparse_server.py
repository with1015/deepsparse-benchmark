import time

from deepsparse import Pipeline, compile_model
from deepsparse.utils import generate_random_inputs
from flask import Flask, request, make_response, jsonify

app = Flask(__name__)

model_path = "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/12layer_pruned80_quant-none-vnni"
model = Pipeline.create(task="question-answering",
                        model_path=model_path)

"""
start = time.time()
for _ in range(100):
    inference = qa_pipeline(question="What's your team?", context="My name is MLOps")
    print(inference)
print(time.time() - start)
"""

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        query = request.get_json()["data"]
        context = "My team is MLOps"
        inference = model(question=query, context=context)
        inference = dict(inference)
        print(inference)
        return jsonify(inference)

if __name__ == '__main__':
    app.run(host='192.168.219.100', port='51015',
            use_reloader=False,
            threaded=True)
