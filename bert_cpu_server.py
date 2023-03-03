from transformers import AutoTokenizer, BertForQuestionAnswering
from flask import Flask, request, jsonify

import time
import torch

app = Flask(__name__)

device = "cpu"
tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
model = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
model = model.to(device)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        query = request.get_json()['data']
        context = "My team is MLOps"
        inputs = tokenizer(query, context, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        answer_start_index = outputs.start_logits.argmax()
        answer_end_index = outputs.end_logits.argmax()
        predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
        ans = tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)
        print("Answer:", ans)
        return jsonify({"Answer": ans})


if __name__ == "__main__":
    app.run(host="192.168.219.100", port="51015",
            use_reloader=False,
            threaded=True)
