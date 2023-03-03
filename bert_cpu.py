from transformers import AutoTokenizer, BertForQuestionAnswering

import time
import torch

device = "cpu"
tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
model = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

question, text = "Who was Maru?", "Maru was a really cute puppy"
print("Question", question)
inputs = tokenizer(question, text, return_tensors="pt")
model = model.to(device)
inputs = inputs.to(device)

start = time.time()

for _ in range(100):
    with torch.no_grad():
        outputs = model(**inputs)
    answer_start_index = outputs.start_logits.argmax()
    print(outputs.start_logits)
    answer_end_index = outputs.end_logits.argmax()
    predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
    ans = tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)
    print("Answer:", ans)

print(time.time() - start)
