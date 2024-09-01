# Large-Language-Models-for-Assisting-with-Mental-Health-Diagnostic-Assessments
Exploring The Potential of Large Language Models for Assisting with Mental Health Diagnostic Assessments

## 💿 Model Link
🔗 [DiagnosticLlama Model Link](https://huggingface.co/barca-boy/primate_autotrain_mental_llama)

### ℹ️ Code Snippet for Running the Model
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "PATH_TO_MODEL_REPO"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype='auto'
).eval()

# Prompt content: "<INSERT DIAGNOSIS PROMPT>"
messages = [
    {"role": "user", "content": "<INSERT DIAGNOSIS PROMPT>"}
]

input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
output_ids = model.generate(input_ids.to('cuda'))
response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

# Model response: "Hello! How can I assist you today?"
print(response)
```

## 💾 Dataset Links
🔗 [GPT-3.5-PHQ-9](https://huggingface.co/datasets/darssanle/GPT-3.5-PHQ-9)

🔗 [GPT-4o\_mini-PHQ-9](https://huggingface.co/datasets/darssanle/GPT-4o_mini-PHQ-9)

🔗 [GPT-4o-PHQ-9](https://huggingface.co/datasets/darssanle/GPT-4o-PHQ-9)

🔗 [llama3.1\_8b-PHQ-9](https://huggingface.co/datasets/darssanle/llama-3.1_8b-PHQ-9)

🔗 [mixtral-8x7b-PHQ-9](https://huggingface.co/datasets/darssanle/mixtral-8x7b-PHQ-9)

🔗 [GPT-4o\_mini-GAD-7](https://huggingface.co/datasets/darssanle/GPT-4o_mini-GAD-7)

🔗 [GPT-4o-GAD-7](https://huggingface.co/datasets/darssanle/GPT-4o-GAD-7)

🔗 [llama3.1\_8b-GAD-7](https://huggingface.co/datasets/darssanle/llama-3.1_8b-GAD-7)

🔗 [mixtral-8x7b-GAD-7](https://huggingface.co/datasets/darssanle/mixtral-8x7b-GHD-7)
