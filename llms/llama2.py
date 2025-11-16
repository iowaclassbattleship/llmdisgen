from transformers import AutoTokenizer, LlamaForCausalLM
import torch

available_models = [
    "meta-llama/Llama-2-7b-chat-hf",
]

class Llama2LLM:
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
        self.system_prompt = (
            "You are a scientific research assistant. Always answer as helpfully as possible, "
            "while being safe and unbiased. If a question is unclear or false, clarify or correct it."
            "Write a scientific discussion based on the given abstract with no fluff, just the discussion"
        )

    def prompt(self, user_prompt):
        inputs = self.tokenizer(
            self.format_prompt(user_prompt),
            return_tensors="pt",
            return_attention_mask=True
        ).to(self.model.device)

        input_ids = inputs["input_ids"]

        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=inputs["attention_mask"],
            temperature=0.7,
            do_sample=True,
        )

        new_tokens = outputs[0][input_ids.shape[1]:]

        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    
    def format_prompt(self, user_prompt):
        return (
            f"[INST] <<SYS>>\n"
            f"{self.system_prompt}\n"
            f"<</SYS>>\n\n"
            f"{user_prompt} [/INST]"
        )
