import transformers
import torch

available_models = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct"
]

class Llama3LLM:
    def __init__(self, model_name):
        self.model_name = model_name
        self.system_prompt = (
            "You are a scientific research assistant. Always answer as helpfully as possible, "
            "while being safe and unbiased. If a question is unclear or false, clarify or correct it."
            "Write a scientific discussion based on the given abstract with no fluff, just the discussion"
        )

        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model_name,
            model_kwargs={
                "dtype": torch.float8_e4m3fn,
                "quantization_config": {"load_in_4bit": True},
                "low_cpu_mem_usage": True,
            },
        )

    def prompt(self, user_prompt):
        messages = [
            { "role": "system", "content": self.system_prompt },
            { "role": "user", "content": user_prompt }
        ]

        prompt = self.pipeline.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        
        outputs = self.pipeline(
            prompt,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        
        return outputs[0]["generated_text"][len(prompt):]