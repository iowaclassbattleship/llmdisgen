from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class OLlama:
    def __init__(
            self,
            model_name="unsloth/llama-3-8b-bnb-4bit"
        ):
        
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    
    def prompt(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=1000,
                do_sample=True,
                temperature=0.7
            )

        return self.tokenizer.decode(output[0], skip_special_tokens=True)