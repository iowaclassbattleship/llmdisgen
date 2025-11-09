from transformers import AutoTokenizer, LlamaForCausalLM


class OLlama:
    def __init__(
            self,
            model_name="unsloth/llama-3-8b-bnb-4bit"
        ):
        
        self.model_name = model_name
        self.model = LlamaForCausalLM.from_pretrained(model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    
    def prompt(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        generate_ids = self.model.generate(inputs.input_ids, max_length=500)

        return self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]