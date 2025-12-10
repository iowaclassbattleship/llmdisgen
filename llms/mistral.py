import torch
from transformers import AutoModelForCausalLM
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest

class Mistral:
    available_models = [
        "mistralai/Mistral-7B-Instruct-v0.2"
    ]
    system_prompt = ""

    def __init__(
            self,
            model_name,
            device="cuda",
            system_prompt=(
                "You are a scientific research assistant. Always answer as helpfully as possible, "
                "while being safe and unbiased. If a question is unclear or false, clarify or correct it."
                "Write a scientific discussion based on the given abstract with no fluff, just the discussion"
            )
        ):
        self.device = device
        self.model_name = model_name
        
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        try:
            self.model.to(device)
        except torch.OutOfMemoryError as e:
            print("Not enough memory on CUDA, proceeding with CPU")
            self.device = "cpu"
            self.model.to("cpu")

        self.tokenizer = MistralTokenizer.v1()
        self.system_prompt = system_prompt

    def prompt(self, user_prompt: str, system_prompt: str):
        completion_request = ChatCompletionRequest(
            messages=[UserMessage(
                content=self.format_prompt(user_prompt, system_prompt)
            )]
        )

        encoded = self.tokenizer.encode_chat_completion(completion_request)
        input_ids = torch.tensor([encoded.tokens]).to(self.device)

        generated_ids = self.model.generate(
            input_ids,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7
        )
        
        return self.tokenizer.decode(generated_ids[0].tolist())
    
    def format_prompt(self, user_prompt, system_prompt):
        return (
            f"[INST] <<SYS>>\n"
            f"{system_prompt}\n"
            f"<</SYS>>\n\n"
            f"{user_prompt} [/INST]"
        )
    
LLMClass = Mistral