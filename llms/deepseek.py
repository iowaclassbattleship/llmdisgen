import os
from openai import OpenAI

class DeepSeek:
    available_models = [
        "deepseek-chat"
    ]

    def __init__(
            self,
            model_name,
            system_prompt = (
                "You are a scientific research assistant. Always answer as helpfully as possible, "
                "while being safe and unbiased. If a question is unclear or false, clarify or correct it."
                "Write a scientific discussion based on the given abstract with no fluff, just the discussion"
            )
        ):
        self.model_name = model_name
        self.client = OpenAI(
            api_key=os.environ.get("DEEPSEEK"),
            base_url="https://api.deepseek.com"
        )
        self.system_prompt = system_prompt

    def prompt(self, user_prompt, system_prompt):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                { "role": "system", "content": system_prompt },
                { "role": "user", "content": user_prompt }
            ]
        )

        return response.choices[0].message.content
    
LLMClass = DeepSeek