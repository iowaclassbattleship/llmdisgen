import os
from openai import OpenAI


class OpenAIWrapper:
    available_models = [
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-3.5-turbo",
    ]

    def __init__(
        self,
        model_name,
        system_prompt=(
            "You are a scientific research assistant. Always answer as helpfully as possible, "
            "while being safe and unbiased. If a question is unclear or false, clarify or correct it. "
            "Write a scientific discussion based on the given abstract with no fluff, just the discussion."
        ),
    ):
        if model_name in self.available_models:
            self.model_name = model_name
        else:
            raise ValueError(f"Model {model_name} not in {self.available_models}")
        self.system_prompt = system_prompt

        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def prompt(self, user_prompt, system_prompt=None):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt or self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        return response.choices[0].message.content
