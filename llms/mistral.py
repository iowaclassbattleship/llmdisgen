from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest

from huggingface_hub import snapshot_download
from pathlib import Path

available_models = [
    "mistralai/Mistral-7B-Instruct-v0.3"
]

class Mistral():
    def __init__(self, model_name):
        self.model_name = model_name

        mistral_models_path = Path.home().joinpath('mistral_models', model_name.split("/")[1])
        if not mistral_models_path.exists():
            self.download_model(mistral_models_path, model_name)

        self.tokenizer = MistralTokenizer.from_file(f"{mistral_models_path}/tokenizer.model.v3")
        self.model = Transformer.from_folder(mistral_models_path)

    def prompt(self, user_prompt):
        completion_request = ChatCompletionRequest(messages=[UserMessage(content=user_prompt)])

        tokens = self.tokenizer.encode_chat_completion(completion_request).tokens

        out_tokens, _ = generate([tokens], self.model, max_tokens=64, temperature=0.0, eos_id=self.tokenizer.instruct_tokenizer.tokenizer.eos_id)
        result = self.tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])

        return result

    def download_model(self, mistral_models_path: Path, model_name: str):
        mistral_models_path.mkdir(parents=True, exist_ok=True)

        snapshot_download(repo_id=model_name, allow_patterns=["params.json", "consolidated.safetensors", "tokenizer.model.v3"], local_dir=mistral_models_path)