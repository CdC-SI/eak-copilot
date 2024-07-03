from components.llm.base import LLM
from components.llm.implementations import OpenAILLM, MlxLLM, LlamaCppLLM, HuggingFaceLLM
from components.config import SUPPORTED_OPENAI_LLM_MODELS, SUPPORTED_MLX_LLM_MODELS, SUPPORTED_LLAMACPP_LLM_MODELS, SUPPORTED_HUGGINGFACE_LLM_MODELS

class LLMFactory:
    @staticmethod
    def get_llm_client(llm_model: str) -> LLM:
        """
        Factory method to instantiate llm clients based on a string identifier.

        Parameters
        ----------
        llm_model : str
            The name of the LLM model. Currently supported models are "gpt-3.5-turbo-0125", "gpt-4-turbo-preview", "gpt-4o", "mlx-community/Nous-Hermes-2-Mistral-7B-DPO-4bit-MLX" and "Qwen/Qwen1.5-0.5B-Chat-GGUF".

        Returns
        -------
        LLM
            An instance of the appropriate llm client.

        Raises
        ------
        ValueError
            If the `llm_model` is not supported.
        """
        if llm_model in SUPPORTED_OPENAI_LLM_MODELS:
            return OpenAILLM(model_name=llm_model)
        elif llm_model in SUPPORTED_MLX_LLM_MODELS:
            return MlxLLM(model_name=llm_model)
        elif llm_model in SUPPORTED_LLAMACPP_LLM_MODELS:
            return LlamaCppLLM(model_name=llm_model)
        elif llm_model in SUPPORTED_HUGGINGFACE_LLM_MODELS:
            return HuggingFaceLLM(model_name=llm_model)
        else:
            raise ValueError(f"Unsupported embedding model type: {llm_model}")
