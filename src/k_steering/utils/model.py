


def get_transformer_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers          # LLaMA / Mistral / Qwen
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h         # GPT-2 / GPT-Neo
    raise ValueError("Unsupported model architecture")