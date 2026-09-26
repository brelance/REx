from inspect_ai.model import ModelAPI, modelapi


@modelapi(name="opencode")
def opencode() -> type[ModelAPI]:
    from inspect_evals.model_providers.opencode import OpenCodeAPI

    return OpenCodeAPI
