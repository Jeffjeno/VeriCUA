from agent_runtime.harnesses.computer_use.harness import ComputerUseHarness

class OpenAIComputerUse(ComputerUseHarness):

    def __init__(self, **kwargs):
        kwargs.setdefault("provider", "openai")
        super().__init__(**kwargs)

__all__ = ["ComputerUseHarness", "OpenAIComputerUse"]
