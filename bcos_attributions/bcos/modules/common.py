from torch import nn


class DetachableModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.detach = False

    def set_explanation_mode(self, activate: bool = True) -> None:
        """
        Turn explanation mode on or off.

        Parameters
        ----------
        activate : bool
            Turn it on.
        """
        self.detach = activate

    @property
    def is_in_explanation_mode(self) -> bool:
        """
        Whether the module is in explanation mode or not.
        """
        return self.detach
