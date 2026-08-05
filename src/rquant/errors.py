class RQuantError(Exception):
    """Base exception for actionable user-facing failures."""


class ConfigurationError(RQuantError):
    pass


class DependencyError(RQuantError):
    pass


class DataContractError(RQuantError):
    pass


class FactorContractError(RQuantError):
    pass
