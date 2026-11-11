"""Explicit, fail-closed, generator-neutral interfaces (not native parsers)."""
from .chains import ExplicitChainsAdapter, GidNETChainsAdapter, CaQRChainsAdapter, DQCCChainsAdapter
__all__ = ['ExplicitChainsAdapter','GidNETChainsAdapter','CaQRChainsAdapter','DQCCChainsAdapter']
