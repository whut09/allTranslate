"""Kernel package — hot-pluggable translation kernel registry."""

from allTranslate.kernel.registry import KernelRegistry
from allTranslate.kernel.legacy import LegacyKernel
from allTranslate.kernel.precise import PreciseKernel

# Always register both kernels.
# PreciseKernel.is_available() returns False if submodule/venv not set up.
KernelRegistry.register(LegacyKernel())
KernelRegistry.register(PreciseKernel())

__all__ = ["KernelRegistry"]
