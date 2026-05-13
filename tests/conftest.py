"""pytest configuration: mock heavy optional dependencies (torch, lightning, etc.)
so that pure-Python tests can run in environments where these are not installed.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock


def _make_module(name: str, **attrs) -> types.ModuleType:
    """Create a fake module and register all its dot-separated parents."""
    parts = name.split(".")
    # Register parents first
    for i in range(1, len(parts)):
        parent_name = ".".join(parts[:i])
        if parent_name not in sys.modules:
            sys.modules[parent_name] = types.ModuleType(parent_name)

    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    # Attach as attribute on parent
    if len(parts) > 1:
        parent = sys.modules[".".join(parts[:-1])]
        setattr(parent, parts[-1], mod)
    return mod


# ---------------------------------------------------------------------------
# torch stubs
# ---------------------------------------------------------------------------
_torch = _make_module("torch")
_torch.Tensor = MagicMock
_torch.tensor = MagicMock()
_torch.cat = MagicMock()
_torch.cuda = MagicMock()
_torch.cuda.is_available = lambda: False
_torch.norm = MagicMock()
_torch.profiler = MagicMock()

_torch_nn = _make_module("torch.nn")
_torch_nn.functional = MagicMock()

_torch_nn_functional = _make_module("torch.nn.functional")
_torch_nn_functional.cosine_similarity = MagicMock(return_value=MagicMock())

_make_module("torch.nn.modules")
_make_module("torch_geometric")
_tg_data = _make_module("torch_geometric.data")
_tg_data.Batch = MagicMock()

# ---------------------------------------------------------------------------
# Lightning stubs
# ---------------------------------------------------------------------------
class _Callback:
    """Minimal Callback base used as a stand-in for lightning.pytorch.callbacks.Callback."""

    def set_logger(self, logger):
        pass


_lightning_cb_mod = _make_module("lightning.pytorch.callbacks")
_lightning_cb_mod.Callback = _Callback
_make_module("lightning.pytorch")
_make_module("lightning")

# ---------------------------------------------------------------------------
# AGeDi stubs
# ---------------------------------------------------------------------------
_agedi = _make_module("agedi")
_agedi.create_diffusion = MagicMock()
_agedi.create_dataset = MagicMock()
_agedi.create_trainer = MagicMock()
_agedi.train = MagicMock()
_agedi.sample = MagicMock(return_value=[])
