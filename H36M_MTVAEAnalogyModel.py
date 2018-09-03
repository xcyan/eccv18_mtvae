"""Analogy-making using MT-VAE on Human3.6M dataset."""

from nets import h36m_mtvae_analogy_factory as model_factory
from H36M_BaseAnalogyModel import BaseAnalogyModel

class MTVAEAnalogyModel(BaseAnalogyModel):
  """Defines the MT-VAE analogy class."""
  def __init__(self, params):
    super(MTVAEAnalogyModel, self).__init__(params)

  def get_model_fn(self, is_training, reuse):
    return model_factory.get_model_fn(self._params, is_training, reuse)

