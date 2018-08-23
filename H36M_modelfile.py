"""Model specs."""

VAE = {
  'model_name': 'H36M_VAE',
  'enc_rnn_size': 1024,
  'dec_rnn_size': 1024,
  'embed_dim': 1024,
  'content_dim': 512,
  'noise_dim': 512,
  'enc_fc_layers': 0,
  'latent_fc_layers': 1,
  'dec_fc_layers': 1,
  'use_latent': 1,
  'dec_style': 1,
  'T_layer_norm': 1,
}

MTVAE = {
  'model_name': 'H36M_MTVAE',
  'dec_interaction': 'add',
  'enc_rnn_size': 1024,
  'dec_rnn_size': 1024,
  'embed_dim': 1024,
  'content_dim': 512,
  'noise_dim': 512,
  'enc_fc_layers': 0,
  'latent_fc_layers': 1,
  'dec_fc_layers': 1,
  'use_latent': 1,
  'dec_style': 1,
  'T_layer_norm': 1,
}
