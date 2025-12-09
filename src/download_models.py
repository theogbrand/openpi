from openpi.training import config as _config
from openpi.shared import download
from openpi.policies import policy_config as _policy_config

model_name = "pi05_droid"
model_link = "gs://openpi-assets/checkpoints/pi05_droid"

config = _config.get_config(model_name)
checkpoint_dir = download.maybe_download(model_link)

policy = _policy_config.create_trained_policy(config, checkpoint_dir)