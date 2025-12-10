from openpi.training import config
from openpi.policies import policy_config
from openpi.shared import download

model_name = "pi05_droid"
model_link = "gs://openpi-assets/checkpoints/pi05_droid"

config = config.get_config(model_name)
checkpoint_dir = download.maybe_download(model_link)