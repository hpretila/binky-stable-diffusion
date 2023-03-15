import json

from model.diffusion import DiffusionModel
from model.config.diffusion import DiffusionConfig

from util.hf_model_helper import HFDiffuserModelHelper as hf

# Load JSON and deserialise into DiffusionConfig
config_json = open("config/config.json", "r").read()
config_dict = json.loads(config_json)
diffusion_config = DiffusionConfig.from_dict(config_dict)

# Create DiffusionModel
diffusion_model : DiffusionModel = DiffusionModel(diffusion_config)
diffusion_model.configure_sharded_model()
diffusion_model = diffusion_model.to("cuda")

# Inference test
def inference_test():
    global diffusion_model

    import matplotlib.pyplot as plt

    # Do inference
    res = diffusion_model.forward(["Hello world!"])

    img = hf.decode_latents(res)
    plt.imshow(img)

inference_test()