import platform 
import torch

def get_env_info(device):
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda, 
        "cpu": platform.processor() or platform.machine(),
    }
    if str(device).startswith("cuda"):
        info["gpu"] = torch.cuda.get_device_name(0)
    return info