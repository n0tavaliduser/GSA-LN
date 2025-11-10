import torch

def get_device(config):
    """
    Sets up the device for training or evaluation based on the configuration.
    """
    num_gpu = config.get('num_gpu', 'auto')
    if num_gpu == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            print(f"Using {torch.cuda.device_count()} available GPU(s).")
        else:
            print("CUDA not available, using CPU.")
    else:
        num_gpu = int(num_gpu)
        if num_gpu > 0 and torch.cuda.is_available():
            available_gpus = torch.cuda.device_count()
            if num_gpu > available_gpus:
                print(f"Warning: Requested {num_gpu} GPUs, but only {available_gpus} are available. Using {available_gpus} GPU(s).")
                device = torch.device('cuda')
            else:
                device = torch.device('cuda')
                print(f"Using {num_gpu} GPU(s).")
        else:
            device = torch.device('cpu')
            print("CUDA not available or num_gpu set to 0, using CPU.")
    return device