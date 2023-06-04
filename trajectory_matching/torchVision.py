import torch
import subprocess

def main():
    print(torch.__version__)
    print(torch.version.cuda)
    if torch.cuda.is_available():
        if torch.__version__ == '1.8.0':
            subprocess.call(['pip', 'install', 'torchvision==0.9.0'])
        elif torch.__version__ == '1.9.0':
            subprocess.call(['pip', 'install', 'torchvision==0.10.0'])
        else:
            print('torchvision version not found for this torch version and CUDA combination.')

if __name__ == "__main__":
    main()