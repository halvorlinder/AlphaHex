## Get access to the vm
Go to the connect tab and follow the instructions in azure
## Get the repo
On the vm, create a key with 
```
ssh-keygen -t rsa -b 4096
```
Add the key on github 
```
git clone git@github.com:halvorlinder/AlphaHex.git
```

## Install python and pip 
```
sudo apt update && sudo apt upgrade -y
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.10
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
```
## Add python to path
```
vim ~/.bashrc
```
On the bottom of the file, add:
```
export PATH="/home/halvor/.local/bin:$PATH"
```
Then
```
source ~/.bashrc
```

## Tmux
Tmux is required to keep the process running when connection to the server is lost
```
sudo apt install tmux
tmux 
```

To keep the tmux session running in the background do 
```
ctrl-b d
```

## Install anaconda
```
cd
wget https://repo.anaconda.com/archive/Anaconda3-2023.03-Linux-x86_64.sh
conda create --name AlphaHex python=3.10
conda activate AlphaHex
```

## Install packages
There might be more
```
conda install pip 
conda install pytorch torchvision torchaudio cpuonly -c pytorch
pip install matplotlib
pip install networkx
pip install wandb
```

## Running 
There is no gui on the server, so just insert return on the first line of the plot function in hex.py
```
cd ~/AlphaHex
python RL.py
```

## Editing
Use Vim
