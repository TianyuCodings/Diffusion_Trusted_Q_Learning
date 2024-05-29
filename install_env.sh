conda create -n rl python=3.9
conda activate rl

# make sure the cuda verion fits the local machine
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
pip install mujoco
pip install git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl
pip install "cython<3"
pip install python-dateutil


wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz

# do this in the home directory, it will extract mujoco210 into /home/username/.mujoco/mujoco210...
mkdir -p .mujoco/
tar -xvf mujoco210-linux-x86_64.tar.gz -C .mujoco/

# add this to .bashrc
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/tychen/.mujoco/mujoco210/bin
source .bashrc