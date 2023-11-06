
[Troubleshooting guide](https://pytorch.org/rl/reference/generated/knowledge_base/MUJOCO_INSTALLATION.html)

```bash
MUJOCO_DIR=~/.mujoco
mkdir -p $MUJOCO_DIR

# 200
wget https://roboti.us/download/mujoco200_linux.zip -O $MUJOCO_DIR/mujoco200.zip
unzip $MUJOCO_DIR/mujoco200.zip
wget https://roboti.us/file/mjkey.txt -P $MUJOCO_DIR
rm $MUJOCO_DIR/mujoco200.zip

# 210
wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz -O $MUJOCO_DIR/mujoco210.tar.gz
tar -xzf $MUJOCO_DIR/mujoco.tar.gz -C $MUJOCO_DIR
rm $MUJOCO_DIR/mujoco.tar.gz
```


```bash
mamba create -n diayn "python<3.9"
mamba activate diayn
mamba install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
mamba install ipython jupyter
pip install mujoco==3.0.0
```

```python
import mujoco
xml = """
<mujoco>
  <worldbody>
        <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="1 1 0.1" rgba=".9 0 0 1"/>
    <body pos="0 0 1">
      <joint type="free"/>
      <geom type="box" size=".1 .2 .3" rgba="0 .9 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
model
```

```bash
wget https://github.com/google-deepmind/mujoco/releases/download/3.0.0/mujoco-3.0.0-linux-x86_64.tar.gz
tar xvf mujoco-3.0.0-linux-x86_64.tar.gz 
mv mujoco-3.0.0 ~/.mujoco/mujoco300
rm mujoco-3.0.0-linux-x86_64.tar.gz 
conda env config vars set PATH=$PATH:/home/mimoralea/.mujoco/mujoco300/bin
conda deactivate && conda activate diayn
which simulate
```

```bash
pip install dm_control==1.0.15
pip install pip==22.0.4 setuptools==59.8.0 wheel==0.37.1
pip install gym==0.21
```

