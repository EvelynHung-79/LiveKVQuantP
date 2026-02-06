python3.11 -m venv finchenv
source finchenv/bin/activate
pip install --upgrade pip
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

### 只需一次
export PYTHONPATH=$PYTHONPATH:$(pwd)/src