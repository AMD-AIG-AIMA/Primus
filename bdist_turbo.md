build primus_turbo bdist_wheel
```bash
git clone -b zhenhuang12/fix_deep_ep_dispatch git@github.com:AMD-AIG-AIMA/Primus-Turbo.git
git submodule update --init --recursive
cd primus_turbo || exit 1
pip install -r requirements.txt
python setup.py bdist_wheel
```
build aiter bdist_wheel
```bash
git clone https://github.com/ROCm/aiter.git
cd aiter
python setup.py bdist_wheel
```
