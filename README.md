# MLP_CW3

Install codebase:
- pip install -e .

Fetch and install environment:

1. git submodule init
2. git submodule update
3. cd environments/PettingZoo
4. pip insatll -e .

Run env loading (future training script):

- python MLP_CW3/scripts/train.py +seed=1 +env=simple_tag_v1 +alg=a2c_gnn

### setuptools and wheel
- `pip install wheel==0.38.4`
- `pip install setuptools==65.5.0`
