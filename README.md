# WILTing Trees: Interpreting the Distance Between MPNN Embeddings

### Setup Environments
```
$ pwd 
xxx/wilt
$ pyenv global 3.10.11
$ python -m venv wilt_env
$ source wilt_env/bin/activate
$ pip install --upgrade pip
$ pip list 
Package    Version
---------- -------
pip        24.2
setuptools 65.5.0
$ pip install -r requirements.txt
$ pip install torch_scatter -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
$ pip install torch_sparse -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
$ pip install torch_cluster -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
```
Note: The above procedure assumes you have cuda 11.8. You can instead install cpu versions of torch, torchvision, torch_scatter, torch_sparse, and torch_cluster. Our code will still work.

After running the above commands, please replace `fs1.mv(path1, path2, recursive)` in line 192 of torch_geometric/io/fs.py with `fs1.mv(path1, path2)`.

Then, please run `pytest test` to make sure all setups are successful.

For VSCode users, you can open this repo with wilt.code-workspace after installing the following extensions:
- ms-python.black-formatter
- ms-python.flake8
- ms-python.isort
- ms-python.mypy-type-checker