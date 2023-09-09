# BERT-Gen model for CommonGEN


## Installation

```bash
conda create -n bert_base python=3.7
conda activate bert_base
pip install torch==1.4.0
git clone https://github.com/NVIDIA/apex.git && cd apex && python setup.py install --cuda_ext --cpp_ext
cd ..
```

The following Python package need to be installed:
```bash
pip install --user methodtools py-rouge pyrouge nltk
python -c "import nltk; nltk.download('punkt')"
```

Install the repo as a package:
```bash
cd unilm/s2s-ft ; pip install --editable .
```