# GRACE-pytorch
This repo is an unofficial implementation of [Graph Clustering with Embedding Propagation](https://cs.emory.edu/~jyang71/files/grace.pdf) in Python 3 and PyTorch. See [original code](https://github.com/yangji9181/GRACE) in Python 2 and TensorFlow. Note that I do not guarantee the correctness of this code.


## TO-DO

- [x] add GRACE model
- [x] fix bugs: the gradient of `self.mean` is not being properly updated
- [ ] fix other bugs...

## Only test on...

- Python 3.11
- SciPy 1.11.3
- Numpy 1.26.0
- tqdm 4.66.1
- scikit-learn 1.3.1
- PyTorch 2.1.0

**Note:** I don't think this code is version-sensitive, so maybe you don't need to be fully compliant with these.


## Train GRACE

Try this script to train the model on `cora` dataset.
```shell
python train.py --device cuda --dataset cora --embed_dim 512 --encoder_hidden 512 --decoder_hidden 512 --learning_rate 5e-5 --pre_epoch 1000 --epoch 2000
```

## Acknowledgments

This work partly uses the code from the [original version](https://github.com/yangji9181/GRACE).

## Cite

If you find this work useful, please cite our paper.
```
@INPROCEEDINGS{9378031,
  author={Yang, Carl and Liu, Liyuan and Liu, Mengxiong and Wang, Zongyi and Zhang, Chao and Han, Jiawei},
  booktitle={2020 IEEE International Conference on Big Data (Big Data)}, 
  title={Graph Clustering with Embedding Propagation}, 
  year={2020},
  volume={},
  number={},
  pages={858-867},
  doi={10.1109/BigData50022.2020.9378031}}
```
