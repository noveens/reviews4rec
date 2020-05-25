# Reviews4Rec

This repository contains many popular recommender algorithms which use reviews as additional data. The code accompanies the paper ***"How Useful are Reviews for Recommendation? A Critical Review and Potential Improvements"*** [[ACM]](https://doi.org/10.1145/3397271.3401281) [[Public PDF]](https://cseweb.ucsd.edu/~jmcauley/pdfs/sigir20.pdf) where we critique different existing review-based recommendation algorithms and questions their reported performance.

If you find any module of this repository helpful for your own research, please consider citing the below SIGIR'20 paper. Thanks!
```
@inproceedings{SachdevaMcAuley20,
  author = {Noveen Sachdeva and Julian McAuley},
  title = {How Useful are Reviews for Recommendation? A Critical Review and Potential Improvements},
  booktitle = {ACM Conference on Research and Development in Information Retrieval (SIGIR)},
  year = {2020}
}
```

**Code Author**: Noveen Sachdeva (ernoveen@gmail.com)

---
### Environents
- Python3 
    - Pytorch >= 0.4.0
    - Surprise: https://github.com/NicolasHug/Surprise
    - Gensim: https://github.com/RaRe-Technologies/gensim
    - h5py: https://github.com/h5py/h5py
    
- Python2.7 (only for MPCN)
    - Tensorflow-gpu >= 1.17.0
    - Keras
    - sklearn
    - scipy

---
### Setup
##### Data Setup
Once you've correctly setup the python environments and downloaded the dataset of your choice (Amazon: http://jmcauley.ucsd.edu/data/amazon/), the following steps need to be run:

```bash
$ ./prep_all_data.sh <HUMAN_FRIENDLY_DATASET_NAME> path/to/data/file.json
```

The above command will create the train/test/val splits along with some pre-processing scripts for running review-based methods like DeepCoNN/NARRE/TransNet much faster.

##### Setup for MPCN (Skip if not needed)
Since running MPCN requires a Python2.7 environment, you will need to modify Line#2 in the script `run_MPCN_in_p2.sh` to edit how to switch to the Python2.7 environment.

---
### Run Instructions
- Edit the `hyper_params.py` file which lists all config parameters, including what type of model to run. Currently supported models:

| Model Type | Model Name | Paper Link |
| --- | ------ | ------ |
| Non-textual | bias_only **(or)** baseline |  |
| Non-textual | MF_dot **(or)** NMF |  |
| Non-textual | SVD **(or)** SVD++ |  |
| Non-textual | MF |  |
| Non-textual | NeuMF | [LINK](http://papers.www2017.com.au.s3-website-ap-southeast-2.amazonaws.com/proceedings/p173.pdf) |
| Reviews as regularizer | HFT | [LINK](https://cs.stanford.edu/people/jure/pubs/reviews-recsys13.pdf) |
| Reviews as features | deepconn **(or)** deepconn++ | [LINK](https://arxiv.org/abs/1701.04783) |
| Reviews as features | NARRE | [LINK](http://www.thuir.cn/group/~YQLiu/publications/WWW2018_CC.pdf) |
| Reviews as features | transnet **(or)** transnet++ | [LINK](https://arxiv.org/abs/1704.02298) |
| Reviews as features | MPCN | [LINK](https://arxiv.org/abs/1801.09251) |

- Finally, type the following command to run:
```
$ CUDA_VISIBLE_DEVICES=<SOME_GPU_ID> python main.py
```
---
### Contribution
As more and more algorithms using reviews for recommendation are published, please feel free to send a pull request with your algorithm and I'll be happy to merge it into this repository.


### License
----

MIT

