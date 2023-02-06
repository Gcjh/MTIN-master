# MTIN
This project involves the code and supplementary materials of paper "Improving Stance Detection with Multi-task Interaction Network".

# Citation
If you are use this code for you research, please cite our paper.

 @inproceedings{CHAI-MTIN,
                title = {Improving Multi-task Stance Detection with Multi-task Interaction Network},
                author = {Chai, Heyan  and
                    Tang, Siyu  and
                    Cui, Jinhao  and
                    Ding, Ye  and
                    Fang, Binxing  and
                    Liao, Qing},
                booktitle = {Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, EMNLP 2022},
                year = {2022},
                address = {Abu Dhabi, United Arab Emirates},
                publisher = {Association for Computational Linguistics},
                url = {https://aclanthology.org/2022.emnlp-main.193},
                pages = {2990--3000},
}

# Dependencies
* pytorch == 1.9.0
* numpy == 1.20.1
* scikit-learn == 0.24.1
* transformer == 4.12.3
* scipy == 1.6.2
* spacy == 3.2.0
* digitalepidemiologylab/covid-twitter-bert-v2-mnli


# Run
Running MTIN is followed as:

    python -W ignore::RuntimeWarning run.py

If you want to switch datasets, modify the index of "target_list" in run.py
