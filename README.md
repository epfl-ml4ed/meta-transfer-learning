# meta-transfer-learning
This repository is the official implementation of the L@S 2022 Paper entitled "Meta Transfer Learning for Early Success Prediction in MOOCs" written by [Vinitra Swamy](http://github.com/vinitra), [Mirko Marras](https://www.mirkomarras.com/), and [Tanja Käser](https://people.epfl.ch/tanja.kaeser/?lang=en).

Experiments are located in `scripts/`, corresponding directly to the experimental methodology mentioned in the paper. At the beginning of each script, you will find the research question this experiment targets and a short objective statement regarding the model the script generates. For more information about each experiment, please reference the paper directly. The best behavior-only (BO), behavior-time-meta (BTM), and behavior-static-meta (BSM) models for each early prediction setting (40% and 60%) can be found in `models/`. These models can be used to warm-start downstream student performance predictions for new or ongoing courses.

## Usage guide

0. Install relevant dependencies with `pip install -r requirements.txt`.

1. Extract relevant features sets (`BouroujeniEtAl`, `MarrasEtAl`, `LalleConati`, and `ChenCui`) through our lab's EDM 2021 contribution on [benchmarks for feature predictive power](https://github.com/epfl-ml4ed/flipped-classroom). Place the results of these feature extraction scripts in `data/`. A toy course example of extracted features is included in the `data/` folder.

2. Fill out the `data/metadata.csv` file with relevant meta information about your course(s). One example of course metadata is currently showcased in the file. This information is necessary for extracting the meta features in the BTM and BSM models.

3. Run your desired experiment from `scripts/` by executing the script with Python 3.7 or higher.

## Scripts
We extensively evaluate our models on a large data set including 26 MOOCs and 145,714 students in total. With our analyses, we target the following three research questions, addressed through scripts in this repository:

- **RQ 1: Can student behavior transfer across iterations of the same course and across different courses?**
  - `BO_1to1.py`
  - `BO_Nto1_Diff.py`
  - `BO_Nto1_Same.py`

- **RQ 2: Is a meta learning model trained on a combination of behavior and course metadata information more transferable?**
  - `BSM_Nto1_Diff.py`
  - `BTM_Nto1_Diff.py`
  - **Encoding experiments**: encoding the course title and description text using different encoders (FastText, SBert, USE).  
    - `BTM_Nto1_Diff_FastText.py`
    - `BTM_Nto1_Diff_SentenceBERT.py`
    - `BTM_Nto1_Diff_UniversalSentEncoder.py`

- **RQ 3: Can fine-tuning a combined model on past iterations of an unseen course lead to better transferable models?**
  - `BO_Nto1_Diff_Finetune.py`
  - `BO_NtoC_Diff_Finetune.py`
  - `BSM_NtoC_Diff_Finetune.py`
  - `BTM_NtoC_Diff_Finetune.py`

- **Helper Utilities**
  - `predict_on_all_students.py`: Evaluate a trained model on all students, not just the subset of hard students.
  - `rnn_models.py`: Bi-LSTM model architectures used in BO experiments.

## Models

## Contributing 

This code is provided for educational purposes and aims to facilitate reproduction of our results, and further research 
in this direction. We have done our best to document, refactor, and test the code before publication.

If you find any bugs or would like to contribute new models, training protocols, etc, please let us know. Feel free to file issues and pull requests on the repo and we will address them as we can.

## Citations
If you find this code useful in your work, please cite our paper:

```
Swamy, V., Marras, M., Käser, T. (2022). 
Meta Transfer Learning for Early Success Prediction in MOOCs. 
In: Proceedings of the 2022 ACM Conference on Learning at Scale (L@S 2022). 
```

## License
This code is free software: you can redistribute it and/or modify it under the terms of the [MIT License](LICENSE).

This software is distributed in the hope that it will be useful, but without any warranty; without even the implied warranty of merchantability or fitness for a particular purpose. See the [MIT License](LICENSE) for details.
