# meta-transfer-learning
This repository is the official implementation of the L@S 2022 Paper entitled "Meta Transfer Learning for Early Success Prediction in MOOCs" written by [Vinitra Swamy](http://github.com/vinitra), [Mirko Marras](https://www.mirkomarras.com/), and [Tanja Käser](https://people.epfl.ch/tanja.kaeser/?lang=en).

Experiments are located in `scripts/`, corresponding directly to the experiment codes mentioned in the paper. At the beginning of each script, the research question this experiment targets and a short description of the model the file generates is included. For more information about each experiment, please reference the paper directly. The best behavior-only (BO), behavior-time-meta (BTM), and behavior-static-meta (BSM) models that can be used to warm-start downstream predictions can be found in `models/`.

## Usage guide

0. Install relevant dependencies with `pip install -r requirements.txt`.

1. Extract relevant features sets (`BouroujeniEtAl`, `MarrasEtAl`, `LalleConati`, and `ChenCui`) through our lab's EDM 2021 contribution on [benchmarks for feature predictive power](https://github.com/epfl-ml4ed/flipped-classroom). Place the results of these feature extraction scripts in `data/` with the same folder structure as shown through the toy-course.

2. Fill out the `data/metadata.csv` file with relevant meta information about your course(s). One example of course metadata is currently showcased in the file. This information is necessary for extracting the meta features in the BTM and BSM models.

3. Run your desired experiment from `scripts/` by executing the script with Python 3.7 or higher.

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
