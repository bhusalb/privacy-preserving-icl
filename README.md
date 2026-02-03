# Privacy Preserving In-Context-Learning Framework for Large Language Models
This is a codebase to perform privacy-preserving in-context learning. 

Read our AAAI 2026 paper:
Privacy Preserving In-Context-Learning Framework for Large Language Models https://arxiv.org/abs/2509.13625

Authors: Bishnu Bhusal, Manoj Acharya, Ramneet Kaur, Colin Samplawski, Anirban Roy, Adam D. Cobb, Rohit Chadha, Susmit Jha


## Experiments
For AGNEWS, see `run_agnews.sh`
For DBPedia, see `run_dbpedia.sh`
For TREC, see `run_trec.sh`
For MIT-G, see `run_movieg.sh`
For MIT-D, see `run_movied.sh`

Note: Due to the randomness in generations caused by DP noise, the results may be slightly different from the reported values in the paper.


## Setup
1. Install Python 3.10.

   One way is to use [pyenv](https://github.com/pyenv/pyenv).
   Run `pyenv install --list | grep '^ *3.10' | tail -n1` to discover the most recent minor version of Python 3.10.
   Run `pyenv install 3.10.X` where `X` is the latest minor version available.

1. Install [Poetry](https://python-poetry.org/) following the [instructions](https://python-poetry.org/docs/#installation).
1. Configure Poetry to use your Python 3.10 installation.
    - If using `pyenv` setup above: run `poetry env use $(pyenv prefix 3.10.X)/bin/python`
    - Otherwise: run `poetry env use <path to your Python 3.10 binary>`
1. Run `poetry install` to install the dependencies.


## Project structure
- `src/`: Python code for the project.
- `data/`: Python code for data processing of MIT dataset.
- `privacy_analysis/`: Python code for calculating the noise parameter.

## Acknowledgments
This project is built upon the foundation of [Privacy-Preserving In-Context Learning with Differentially Private Few-Shot Generation](https://github.com/microsoft/dp-few-shot-generation).
We would like to thank the contributors and maintainers of the original repository for their valuable work.

## Cite our work

```
@article{bhusal2025privacy,
  title={Privacy Preserving In-Context-Learning Framework for Large Language Models},
  author={Bhusal, Bishnu and Acharya, Manoj and Kaur, Ramneet and Samplawski, Colin and Roy, Anirban and Cobb, Adam D and Chadha, Rohit and Jha, Susmit},
  journal={AAAI},
  year={2026}
}
```


