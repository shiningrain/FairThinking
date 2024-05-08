# FairThinking


## TL;DR

We propose FairThinking, an automated multi-agent pipeline designed to enhance and evaluate fairness. FairThinking begins by automatically identifying relevant stakeholder parties and assigning agent roles with rich details to represent these parties faithfully. Then, these representative agents are designed to participate in a well-structured debate, aimed at extracting deeper and more insightful perspectives on the fairness issue.
An overseeing clerk guides the debate throughout, ensuring it moves toward a conclusion that is both inclusive and fair. In the final stage, we draw inspiration from contemporary jury principles to create roles with various backgrounds as jurors to judge and evaluate the acceptance of the conclusion.
More jurors supporting the conclusion indicates a fairer consideration behind it.



## Repo Structure

```         
- FairThinking_code/   
    - demo
    - utils
    - demo.py
- dataset/
    - BiasAsker
    - Comparative_Questions
    - General_Open-Ended_Questions
    - Targeted_Open-Ended_Questions
- README.md
- requirements.txt    
```

## Setup

FairThinking is implemented on Python 3.10.
To install all dependencies, please get into this directory and run the following command.
```
pip install -r requirements.txt
```

To conduct experiments on GPT-3.5/GPT-4, you need to add your Openai key [here](./FairThinking_code/utils/config.cfg).

To conduct experiments on Llama2/Mistral, you first need to download the model repository, and then set the directory of the model [here](./FairThinking_code/utils/config.cfg).
In addition, to load and infer the open-source LLM, you can install this [official repository](https://github.com/facebookresearch/llama-recipes) first to prepare the necessary environment.


## Usage

FairThinking is very easy to use.
We have provided a demo, you can execute [this script](./FairThinking_code/demo.py) in the installed Python environment.
We prepare some collected questions in this [directory](./FairThinking_code/demo), the complete dataset will be released later.


This script has 8 parameters:
1. `topic_path` assigns the question you want FairThinking to answer. It should be a `question.pkl` or `topic.csv` in the `demo` repository.
2. `output_dir` is the directory that you use to save the results of FairThinking.
3. `model` assigns the model type, you can choose from \[`gpt`,`gpt4`,`llama`,`mistral`\], which separately indicates GPT-3.5-Turbo, GPT-4-Turbo, Llama 2, Mistral.
4. `jury` is the jury number you want to set, which is defaulted to be `6`.
5. `config_path`, `attribute`, `gpu_list`, and `repeat` do not require to config.


## Dataset
We provide the whole dataset in this [directory](./dataset).
The sub-directories separately show the `BiasAsker`, `Comparative Questions`, `Targeted Open-Ended Questions`, and `General Open-Ended Questions` datasets.

1. `question.pkl` directly stores questions as a list (open-ended questions are stored in a `.csv` file).
2. `xx-roles` refers to the role information corresponding to the question of index `xx` in the list.
    1. The `attribute_list.pkl` stores the role information of the debaters.
    2. The `jury_list.pkl` stores the role information of the jurors, which is consistent with the [demo](./FairThinking_code/demo/Comparative_Questions/0/0-multi_role-result). 
