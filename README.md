# An information-theoretic study of lying in LLMs

This is the code base for our paper "An information-theoretic study of lying in LLMs" published at the 2024 ICML Workshop on Large Language Models and Cognition.  

### Abstract
This study investigates differences in cognitive processing between lying and truth-telling in Large Language Models (LLMs). Taking inspiration from research in humans cognition which shows that lying demands more cognitive resources than truth-telling, we apply information-theoretic measures to internal model activations to explore analogous phenomena in LLMs. Our analysis reveals that LLMs converge more quickly to the output distribution when telling the truth and exhibit higher entropy when constructing lies. These findings are consistent with the hypothesis that lying in LLMs may involve more complex information processing, which could contribute to our ability to understand and detect deceptive behaviors in LLMs.

### Install
Tested on Python 3.10.12

```
git clone https://github.com/annahdo/info_theoretic_study_lying.git
cd info_theoretic_study_lying
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```
## Method
We briefly summarize our experimental process below.

**Data generation:**
- apply format/instruction to tell the truth/lie for each data sample in dataset
- generate truthful/untruthful model completions for each data sample
- filter data samples and only keep samples where we got a successful truthful \emph{and} a successful false response


The following example is meant to illustrate the setup:

<div style="background-color: #D5D5D5; padding: 10px; border-radius: 5px; color: black;">

$$
\underbrace{\text{Be truthful! }}_{\text{truth condition}} \underbrace{\text{France's capital is }}_{\text{input}} \underbrace{\text{Paris, a city ...}}_{\text{model output}}
$$
$$
\underbrace{\text{Please lie! }}_{\text{lie condition}} \underbrace{\text{France's capital is }}_{\text{input}} \underbrace{\text{Rome, which ...}}_{\text{model output}}
$$
</div>
&nbsp;

**Information-theoretic study:**
- select last input token (right \emph{before} truthful/untruthful completion)
- get internal residual stream activations for each layer
- apply logit lens/tuned lens to get a probability distribution over tokens
- calculate information-theoretic measures (entropy, KL-divergence, probability of predicted token)

## Results
We show results for the Statements1000 dataset and zephyr-7b-beta.

![The entropy is higher when the model constructs a lie](plots/zephyr-7b-beta_Statements1000_entropy_logit_lens_.png)

 

### Citation

If you found this code useful, please consider citing our paper.

```
@inproceedings{
dombrowski2024info-theoretic-lying,
title={An information-theoretic study of lying in {LLM}s},
author={Ann-Kathrin Dombrowski and Guillaume Corlouer},
booktitle={ICML 2024 Workshop on LLMs and Cognition},
year={2024},
url={https://openreview.net/forum?id=9AM5i1wWZZ}
}
```
