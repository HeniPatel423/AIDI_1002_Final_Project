# A brief overview of swarm intelligence-based algorithms for numerical association rule mining

The world is highly competitive, and Artificial Intelligence (AI) is changing how we make decisions in various areas of life, from ads to self-driving cars. AI relies heavily on data, similar to how a car needs oil to run. The decisions made by AI are mostly based on past data from specific domains. This data comes in different forms, such as unstructured, semi-structured, or structured. The section mentions that AI has been evolving, and there are discussions about the potential superintelligence of machines.

Researchers in the field of Machine Learning (ML) explore different methods under its umbrella. One crucial area is Association Rule Mining (ARM), which finds relationships between attributes in structured databases. This process helps in market basket analysis, building intelligent systems, and creating rule-based classifiers. Despite a peak in interest around 2014, ARM remains popular among researchers, and new methods continue to emerge each year.

The context of the problem is in the wider world of AI, ML, and ARM. Decision-making increasingly relies on data stored in structured databases, where ARM plays a crucial role. The paper highlights the historical trend of interest in ARM, with a peak around 2014. It introduces NARM as an extension of ARM, specifically designed for datasets with both numerical and categorical attributes. The challenge with NARM is its complexity due to a larger search space, leading to the preference for specialized algorithms based on nature-inspired paradigms like SI. The context suggests a need for tailored algorithms to efficiently address the complexities of NARM.

### In this context, we employ the NiaARM (Nature-Inspired Association Rule Mining) framework for mining numerical association rules.

<p align="center">
  <img alt="logo" width="300" src="https://github.com/HeniPatel423/AIDI_1002_Final_Project/blob/main/datasets/logo.png">
</p>

---

### NiaARM - A minimalistic framework for Numerical Association Rule Mining


NiaARM is a framework for Association Rule Mining based on nature-inspired algorithms for optimization. The framework is written fully in Python and runs on all platforms. NiaARM allows users to preprocess the data in a transaction database automatically, to search for association rules and provide a pretty output of the rules found. This framework also supports integral and real-valued types of attributes besides the categorical ones. Mining the association rules is defined as an optimization problem, and solved using the nature-inspired algorithms that come from the related framework called [NiaPy](https://github.com/NiaOrg/NiaPy).

* **Documentation:** https://niaarm.readthedocs.io/en/latest/
* **Tested OS:** Windows, Ubuntu, Fedora, Alpine, Arch, macOS. **However, that does not mean it does not work on others**

## Detailed insights
The current version includes (but is not limited to) the following functions:

- loading datasets in CSV format,
- preprocessing of data,
- searching for association rules,
- providing output of mined association rules,
- generating statistics about mined association rules,
- visualization of association rules,
- association rule text mining (experimental).

## Installation

### pip

Install NiaARM with pip:

```sh
pip install niaarm
```

To install NiaARM on Alpine Linux, please enable Community repository and use:

```sh
$ apk add py3-niaarm
```

To install NiaARM on Arch Linux, please use an [AUR helper](https://wiki.archlinux.org/title/AUR_helpers):

```sh
$ yay -Syyu python-niaarm
```

To install NiaARM on Fedora, use:

```sh
$ dnf install python3-niaarm
```

To install NiaARM on NixOS, please use:

```sh
nix-env -iA nixos.python311Packages.niaarm
```

## About Dataset

### The new experiment is done on the 'heart.csv' data.

Dataset link: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset


#### This data set dates from 1988 and consists of four databases: Cleveland, Hungary, Switzerland, and Long Beach V. It contains 76 attributes, including the predicted attribute, but all published experiments refer to using a subset of 14 of them. The "target" field refers to the presence of heart disease in the patient. It is integer valued 0 = no disease and 1 = disease.

Attribute Information:

#### 1. age
#### 2. sex
#### 3. chest pain type (4 values)
#### 4. resting blood pressure
#### 5. serum cholestoral in mg/dl
#### 6. fasting blood sugar > 120 mg/dl
#### 7. resting electrocardiographic results (values 0,1,2)
#### 8. maximum heart rate achieved
#### 9. exercise induced angina
#### 10. oldpeak = ST depression induced by exercise relative to rest
#### 11. the slope of the peak exercise ST segment
#### 12. number of major vessels (0-3) colored by flourosopy
#### 13. thal: 0 = normal; 1 = fixed defect; 2 = reversable defect
The names and social security numbers of the patients were recently removed from the database, replaced with dummy values.

## Usage

### Loading data

In NiaARM, data loading is done via the `Dataset` class. There are two options for loading data:

#### Option 1: From a pandas DataFrame (recommended)

```python
import pandas as pd
from niaarm import Dataset


df = pd.read_csv('datasets/heart.csv')
# preprocess data...
data = Dataset(df)
print(data) # printing the dataset will generate a feature report
```

#### Option 2: From CSV file directly

```python
from niaarm import Dataset


data = Dataset('datasets/heart.csv')
print(data)
```

### Preprocessing

#### Data Squashing

Optionally, a preprocessing technique, called data squashing [5], can be applied. This will significantly reduce the number of transactions, while providing similar results to the original dataset.

```python
from niaarm import Dataset, squash

dataset = Dataset('datasets/heart.csv')
squashed = squash(dataset, threshold=0.9, similarity='euclidean')
print(squashed)
```

### Making Rules

```python
from niaarm import Dataset, Feature, Rule

#### load the Heart dataset
data = Dataset("datasets/heart.csv")

#### making the rule All Features => Target([0, 1]) for our heart data

antecedent = [
    Feature("age", dtype="int", min_val=29, max_val=77),
    Feature("sex", dtype="int", min_val=0, max_val=1),
    Feature("cp", dtype="int", min_val=0, max_val=3),
    Feature("trestbps", dtype="int", min_val=94, max_val=200),
    Feature("chol", dtype="int", min_val=126, max_val=564),
    Feature("fbs", dtype="int", min_val=0, max_val=1),
    Feature("restecg", dtype="int", min_val=0, max_val=2),
    Feature("thalach", dtype="int", min_val=71, max_val=202),
    Feature("exang", dtype="int", min_val=0, max_val=1),
    Feature("oldpeak", dtype="float", min_val=0.0, max_val=6.2),
    Feature("slope", dtype="int", min_val=0, max_val=2),
    Feature("thal", dtype="int", min_val=0, max_val=3)
]
consequent = [Feature("target", dtype="int", min_val=0, max_val=1)]

#### pass the transaction data to the Rule constructor to enable the calculation of metrics
rule = Rule(antecedent, consequent, transactions=data.transactions)

print(rule)
print(f"Support: {rule.support}")
print(f"Confidence: {rule.confidence}")
print(f"Lift: {rule.lift}")
```

### Mining association rules

#### The easy way (recommended)

Association rule mining can be easily performed using the `get_rules` function:

```python
from niaarm import Dataset, get_rules
from niapy.algorithms.basic import DifferentialEvolution

data = Dataset("datasets/heart.csv")

algo = DifferentialEvolution(population_size=50, differential_weight=0.5, crossover_probability=0.9)
metrics = ('support', 'confidence')

rules, run_time = get_rules(data, algo, metrics, max_iters=30, logging=True)

print(rules) # Prints basic stats about the mined rules
print(f'Run Time: {run_time}')
rules.to_csv('output.csv')
```

### Output of Mining Rules and Measures

The Numerical rule mining output consists of association rules indicating strong relationships in the dataset. All rules exhibit perfect fitness and confidence, suggesting a robust fit to the data. While the rules are easily understandable. https://github.com/HeniPatel423/AIDI_1002_Final_Project/blob/main/output.csv here is the output file.

### Visualization

The framework currently supports the hill slopes visualization method presented in [4]. More visualization methods are planned
to be implemented in future releases. 

```python
from matplotlib import pyplot as plt
from niaarm import Dataset, get_rules
from niaarm.visualize import hill_slopes

dataset = Dataset('datasets/heart.csv')
metrics = ('support', 'confidence')
rules, _ = get_rules(dataset, 'DifferentialEvolution', metrics, max_evals=1000, seed=1234)
some_rule = rules[150]
hill_slopes(some_rule, dataset.transactions)
plt.show()
```

<p>
    <img alt="hill_Slope" src="https://github.com/HeniPatel423/AIDI_1002_Final_Project/blob/main/datasets/hill_slopes.png">
</p>


### Command line interface

We provide a simple command line interface, which allows you to easily
mine association rules on any input dataset, output them to a csv file and/or perform
a simple statistical analysis on them. For more details see the [documentation](https://niaarm.readthedocs.io/en/latest/cli.html).

```shell
niaarm -h
```

```
usage: niaarm [-h] [-v] [-c CONFIG] [-i INPUT_FILE] [-o OUTPUT_FILE] [--squashing-similarity {euclidean,cosine}] [--squashing-threshold SQUASHING_THRESHOLD] [-a ALGORITHM] [-s SEED] [--max-evals MAX_EVALS] [--max-iters MAX_ITERS]
              [--metrics METRICS [METRICS ...]] [--weights WEIGHTS [WEIGHTS ...]] [--log] [--stats]

Perform ARM, output mined rules as csv, get mined rules' statistics

options:
  -h, --help            show this help message and exit
  -v, --version         show program's version number and exit
  -c CONFIG, --config CONFIG
                        Path to a TOML config file
  -i INPUT_FILE, --input-file INPUT_FILE
                        Input file containing a csv dataset
  -o OUTPUT_FILE, --output-file OUTPUT_FILE
                        Output file for mined rules
  --squashing-similarity {euclidean,cosine}
                        Similarity measure to use for squashing
  --squashing-threshold SQUASHING_THRESHOLD
                        Threshold to use for squashing
  -a ALGORITHM, --algorithm ALGORITHM
                        Algorithm to use (niapy class name, e.g. DifferentialEvolution)
  -s SEED, --seed SEED  Seed for the algorithm's random number generator
  --max-evals MAX_EVALS
                        Maximum number of fitness function evaluations
  --max-iters MAX_ITERS
                        Maximum number of iterations
  --metrics METRICS [METRICS ...]
                        Metrics to use in the fitness function.
  --weights WEIGHTS [WEIGHTS ...]
                        Weights in range [0, 1] corresponding to --metrics
  --log                 Enable logging of fitness improvements
  --stats               Display stats about mined rules
```
Note: The CLI script can also run as a python module (`python -m niaarm ...`)

## Reference Papers:

Ideas are based on the following research papers:

[1] I. Fister Jr., A. Iglesias, A. Gálvez, J. Del Ser, E. Osaba, I Fister. [Differential evolution for association rule mining using categorical and numerical attributes](http://www.iztok-jr-fister.eu/static/publications/231.pdf) In: Intelligent data engineering and automated learning - IDEAL 2018, pp. 79-88, 2018.

[2] I. Fister Jr., V. Podgorelec, I. Fister. [Improved Nature-Inspired Algorithms for Numeric Association Rule Mining](https://link.springer.com/chapter/10.1007/978-3-030-68154-8_19). In: Vasant P., Zelinka I., Weber GW. (eds) Intelligent Computing and Optimization. ICO 2020. Advances in Intelligent Systems and Computing, vol 1324. Springer, Cham.

[3] I. Fister Jr., I. Fister [A brief overview of swarm intelligence-based algorithms for numerical association rule mining](https://arxiv.org/abs/2010.15524). arXiv preprint arXiv:2010.15524 (2020).

[4] Fister, I. et al. (2020). [Visualization of Numerical Association Rules by Hill Slopes](http://www.iztok-jr-fister.eu/static/publications/280.pdf).
    In: Analide, C., Novais, P., Camacho, D., Yin, H. (eds) Intelligent Data Engineering and Automated Learning – IDEAL 2020.
    IDEAL 2020. Lecture Notes in Computer Science(), vol 12489. Springer, Cham. https://doi.org/10.1007/978-3-030-62362-3_10

[5] I. Fister, S. Deb, I. Fister, [Population-based metaheuristics for Association Rule Text Mining](http://www.iztok-jr-fister.eu/static/publications/260.pdf),
    In: Proceedings of the 2020 4th International Conference on Intelligent Systems, Metaheuristics & Swarm Intelligence,
    New York, NY, USA, mar. 2020, pp. 19–23. doi: [10.1145/3396474.3396493](https://dl.acm.org/doi/10.1145/3396474.3396493).

[6] I. Fister, I. Fister Jr., D. Novak and D. Verber, [Data squashing as preprocessing in association rule mining](https://iztok-jr-fister.eu/static/publications/300.pdf), 2022 IEEE Symposium Series on Computational Intelligence (SSCI), Singapore, Singapore, 2022, pp. 1720-1725, doi: [10.1109/SSCI51031.2022.10022240](https://doi.org/10.1109/SSCI51031.2022.10022240).

[7] Iztok Fister Jr. and Iztok Fister, A brief overview of swarm intelligence-based algorithms for numerical association rule mining.https://arxiv.org/pdf/2010.15524v1.pdf
