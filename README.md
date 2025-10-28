## Introduction
This project aims to implement a vulnerability detection function based on code slicing and graph neural network (GNN). It can automatically identify potential vulnerabilities in code through deep learning. The project extracts code structure information by constructing a code property graph (CPG) through code slicing, analyzes key code slices, and trains a graph neural network model (Ordered GNN) to achieve accurate vulnerability detection.
## Directory Structure
```text
slice/
├── data/                 # Data storage directory
│   ├── exception/        # Slicing exception records
│   └── index.json        # Index of all files, maintaining the mapping relationship with CPG data
├── joernslice/           # Code slicing module
│   ├── slice.py          # Code slicing core implementation
│   ├── gen_cpg.py        # Generate code property graph (CPG)
│   ├── reader.py         # Data reading and processing
│   └── neo4j_admin_generator.py # Neo4j database import tool
├── vul_test/             # Vulnerability detection model training module
│   ├── model.py          # GNN model definition
│   ├── train.py          # Training core implementation
│   └── train_gonn.py     # Training script
└── requirements.txt      # Project dependency list
```
## Main Functions
1. **Code Property Graph (CPG) Construction**
* Use Joern tool to generate CPG from source code
* Support exporting CPG as Neo4j CSV format for graph database storage
2. **Code Slicing**
* Implement forward/backward code slicing to analyze vulnerability-related code slices
* Automatically record slicing failure cases in slice_good_error.txt/slice_bad_error.txt
3. **Vulnerability Detection Model**
* Implement graph neural network (Ordered GNN) as the core graph classification model to process graph structure data
* Support hierarchical graph structure-based vulnerability detection
* Provide model training and validation evaluation functions
## Dependencies
```text
dgl==1.1.3
gensim==4.2.0
matplotlib==3.5.3
nltk==3.8.1
numpy==1.21.2
pandas==1.3.5
pytorch_warmup==0.1.1
scikit_learn==1.0.2
tokenizers==0.13.3
torch==1.13.0
torch_geometric==2.3.1
torchtext==0.11.0
tqdm==4.61.2
transformers==4.30.2
```
Installation command:
```bash
pip install -r requirements.txt
```
## Usage Instructions
1. Environment Preparation
* Install the Joern toolchain and configure environment variables to ensure that the joern command can be directly invoked in the terminal.
* Install the Neo4j database (optional, used for CPG visualization and analysis).
* Prepare the code sample dataset and place it in the specified directory.
2. Generate Code Property Graph (CPG)
```bash
python joernslice/gen_cpg.py
```
* This script generates CPG binary files from source code 
* Export CPG as Neo4j CSV format and store locally
3. ִExecute Code Slicing
```bash
python joernslice/slice.py
```
* Perform forward and backward slicing on the code 
* Extract key code slices related to vulnerabilities
* Exception information is recorded in the data/exception directory
4. Model Training and Evaluation
```bash
# Method 1: Use the predefined GNN model for training to detect vulnerabilities in code slices
python vul_test/train_gonn.py

# Method 2: Use the custom training core to customize the model
python vul_test/train.py
```
* Train the Ordered GNN vulnerability detection model
* Support hierarchical graph structure-based classification tasks
* Output evaluation metrics such as precision, recall, accuracy, and F1 score

## Model Description
The project focuses on graph neural networks (Ordered GNN). The model details are as follows:
1. **GGNN (Gated Graph Neural Network)**
* Support node attribute information integration
* Model graph structure through node connections
* Specialize in learning structural features related to vulnerabilities
2. **Ordered GNN (Ordered Graph Neural Network)**
* Incorporate node order information to pay attention to execution sequences
* Integrate dual information: code execution + structural features
* Support vulnerability detection for complex code structures

## Notes
1. Ensure the Joern tool is correctly installed and configured with environment variables before use
2. Neo4j database needs to be pre-installed for CPG storage and query
3. The dataset needs to be uploaded to the code source directory in advance
4. Exception logs record details of slicing failures and model training errors

## References
```BibTex
@inproceedings{song2023ordered,
    title={Ordered GNN: Ordering Message Passing to Deal with Heterophily and Over-smoothing},
    author={Yunchong Song and Chenghu Zhou and Xinbing Wang and Zhouhan Lin},
    booktitle={The Eleventh International Conference on Learning Representations},
    year={2023}
}
```
