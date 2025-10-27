# Vulnerability Detection Based on Code Slicing and Graph Neural Network
## Introduction
This project aims to implement a vulnerability detection function based on code slicing and graph neural network (GNN) technology. Specifically, it can automatically identify potential vulnerabilities in code through program analysis and machine learning. The project extracts code structure information by constructing code property graphs (CPG), analyzes key code slices through program slicing, and trains a GNN model to achieve accurate vulnerability detection.
## Directory Structure
```text
slice/
├── data/                 # Data storage directory
│   ├── exception/        # Slicing exception records
│   ├── cpg_bin_all_data/ # Code property graph (CPG) binary files
│   └── cpg_csv_all_data/ # CPG exported in Neo4j CSV format
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
1. Code Property Graph (CPG) Construction
* Use Joern tool to generate CPG from source code
* Support exporting CPG as Neo4j CSV format for graph database storage
2. Code Slicing
* Implement forward/backward code slicing to analyze vulnerability-related code slices
* Automatically record slicing failure cases in slice_good_error.txt/slice_bad_error.txt
3. Vulnerability Detection Model
* Implement graph neural network models (such as GGNN)
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
1. Generate Code Property Graph (CPG)
```bash
python joernslice/gen_cpg.py
```
* This script generates CPG binary files from source code 
* Export CPG as Neo4j CSV format and store locally
2. ִExecute Code Slicing
```bash
python joernslice/slice.py
```
* Perform forward and backward slicing on the code 
* Extract key code slices related to vulnerabilities
* Exception information is recorded in the data/exception directory
3. Model Training and Evaluation
```bash
python vul_test/train_gonn.py
```
* Train the GNN vulnerability detection model
* Support hierarchical graph structure-based classification tasks
* Output evaluation metrics such as precision, recall, accuracy, and F1 score

## Model Description
The project implements a graph neural network-based vulnerability detection method, including the following models:
1. GGNN (Gated Graph Neural Network)
* Support node attribute information integration
* Model graph structure through node connections
* Specialize in learning structural features related to vulnerabilities
2. GONN (Graph Overlay Neural Network)
* Introduce attention mechanism for feature enhancement
* Support hierarchical graph structure-based multi-level learning
* Effectively handle vulnerability feature differences and hidden patterns
## Notes
1. Ensure the Joern tool is correctly installed and configured with environment variables before use
2. Neo4j database needs to be pre-installed for CPG storage and query
3. The dataset needs to be uploaded to the code source directory in advance
4. Exception logs record details of slicing failures and model training errors
