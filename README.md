## FinMMDocR

This project was developed with the assistance of modern AI-powered development tools, including Cursor IDE and Tongyi Qianwen. All code has been carefully reviewed to ensure originality and compliance with best practices. The implementation represents original work by the authors.

The data and code for the paper `FinMMDocR: Benchmarking Financial Multimodal Reasoning with Scenario Awareness, Document Understanding, and Multi-Step Computation`.

**FinMMDocR** is a new bilingual multimodal benchmark for evaluating MLLMs in financial numerical reasoning, featuring real-world scenarios, visually-rich documents, and multi-step computations.

## Table of Contents

- [Environment Setup](#environment-setup)
- [Dataset Overview](#finmmdoc-dataset)
- [Data Preprocessing](#data-preprocessing)
- [RAG Retrieval System](#rag-retrieval-system)
- [Inference Pipeline](#running-inference)
- [Evaluation Framework](#automated-evaluation)
- [Project Structure](#project-structure)

## Environment Setup

You can install the dependencies by the following command:
```bash
pip install -r requirements.txt
```

## FinMMDocR Dataset

The FinMMDocR dataset is a comprehensive multimodal benchmark containing **1,200 carefully crafted financial reasoning questions** derived from real-world financial research documents. Each question requires multimodal understanding, combining text analysis, visual document interpretation, and numerical reasoning.

### Data Availability

**Current Release**: Due to AAAI submission requirements limiting supplementary materials to 50MB, we currently provide:
- **Complete Dataset**: Full `test.json` with all 1,200 questions and complete data structure
- **Sample Images**: One document (`0000`) with 10 associated images and all OCR results
- **Full Dataset Images**: Complete image dataset for all 1,200 questions will be publicly released after paper acceptance

**What Will Be Released After Acceptance**:
- Complete image dataset for all 1,200 questions (all document images)
- All OCR results for full document collection
- Additional preprocessing variants (images_50, images_15, etc.)

**Note**: The current release includes the complete question dataset with all 1,200 questions, but only provides sample images for one document due to size constraints. This allows reviewers to understand the complete data structure and run experiments with the available sample data, while the full image dataset will be released after paper acceptance.

### Dataset Overview

- **Total Questions**: 1,200 questions (test-0 to test-1199)
- **Document Types**: 9 major categories of financial research
- **Multimodal Components**: Text + Images + Numerical data
- **Language**: Bilingual (Chinese and English)
- **Domain**: Financial analysis and numerical reasoning


### Data Structure

Each question in the dataset follows this comprehensive structure:

```json
{
    "question_id": "test-0",
    "doc_id": "0000",
    "doc_type": "Market Interpretation",
    "question": "Assume the Information Technology sector maintains its past 5-year annualized sales growth rate as reported over the next three years. Project the sector's total annual sales revenue three years from the report date. Use the sector's Total Market Cap and P/S ratio provided in the report for your calculations. Provide the answer in billions of USD, rounded to two decimal places.",
    "evidence": {
        "table": [9],
        "image": [],
        "plain_text": [],
        "generalized_text (layout)": [],
        "pie_chart": [],
        "bar_chart": [],
        "scatter_chart": [],
        "line_chart": []
    },
    "python_solution": "import numpy as np\n\ndef solution():\n    # Define variables with their values from the report\n    it_market_cap_billion_usd = 15035.99 # Table 11, page 9\n    it_ps_ratio = 5.15                 # Table 11, page 9\n    it_sales_growth_rate = 0.1673      # Table 11, page 9 (16.73%)\n\n    # Calculate current annual sales\n    current_sales = it_market_cap_billion_usd / it_ps_ratio\n\n    # Calculate projected sales in 3 years\n    num_years = 3\n    projected_sales = current_sales * np.power(1 + it_sales_growth_rate, num_years)\n\n    # Round final result to two decimal places\n    answer = round(projected_sales, 2)\n\n    # Return final result\n    return answer",
    "ground_truth": 4643.79,
    "source_id": "0000-01",
    "pages_num": 15,
    "images": [
        "/data/images/0000/page_1.png",
        "/data/images/0000/page_2.png",
        "/data/images/0000/page_3.png",
        "/data/images/0000/page_4.png",
        "/data/images/0000/page_5.png",
        "/data/images/0000/page_6.png",
        "/data/images/0000/page_7.png",
        "/data/images/0000/page_8.png",
        "/data/images/0000/page_9.png",
        "/data/images/0000/page_10.png",
        "/data/images/0000/page_11.png",
        "/data/images/0000/page_12.png",
        "/data/images/0000/page_13.png",
        "/data/images/0000/page_14.png",
        "/data/images/0000/page_15.png"
    ],
    "texts": "/data/texts/0000.json"
}
```

### Key Components

#### 1. **Question Information**
- `question_id`: Unique identifier (test-0 to test-1199)
- `doc_id`: Source document identifier
- `doc_type`: Category of financial research
- `question`: Detailed financial reasoning question requiring multimodal analysis

#### 2. **Evidence Tracking**
The `evidence` field tracks which visual elements are relevant:
- `table`: Page numbers containing relevant tables
- `image`: General images (charts, graphs, etc.)
- `plain_text`: Text-only content
- `generalized_text (layout)`: Layout-aware text extraction
- `pie_chart`, `bar_chart`, `scatter_chart`, `line_chart`: Specific chart types

#### 3. **Multimodal Resources**
- `images`: Array of image paths for each page of the source document
- `texts`: Path to extracted text content from the document
- `pages_num`: Total number of pages in the source document

#### 4. **Solution and Ground Truth**
- `python_solution`: Expert-written Python code with clear variable names and execution logic
- `ground_truth`: Numerical answer (typically float) derived from executing the solution
- `source_id`: Specific identifier linking to the source document

### Example Question Analysis

**Question**: "Assume the Information Technology sector maintains its past 5-year annualized sales growth rate as reported over the next three years. Project the sector's total annual sales revenue three years from the report date."

**Required Skills**:
1. **Document Understanding**: Locate and extract relevant financial data from tables
2. **Numerical Reasoning**: Apply growth rate calculations and projections
3. **Financial Knowledge**: Understand P/S ratios, market cap, and sector analysis
4. **Multimodal Processing**: Combine information from text and visual elements

**Solution Approach**:
1. Extract IT sector market cap ($15,035.99 billion) and P/S ratio (5.15) from Table 11
2. Calculate current annual sales: $15,035.99 ÷ 5.15 = $2,919.61 billion
3. Apply 16.73% annual growth rate for 3 years: $2,919.61 × (1.1673)³
4. Result: $4,643.79 billion

## Data Preprocessing

The FinMMDocR dataset includes comprehensive preprocessing tools to handle multimodal document images with different resolution and quantity constraints.

### Image Quantity Control

The `merge_image.py` script manages image quantity through intelligent concatenation strategies:

**Processing Strategy**:
- **≤50 images**: Direct copy without modification
- **>50 images**: Intelligent concatenation to reduce total image count
- **Special handling**: Different column layouts for specific document types

**Concatenation Parameters**:
```python
# For documents with >50 images
k = math.ceil(len(image_list)/50)  # Target image count
concat_num = len(image_list)//k     # Images per concatenated result
column_num = k                      # Column layout for concatenation
```

**Output Structure**:
```
data/
├── images/           # Original images (variable count)
└── images_50/       # Processed images (≤50 per document)
    ├── 0000/
    │   ├── page_1.png
    │   ├── page_2.png
    │   └── ...
    └── 0001/
        ├── page_1.png
        └── ...
```

**Usage**:
```bash
python merge_image.py
```

### Image Resolution Processing

The `resize_image.py` script handles multiple resolution requirements:

**Supported Resolutions**:
- **Original**: Maintains original image quality
- **3840px**: High-resolution processing for detailed analysis
- **1920px**: Standard resolution for balanced performance

**Processing Logic**:
```python
def resize_to_1920(image_path, output_path=None):
    """
    Resize the maximum side of the image to 1920 while maintaining aspect ratio
    """
    # Calculate scaling ratio
    if width > height:
        scale = 3840 / width      # Width-based scaling
    else:
        scale = 3840 / height     # Height-based scaling
    
    # Apply high-quality resizing
    resized_img = img.resize((new_width, new_height), Image.LANCZOS)
```

**Resolution Variants**:
```
data/
├── images_15/              # ≤15 images per document
│   ├── 0000/
│   └── 0001/
├── images_15_1920/        # 1920px resolution
│   ├── 0000/
│   └── 0001/
└── images_15_3840/        # 3840px resolution
    ├── 0000/
    └── 0001/
```

**Usage**:
```bash
python resize_image.py
```

### Preprocessing Pipeline

**Complete Workflow**:
1. **Image Quantity Control**:
   ```bash
   python merge_image.py
   # Creates images_50/ with ≤50 images per document
   ```

2. **Further Quantity Reduction** (if needed):
   ```bash
   # Manual processing for ≤15 images
   # Creates images_15/ directory
   ```

3. **Resolution Processing**:
   ```bash
   python resize_image.py
   # Creates images_15_3840/ with high-resolution images
   ```

**Configuration Options**:
- **Process Count**: Control number of documents processed
- **Parallel Processing**: 32 concurrent processes for efficiency
- **Memory Management**: Automatic image cleanup to prevent memory overflow
- **Error Handling**: Comprehensive logging and retry mechanisms


## Running Inference
We support inference with various LLM models through the `inference.py` script. The inference system supports both text and multimodal inputs with automatic retry mechanisms and rate limiting.

**Direct Script Execution**
```bash
python inference.py
```

**Configuration Setup**
Before running inference, you need to configure the following parameters in the `Config` class within `inference.py`:

1. **Dataset Configuration**:
   ```python
   dataset_file: str = "path/to/your/dataset.json"  # Dataset file path
   outcome_dir: str = "path/to/output/directory"     # Root directory for saving output results
   ```

2. **Model Configuration**:
   ```python
   # Choose one of the supported models:
   model_name: str = "qwen/qwen2.5-vl-32b-instruct"  # Currently active
   # model_name: str = "google/gemini-2.5-pro-preview-03-25"
   # model_name: str = "openai/gpt-4o"
   # model_name: str = "meta-llama/llama-4-maverick"
   # model_name: str = "anthropic/claude-3.7-sonnet"
   # And many more...
   ```

3. **API Configuration**:
   ```python
   # OpenRouter (default)
   client: AsyncOpenAI = AsyncOpenAI(
       api_key="your_api_key",
       base_url="https://openrouter.ai/api/v1",
   )
   
   # Or use other providers:
   # Alibaba DashScope
   # XAI Grok
   # Anthropic Claude
   ```

4. **Processing Parameters**:
   ```python
   rpm: int = 100                                    # Requests per minute limit
   max_no_improve_round_count: int = 3               # Maximum retry rounds for failed requests
   process_count: int = -1                           # Number of items to process (-1 for all)
   max_input_images: int = 1000                      # Maximum number of input images
   ```

**Input Data Format**
The script expects a JSON dataset with the following structure:
```json
[
    {
        "question_id": "unique_id",
        "question": "question text",
        "system_input": "optional system prompt",
        "user_input": "user question or instruction",
        "images": ["path/to/image1.png", "path/to/image2.png"]
    }
]
```

**Output Structure**
The inference process creates organized output directories:
```
output_dir/
├── dataset_name/
│   └── model_name/
│       └── timestamp_process_count_X/
│           ├── execution.log
│           ├── final_results.json
│           └── rounds_outcome/
│               ├── round_1.json
│               ├── round_2.json
│               └── ...
```

**Key Features**:
- **Async Processing**: Uses asyncio for efficient concurrent API calls
- **Rate Limiting**: Built-in rate limiting to respect API quotas
- **Automatic Retries**: Failed requests are automatically retried up to 3 rounds
- **Multimodal Support**: Handles both text and image inputs with base64 encoding
- **Progress Tracking**: Real-time progress bars and detailed logging
- **Error Handling**: Comprehensive error handling with detailed error messages
- **Round-based Processing**: Saves intermediate results after each round for recovery

### Automated Evaluation

The evaluation system supports both Chain-of-Thought (COT) and Program-of-Thought (POT) evaluation methods with comprehensive metrics and automated processing.

**Individual Model Evaluation**
```bash
python evaluation.py \
    --prediction_path "outputs/test/raw_cot_outputs/model_output.json" \
    --evaluation_output_dir "outputs/test/processed_cot_outputs" \
    --prompt_type "cot" \
    --ground_truth_file "data/test.json" \
    --result_file "outputs/results/test_cot_results.json" \
    --api_base "https://openrouter.ai/api/v1" \
    --api_key "your_api_key"
```

**Batch Evaluation Script**
```bash
bash scripts/evaluate_all.sh
```

**Evaluation Configuration**
The `evaluate_all.sh` script supports the following configurations:

1. **Prompt Types**: 
   - `cot` (Chain-of-Thought) - Currently active
   - `pot` (Program-of-Thought) - Commented out

2. **Dataset Subsets**:
   - `test` - Currently active
   - Can be extended to include `train`, `validation`

3. **Model Outputs**: 
   The script includes a comprehensive list of supported model outputs:
   ```bash
   model_outputs=(
       "doubao_vision_rag_vidorag_0720.json"  # Currently active
       # "o4-mini-high.json"
       # "gpt4o.json"
       # "claude37_thinking.json"
       # "gemini.json"
       # "grok2.json"
       # "llama4.json"
       # "gemma3.json"
       # "mistral.json"
       # "qwen2_5vl.json"
       # And many more...
   )
   ```

**Evaluation Methods**

1. **Chain-of-Thought (COT) Evaluation**:
   - Uses LLM-based answer extraction to parse numerical answers from model responses
   - Supports automatic retry for failed extractions
   - Calculates accuracy by comparing extracted answers with ground truth
   - Provides execution rate metrics

2. **Program-of-Thought (POT) Evaluation**:
   - Extracts executable Python code from model responses
   - Executes code in isolated processes with timeout protection
   - Compares execution results with ground truth
   - Includes comprehensive error handling and security measures

**Evaluation Metrics**
The evaluation system provides the following metrics:
- **Accuracy**: Percentage of correct answers
- **Execution Rate**: Percentage of successfully processed responses
- **Token Usage**: Total completion tokens consumed
- **Processing Statistics**: Detailed breakdown by model and prompt type

**Output Structure**
Evaluation results are organized as follows:
```
outputs/
├── test/
│   ├── raw_cot_outputs/          # Raw model outputs
│   ├── processed_cot_outputs/     # Processed evaluation results
│   └── raw_pot_outputs/          # Raw POT outputs
├── results/
│   ├── test_cot_results.json     # Aggregated COT results
│   └── test_pot_results.json     # Aggregated POT results
```

**Key Features**:
- **Multi-process Safety**: Code execution in isolated processes with timeout
- **Security Measures**: Disabled dangerous functions and system calls
- **Automatic Retry**: Failed answer extractions are automatically retried
- **Comprehensive Logging**: Detailed progress tracking and error reporting
- **Flexible Configuration**: Support for multiple models, prompt types, and datasets
- **Batch Processing**: Automated evaluation of multiple models simultaneously

## Project Structure

The FinMMDocR project is organized into several key directories, each serving specific purposes in the multimodal document processing and evaluation pipeline.

### Core Directories

```
FinMMDocR/
├── data/                    # Dataset and preprocessing outputs
│   ├── images/             # Original document images (variable count)
│   ├── images_50/          # Processed images (≤50 per document)
│   ├── images_15/          # Further reduced images (≤15 per document)
│   ├── images_15_1920/     # 1920px resolution images
│   ├── images_15_3840/     # 3840px resolution images
│   ├── texts/              # OCR results for each document
│   └── test.json           # Main dataset file (1,200 questions)
├── models/                  # RAG embedding model parameters
├── outputs/                 # Inference and evaluation results
│   ├── test/
│   │   ├── raw_cot_outputs/     # Raw Chain-of-Thought outputs
│   │   ├── processed_cot_outputs/ # Processed COT evaluation results
│   │   └── raw_pot_outputs/     # Raw Program-of-Thought outputs
│   └── results/            # Aggregated evaluation results
├── retrieved_results/       # RAG experiment results
├── scripts/                # Automation and evaluation scripts
└── utils/                  # Utility functions and tools
```

### Directory Purposes

#### 1. **`data/` - Dataset and Preprocessing**
- **`images/`**: Original document images with variable page counts
- **`images_50/`**: Images processed to ≤50 per document via concatenation
- **`images_15/`**: Further reduced to ≤15 images per document
- **`images_15_1920/`**: 1920px resolution variants for performance optimization
- **`images_15_3840/`**: 3840px resolution variants for high-quality analysis
- **`texts/`**: **OCR results for each document**, enabling text-based retrieval and analysis
- **`test.json`**: Main dataset containing 1,200 multimodal financial reasoning questions

#### 2. **`models/` - RAG Embedding Models**
- **Purpose**: Stores parameters for RAG (Retrieval-Augmented Generation) embedding models
- **Usage**: Enables semantic search and document retrieval for multimodal reasoning
- **Content**: Model weights, configurations, and embeddings for financial document understanding

#### 3. **`outputs/` - Experiment Results**
- **`test/raw_cot_outputs/`**: Raw model responses for Chain-of-Thought evaluation
- **`test/processed_cot_outputs/`**: Processed and evaluated COT results
- **`test/raw_pot_outputs/`**: Raw model responses for Program-of-Thought evaluation
- **`results/`**: Aggregated evaluation metrics and comparative analysis

#### 4. **`retrieved_results/` - RAG Experiment Results**
- **Purpose**: Stores results from RAG-based experiments
- **Content**: Retrieved document chunks, relevance scores, and retrieval-augmented responses
- **Usage**: Analysis of retrieval performance and multimodal document understanding

#### 5. **`scripts/` - Automation Tools**
- **`evaluate_all.sh`**: Batch evaluation script for multiple models
- **Purpose**: Automated evaluation pipeline for comprehensive model comparison

#### 6. **`utils/` - Utility Functions**
- **`evaluation_utils.py`**: Evaluation metrics and answer extraction
- **`openai_utils.py`**: API integration and model communication
- **`bm25_utils.py`**: Text-based retrieval utilities
- **`BGE-M3.py`**: BGE-M3 embedding model integration

### Data Flow

**Preprocessing Pipeline**:
```
Original Images → merge_image.py → images_50/ → resize_image.py → images_15_3840/
```

**OCR Processing**:
```
Document Images → OCR Processing → data/texts/ → RAG Retrieval
```

**Experiment Pipeline**:
```
Dataset → Inference → outputs/ → Evaluation → results/
RAG Models → Retrieved Results → retrieved_results/
```

## RAG Retrieval System

The FinMMDocR project includes a comprehensive Retrieval-Augmented Generation (RAG) system that supports both parallel and serial processing for document retrieval experiments.

### Overview

The RAG system enables semantic search and document retrieval for multimodal financial reasoning by:
- **Text-based Retrieval**: Using OCR results from `data/texts/` for semantic search
- **Multiple Embedding Models**: Support for various embedding models (OpenAI, HuggingFace, BM25)
- **Parallel/Serial Processing**: Efficient processing options for different experimental needs
- **Retrieval Results**: Stored in `retrieved_results/` for analysis

### Supported Retrieval Methods

#### 1. **BM25 Retrieval**
- **Type**: Traditional keyword-based retrieval
- **Usage**: Baseline retrieval method for comparison
- **Implementation**: `utils/bm25_utils.py`
- **Features**: TF-IDF based scoring with text preprocessing

#### 2. **OpenAI Embedding Models**
- **Models**: `text-embedding-3-small`, `text-embedding-3-large`, `text-embedding-ada-002`
- **Type**: Semantic embedding-based retrieval
- **Features**: High-quality semantic understanding
- **Usage**: Advanced semantic search capabilities

#### 3. **HuggingFace Models**
- **Models**: `contriever-msmarco` and other custom models
- **Type**: Local embedding models
- **Features**: Offline processing, customizable
- **Usage**: Cost-effective semantic retrieval

### Processing Modes

#### **Parallel Processing (`retrieve_parallel.py`)**
```bash
python retrieve_parallel.py
```

**Key Features**:
- **Multi-process Execution**: 32 concurrent processes for high throughput
- **Progress Tracking**: Real-time progress bars with detailed logging
- **Memory Optimization**: Efficient batch processing and memory management
- **Error Handling**: Robust error recovery and retry mechanisms

**Configuration**:
```python
# Model selection
model_name = "bm25"  # or "text-embedding-3-small", "contriever-msmarco"

# Processing parameters
num_processes = 32   # Number of parallel processes
top_k = -1          # Retrieve all documents (-1) or specific number
```

#### **Serial Processing (`retrieve_serial.py`)**
```bash
python retrieve_serial.py
```

### Retrieval Pipeline

**Complete Workflow**:
```
1. Load Dataset → 2. Process Queries → 3. Retrieve Documents → 4. Save Results
```

**Detailed Process**:
1. **Data Loading**: Load questions from `data/test.json`
2. **Text Processing**: Extract OCR results from `data/texts/`
3. **Embedding Generation**: Create embeddings for queries and documents
4. **Similarity Search**: Find top-k most relevant documents
5. **Result Storage**: Save to `retrieved_results/{model_name}.json`

### Output Format

**Retrieval Results Structure**:
```json
[
    {
        "question_id": "test-0",
        "question": "Financial reasoning question...",
        "retrieved_results": [
            {
                "page": 1,
                "score": 0.85
            },
            {
                "page": 3,
                "score": 0.72
            }
        ]
    }
]
```

### Configuration Options

**Model Selection**:
```python
EMBEDDING_MODELS = {
    "contriever-msmarco": {
        "type": "hf",
        "path": "model_path",
        "projection_size": 768,
    },
    "text-embedding-3-small": {
        "type": "openai",
        "model": "text-embedding-3-small",
        "projection_size": 1536,
    },
    "bm25": {
        "type": "keyword",
        "features": "TF-IDF based"
    }
}
```

**Processing Parameters**:
- **`top_k`**: Number of documents to retrieve (-1 for all)
- **`batch_size`**: Embedding generation batch size
- **`max_tokens`**: Maximum tokens for text truncation
- **`n_subquantizers`**: Index compression parameters

### Usage Examples

**Basic Retrieval**:
```bash
# Parallel processing with BM25
python retrieve_parallel.py

# Serial processing with OpenAI embeddings
python retrieve_serial.py
```

**Custom Configuration**:
```python
# Modify model selection in script
model_name = "text-embedding-3-large"

# Adjust processing parameters
num_processes = 16  # Reduce for memory constraints
top_k = 10         # Retrieve top 10 documents
```

**Results Analysis**:
```python
# Load retrieval results
with open("./retrieved_results/bm25.json", "r") as f:
    results = json.load(f)

# Analyze retrieval performance
for result in results:
    print(f"Question: {result['question_id']}")
    print(f"Retrieved {len(result['retrieved_results'])} documents")
```