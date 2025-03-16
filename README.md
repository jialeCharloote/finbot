# Financial Sentiment Analysis Bot (FinBot)

[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Model-yellow)](https://huggingface.co/jialeCharlotte/finbot)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

A Question-Answering bot for financial news sentiment analysis, using a fine-tuned language model to provide precise and context-aware market sentiment analysis.

## Abstract

This project aims to fine-tune a pre-trained LLM to enhance financial sentiment analysis on financial news, both in terms of classification and reasoning analysis. This work develops a Question-Answering (QA) bot capable of providing more precise and context-aware market sentiment analysis.


## Model Availability

The fine-tuned model is hosted on Hugging Face and can be accessed at:
[jialeCharlotte/finbot](https://huggingface.co/jialeCharlotte/finbot)

To use this model in your own applications, you can load it directly:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from huggingface_hub import login
import torch
import getpass

hugging_token = getpass.getpass("Enter your Hugging Face Token: ")
login(token=hugging_token)

base_model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
lora_model_name = "jialeCharlotte/finbot"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

# Load model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="fp4",
    bnb_4bit_compute_dtype=torch.float16
)
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config, 
    torch_dtype=torch.float16,
    device_map="auto"
)
model = PeftModel.from_pretrained(model, lora_model_name)
model.config.use_cache = False
```

## Features

- **Sentiment Classification**: Automatically categorize financial news as Bullish, Bearish, or Neutral
- **Sentiment Reasoning**: Generate explanations for why a news article has a particular sentiment
- **Interactive Web Interface**: User-friendly UI for sentiment analysis queries
- **API Access**: RESTful API endpoints for integration with other systems

## Repository Structure

```
finbot/
├── Dataset/               # Financial news datasets
│   ├── balanced_news.json     # Balanced dataset of financial news
│   ├── finan_news.json        # Raw financial news data
│   ├── merged_news.json       # Combined news dataset
│   ├── preprocess.py          # Script for data preprocessing
│   ├── stock_tickers.txt      # List of stock tickers used in analysis
│   └── loghistory.csv         # Training logging history
│
├── Trainning/             # Training scripts and notebooks
│   ├── finetuneds.py          # Main training script
│   ├── Fine_tuning_training.ipynb  # Training notebook for Kaggle
│   └── model-eval(1).ipynb    # Model evaluation script
│
├── WebDeployment/         # Web application deployment
│   ├── app.py                 # Flask application for serving the model
│   ├── ProjectStructure       # Documentation of project structure
│   └── appdeployment.ipynb    # Web deployment on Kaggle
│
└── README.md              # This file
```

## Methodology

### Data Collection
Financial news and sentiment reasoning analysis text data was sourced from both Polygon APIs and Kaggle open dataset. We merged the two datasets to create one large dataset including 5700+ news articles for fine-tuning.

### Data Preprocessing
1. **Data Restructuring**: Reformatted the dataset to associate each news article with only one ticker at a time and removed minority undetermined sentiment labels ('mixed', 'NA', or 'neutral/positive').
2. **Data Augmentation**: Addressed sentiment imbalance (data skewness) using back translation (English → German → English) to increase distribution weights of underrepresented sentiment categories (neutral and negative) towards a more robust model.
3. **Train & Test Set Split**: Utilized stratification sampling to create training and testing sets, ensuring that the proportion of sentiment categories remains the same in both sets to prevent bias.

### Model Fine-Tuning
Fine-tuned DeepSeek-R1-Distill-Qwen-1.5B using the PEFT (Parameter-Efficient Fine-Tuning) approach with LoRA adapters based on the dataset we built, involving supervised learning with labeled financial sentiment data and reasoning analysis.

### Model Evaluation & Testing
The fine-tuned model was tested on unseen financial text data. Classification performance was assessed using accuracy, precision, recall, and F1-score. Text generating capability was evaluated based on perplexity.

## Data Sources

- **Polygon.io APIs**: Most recent financial news with sentiment classification and reasoning analysis per ticker. We collected approximately 200+ unique text entries.
- **Kaggle open dataset**: "Financial News with Ticker-Level Sentiment" dataset, also sourced from Polygon.io APIs, containing 5000+ text entries without duplication.

## Model Description

This model analyzes financial news headlines and descriptions to determine market sentiment for specific stock tickers. It generates sentiment analysis as either:

- **Bullish**: Positive outlook, suggesting potential stock value increase
- **Bearish**: Negative outlook, suggesting potential stock value decrease
- **Neutral**: Balanced outlook with no strong positive or negative indicators

The model uses LoRA (Low-Rank Adaptation) for efficient fine-tuning on financial news data, preserving the base model while adding domain-specific knowledge.

## Results & Evaluations

- The model performs well for Bullish sentiment with 76% precision and 82% recall
- Moderate lexical similarity with reference texts (BLEU scores in the 0.2-0.3 range)
- Decent semantic alignment (BERTScore: 0.545)
- Moderate perplexity (39.61)

## Usage

### API Endpoint

The model is accessible through a Flask API with the following endpoint:

```
POST /analyze
```

Request body:
```json
{
  "ticker": "AAPL",
  "title": "Apple Inc. Reports Record Q4 Earnings",
  "description": "Apple announced record-breaking revenue in their latest earnings call, exceeding analyst expectations."
}
```

Response:
```json
{
  "success": true,
  "response": {
    "ticker": "AAPL",
    "sentiment": "Bullish",
    "sentiment_reasoning": "The news indicates positive financial performance with record-breaking revenue that exceeded analyst expectations, suggesting strong business momentum."
  }
}
```

### Local Deployment

1. Install requirements:
   ```bash
   pip install flask torch transformers peft bitsandbytes huggingface_hub flask-cors
   ```

2. Set up environment variables in `.env`:
   ```
   HUGGINGFACE_API_KEY=your_key_here
   ```

3. Run the Flask application:
   ```bash
   python WebDeployment/app.py
   ```

4. Access the web interface at http://localhost:5000

## Limitations

- The model analyzes text only and does not incorporate quantitative financial data
- Predictions are based on textual content and may not capture all market factors
- The model should be used as one of many tools for financial analysis, not as the sole decision-maker

## Future Improvements

- Integration with real-time market data
- Enhanced reasoning capabilities for complex financial statements
- Support for additional languages
- User feedback integration for continuous learning
- Ensemble methods or hybrid approaches (combining rule-based analysis with deep learning)

## Conclusion

This project successfully fine-tuned DeepSeek-R1-Distill-Qwen-1.5B to enhance financial sentiment classification for market assessment applications. The results indicate the feasibility of AI-driven financial sentiment analysis in improving decision-making. Further enhancements can focus on refining reasoning capabilities and incorporating multimodal financial data.

## References

- Hugging Face DeepSeek-R1-Distill-Qwen-1.5B documentation
- Financial News APIs (Kaggle, Polygon.io)
- Parameter-Efficient Fine-Tuning (PEFT) documentation

## License

This project is licensed under the Apache 2.0 License.

## Contributors

- [Charlotte Zhou](https://github.com/jialeCharloote)
- [Zhilin Zhu](https://github.com/zzhu345)
