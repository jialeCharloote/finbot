# Financial Sentiment Analysis Model

## Abstract

This project aims to fine-tune a pre-trained LLM to enhance financial sentiment analysis on financial news, both in terms of classification and reasoning analysis. This work develops a Question-Answering (QA) bot capable of providing more precise and context-aware market sentiment analysis.

## Model Availability

The fine-tuned model is hosted on Hugging Face and can be accessed at:
https://huggingface.co/jialeCharlotte/econbot

To use this model in your own applications, you can load it directly:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = "distilgpt2"
model = AutoModelForCausalLM.from_pretrained(base_model)

# Load the fine-tuned adapter
model = PeftModel.from_pretrained(model, "jialeCharlotte/econbot")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model)
```

## Repository Structure

- **01Dataset**: Financial news dataset used for training
  - `balanced_news.json`: Balanced dataset of financial news
  - `finan_news.json`: Raw financial news data
  - `merged_news.json`: Combined news dataset
  - `preprocess.py`: Script for data preprocessing
  - `stock_tickers.txt`: List of stock tickers used in analysis

- **02Training**: Training scripts and logs
  - `finetune-log.txt`: Training logs
  - `finetune.sh`: Training shell script
  - `finetune.txt`: Training parameters and configuration
  - `train.py`: Main training script

- **04WebDeploy**: Web deployment files
  - `app.py`: Flask application for serving the model
  - `ProjectStructure`: Documentation of project structure


## Methodology

### Data Collection
Financial news and sentiment reasoning analysis text data was sourced from both Polygon APIs and Kaggle open dataset. We merge the two datasets to create one large dataset including 5700+ news articles for fine-tuning.

### Data Preprocessing
1. **Data Restructuring**: Reformatted the dataset to associate each news article with only one ticker at a time and removed minority undetermined sentiment labels ('mixed', 'NA', or 'neutral/positive').
2. **Data Augmentation**: Addressed sentiment imbalance (data skewness) using back translation (English → German → English) to increase distribution weights of underrepresented sentiment categories (neutral and negative) towards a more robust model.
3. **Train & Test Set Split**: Utilized stratification sampling to create training and testing sets, ensuring that the proportion of sentiment categories remains the same in both sets to prevent bias.

### Model Fine-Tuning
Fine-tuned a DistilGPT-2 model using the PEFT (Parameter-Efficient Fine-Tuning) approach with LoRA adapters based on the dataset we built, involving supervised learning with labeled financial sentiment data and reasoning analysis.

### Model Evaluation & Testing
The fine-tuned model was tested on unseen financial text data. Classification performance was assessed using accuracy, precision, recall, and F1-score. Text generating capability was evaluated based on perplexity.

## Data Sources

- **Polygon.io APIs**: We collected the most recent 100 financial news summaries with sentiment classification and reasoning analysis per ticker. From a US listed company pool, we removed duplicated data to create a dataset including around 200+ text entries.
- **Kaggle open dataset**: We used the "Financial News with Ticker-Level Sentiment" dataset, also sourced from Polygon.io APIs, containing 5000+ text entries without duplication and in the same format as the recent news dataset.

## Model Description

This model analyzes financial news headlines and descriptions to determine market sentiment for specific stock tickers. It generates sentiment analysis as either:

- **Bullish**: Positive outlook, suggesting potential stock value increase
- **Bearish**: Negative outlook, suggesting potential stock value decrease
- **Neutral**: Balanced outlook with no strong positive or negative indicators

The model uses LoRA (Low-Rank Adaptation) for efficient fine-tuning of DistilGPT-2 on financial news data, preserving the base model while adding domain-specific knowledge.

## Results & Evaluations

- The fine-tuned model demonstrated improved sentiment classification accuracy compared to baseline FinBERT model.
- Evaluation metrics (accuracy, precision, recall, F1-score) were used to measure model performance.
- Additional testing focused on reasoning capabilities and interpretability improvements.

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
   python app.py
   ```

4. Access the web interface at http://localhost:5000

## Discussions

- The model showed strong performance in financial sentiment classification but had limitations in handling nuanced sentiments in complex financial statements.
- Future improvements could include integrating more structured financial data (e.g., stock market movements) to refine sentiment correlation.
- Exploring ensemble methods or hybrid approaches (e.g., combining rule-based analysis with deep learning) may enhance interpretability.

## Limitations

- The model analyzes text only and does not incorporate quantitative financial data
- Predictions are based on textual content and may not capture all market factors
- The model should be used as one of many tools for financial analysis, not as the sole decision-maker

## Conclusion

This project successfully fine-tuned GPT-2 to enhance financial sentiment classification for market assessment applications. The results indicate the feasibility of AI-driven financial sentiment analysis in improving decision-making. Further enhancements can focus on refining reasoning capabilities and incorporating multimodal financial data.

## References

- Hugging Face Distil GPT-2 documentation
- Financial News APIs (Kaggle, Polygon.io)
- Parameter-Efficient Fine-Tuning (PEFT) documentation

## License

This project is licensed under the Apache 2.0 License.