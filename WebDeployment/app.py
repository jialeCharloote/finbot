from flask import Flask, render_template, request, jsonify
import os
import torch
import json
import re
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login
from peft import PeftConfig

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Hugging Face authentication
HF_TOKEN = os.getenv("HUGGINGFACE_API_KEY")
login(token=HF_TOKEN)

# Model configuration
# base_model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
base_model_name = "/Users/charlotte/Desktop/UChicago/25_winter/DATA37100-25Win/econ_chatbot/dsr1"  #already download
lora_model_name = "jialeCharlotte/finbot"

tokenizer = AutoTokenizer.from_pretrained(base_model_name,use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id


# For Mac with Apple Silicon
if torch.backends.mps.is_available():
    device_to_use = "mps"
elif torch.cuda.is_available():
    device_to_use = "cuda"
else:
    device_to_use = "cpu"

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="cpu"
)

# Create a new approach for loading the LoRA weights
try:
    # First, try the direct approach but with error handling
    model = PeftModel.from_pretrained(model, lora_model_name, is_trainable=False)
    model = model.merge_and_unload()
except KeyError as e:
    print(f"Got KeyError: {e}. Trying alternative loading approach...")
    
    # Alternative approach 1: Load configuration separately
    config = PeftConfig.from_pretrained(lora_model_name)
    if hasattr(config, "base_model_name_or_path"):
        print(f"Adapter was trained on: {config.base_model_name_or_path}")
    
    # Try loading with adapter_name parameter
    try:
        model = PeftModel.from_pretrained(model, lora_model_name, adapter_name="default")
        model = model.merge_and_unload()
    except Exception as e2:
        print(f"Second approach failed with: {e2}")
        print("Proceeding with base model only (without LoRA weights)")


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    data = request.json
    title = data.get('title', '')
    description = data.get('description', '')
    ticker = data.get('ticker', '')
    
    # Create the prompt
    prompt = f"""You are a financial analyst in a leading hedge fund. 
Analyze the sentiment of the following financial news for the given stock ticker step by step.
Title: "{title}"
Summary: "{description}"
Stock Ticker: {ticker}
Step 1: Identify key financial terms and their implications.
Step 2: Determine whether the news suggests market optimism, pessimism, or neutrality for this specific stock.
Step 3: Based on your analysis, classify the sentiment into one of the following categories:
- "Bullish": If the news suggests confidence, growth, or positive impact on this stock.
- "Bearish": If the news suggests decline, risks, or negative impact on this stock.
- "Neutral": If the news is ambiguous or does not convey strong sentiment.
Finally, **return only** the final result in valid JSON format, with the structure:
{{
  "ticker": "{ticker}",
  "sentiment": "Bullish" | "Bearish" | "Neutral",
  "sentiment_reasoning": "Provide a brief explanation of the sentiment analysis."
}}"""
    
    try:
        # Using the exact same generation code pattern as in Kaggle
        # Tokenize the prompt
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Move all tensors inside inputs to the correct device
        device_to_use = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = {key: value.to(device_to_use) for key, value in inputs.items()}
        
        # Text generation
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=120,
                early_stopping=False,
                do_sample=False,
                num_return_sequences=1,
                num_beams=1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                return_dict_in_generate=True
            )
        
        # Extract only the newly generated part
        generated_tokens = output.sequences[:, inputs["input_ids"].shape[-1]:]
        generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True).strip()
        
        print("Generated Text:\n", generated_text)
        
        # Use the exact same JSON extraction method as in Kaggle
        try:
            json_match = re.search(r'\{.*?\}', generated_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed_json = json.loads(json_str)
                return jsonify({"success": True, "response": parsed_json})
            else:
                return jsonify({"success": True, "response": generated_text})
        except Exception as json_error:
            print(f"Error parsing JSON: {str(json_error)}")
            return jsonify({"success": True, "response": generated_text})
        
    except Exception as e:
        error_message = str(e)
        print(f"Error during generation: {error_message}")
        return jsonify({"success": False, "error": error_message}), 500

if __name__ == '__main__':
    app.run(debug=True)