from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from peft import PeftModel
from pydantic import BaseModel
import sys
from io import StringIO
import nest_asyncio
from pyngrok import ngrok
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import asyncio

# --- CONFIGURATION & NGROK ---
# Ensure you have your Gemini API Key set up if you use 'client'
# from google import generativeai as genai
# genai.configure(api_key="YOUR_GEMINI_KEY")
# client = genai.GenerativeModel('gemini-pro')

auth_token = "3A1KkcM4SMNNpF8Czsdf4Nhc4ez_39MvoiQKjZXxHnoWFaKjp"
ngrok.set_auth_token(auth_token)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MODEL LOADING ---
BASE_MODEL = "Salesforce/codegen-2B-multi"
LORA_WEIGHTS = "/content" # Path to your adapter folder

print("Loading Tokenizer and Model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

if torch.cuda.is_available():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb_config, device_map="auto")
else:
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="cpu")

print("Attaching Adapters...")
model = PeftModel.from_pretrained(base_model, LORA_WEIGHTS)
model.eval()
print("Backend Ready!")

# --- SCHEMAS ---
class PromptRequest(BaseModel):
    prompt: str

class CodeRequest(BaseModel):
    code: str

# --- ENDPOINTS ---

# --- UPDATED ENDPOINTS ---

@app.post("/generate")
async def generate_code(req: PromptRequest):
    # Strictly following the training format
    input_text = f"### prompt: {req.prompt}\n\n### completion:"
    
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=500, 
            temperature=0.1, 
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extracting only the completion part
    if "### completion:" in decoded:
        code_only = decoded.split("### completion:")[1].strip()
    else:
        code_only = decoded.replace(input_text, "").strip()
        
    return {"code": code_only}

@app.post("/summarize")
async def summarize_code(req: CodeRequest):
    """
    Uses the model to summarize code by wrapping the code in the prompt template.
    """
    # Combining the summarization instruction with the source code inside the prompt block
    instruction = f"Summarize the following code and explain what it does: {req.code}"
    input_text = f"### prompt: {instruction}\n\n### completion:"
    
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=256, 
            temperature=0.4, # Slightly higher temp for more natural language explanation
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extracting only the summary part
    if "### completion:" in decoded:
        summary = decoded.split("### completion:")[1].strip()
    else:
        summary = decoded.replace(input_text, "").strip()
        
    return {"summary": summary}

@app.post("/run")
async def run_code(req: CodeRequest):
    old_stdout = sys.stdout
    redirected_output = sys.stdout = StringIO()
    try:
        exec(req.code, {})
        sys.stdout = old_stdout
        return {"output": redirected_output.getvalue() or "Executed successfully."}
    except Exception as e:
        sys.stdout = old_stdout
        return {"output": str(e)}

# --- EXECUTION ---
public_url = ngrok.connect(8000)
print(f"--- PUBLIC CLOUD URL: {public_url} ---")

nest_asyncio.apply()
config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
server = uvicorn.Server(config)
loop = asyncio.get_event_loop()
loop.create_task(server.serve())