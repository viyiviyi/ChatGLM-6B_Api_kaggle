
from fastapi import FastAPI
import uvicorn
from transformers import AutoModel, AutoTokenizer,AutoModelForSeq2SeqLM
import torch

torch.cuda.empty_cache()
modelName = '''+modelPath+'''
tokenizer = AutoTokenizer.from_pretrained(modelName, trust_remote_code=True) 
model = None
model = AutoModelForSeq2SeqLM.from_pretrained(modelName, trust_remote_code=True,device_map='auto').half() 
model = model.eval()

MAX_TURNS = 20
MAX_BOXES = MAX_TURNS * 2

def predict(input, max_length, top_p, temperature, history=None):
    if not model:
        return '测试：这是测试内容'
    if history is None:
        history = []
    response, history = model.chat(tokenizer, input, history, max_length=max_length, top_p=top_p, temperature=temperature)
    return (response, history)

app = FastAPI()

def convert_to_tuples(data):
    messages = []
    user = ''
    assistant = ''
    for item in data:
        if item['role'] == 'user':
            user = item['content']
        elif item['role'] == 'assistant':
            assistant = item['content']
        if assistant:
            messages.append((user,assistant))
            user = ''
            assistant = ''
    return messages

@app.post('/v1/chat/completions')  
def chat_component(data):
    messages = data['messages']
    max_tokens = data.get('max_tokens', 1024)
    top_p = data.get('top_p', 0.9)
    temperature = data.get('temperature', 0.5)
    user = data.get('user', 'default')
    n = data.get('n', 1)
    history = convert_to_tuples(messages)
    # 在这里执行聊天逻辑，返回聊天结果  
    speak = ''
    if len(messages) > 0 and messages[-1]['role'] == 'user':
        speak = messages[-1]['content']
    response,_ = predict(speak, max_tokens, top_p, temperature, history)
    return {'choices': [{'message':{'role':'','content':response}}]}
