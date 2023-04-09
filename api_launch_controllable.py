from fastapi import FastAPI
from fastapi.responses import JSONResponse
from typing import List, Optional
import uvicorn
import argparse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModel, AutoTokenizer,AutoModelForSeq2SeqLM
from huggingface_hub import hf_hub_download
import torch

torch.cuda.empty_cache()

modelName = "THUDM/chatglm-6b"
tokenizer = None
model = None

class Item(BaseModel):  
    msg: str  

class Message(BaseModel):
    role: str
    content: str
class ChatData(BaseModel):
    messages: List[Message]
    max_tokens: Optional[int] = 1024
    top_p: Optional[float] = 0.9
    temperature: Optional[float] = 0.5
    user: Optional[str] = 'user'
    n: Optional[int] = 1

class ChatCompletion(BaseModel):
    message: Message    

class ChatResponse(BaseModel):
    choices: List[ChatCompletion]

def insert_content_to_file(file_path, line_search:str, content):
    # 读取文件内容
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # 插入内容到指定行
    line_num = 0;
    for line in lines:
        line_num=line_num+1;
        if line_search in line:
            break
    print('insert:',line_num)
    lines.insert(line_num - 1, content + '\n')
    
    # 将修改后的内容写回文件
    with open(file_path, 'w') as f:
        f.writelines(lines)

new_func = '''
    @torch.no_grad()
    def call_chat(self, tokenizer, prompt: str, max_length: int = 2048, num_beams=1,
             do_sample=True, top_p=0.7, temperature=0.95, logits_processor=None, **kwargs):
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
        logits_processor.append(InvalidScoreLogitsProcessor())
        gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "do_sample": do_sample, "top_p": top_p,
                      "temperature": temperature, "logits_processor": logits_processor, **kwargs}
        inputs = tokenizer([prompt], return_tensors="pt")
        inputs = inputs.to(self.device)
        outputs = self.generate(**inputs, **gen_kwargs)
        # print(outputs)
        # outputs = outputs.tolist()[0][0:]
        outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):]
        response = tokenizer.decode(outputs)
        return response

'''

def load_model():
    print('load_model')
    global tokenizer 
    global model
    # 单独下载这个文件并往文件里插入一个不对输入和输出内容做任何预处理的方法
    file_path = hf_hub_download(repo_id=modelPath, filename='modeling_chatglm.py')
    if file_path:
        insert_content_to_file(file_path, '@torch.no_grad()',new_func) 
    
    tokenizer = AutoTokenizer.from_pretrained(modelName, trust_remote_code=True)   
    model = AutoModelForSeq2SeqLM.from_pretrained(modelName, trust_remote_code=True,device_map='auto').half() 
    model = model.eval()

MAX_TURNS = 20
MAX_BOXES = MAX_TURNS * 2

def predict(input, max_length, top_p, temperature):
    if not model:
        return '测试：这是测试内容'
    response = model.call_chat(tokenizer, input, max_length=max_length, top_p=top_p, temperature=temperature)
    return response

app = FastAPI()

def convert_to_tuples(data):
    messages = []
    user = ''
    assistant = ''
    for item in data:
        if item.role == 'user' or item.role == 'system':
            user = item.content
        elif item.role == 'assistant':
            assistant = item.content
        if assistant:
            messages.append((user,assistant))
            user = ''
            assistant = ''
    return messages

@app.post('/v1/chat/completions')  
def chat_component(data:ChatData):
    try:
        messages = data.messages
        max_tokens = data.max_tokens
        top_p = data.top_p
        temperature = data.temperature
        user = data.user
        n = data.n
        history = '\n\n'.join([item.content for item in messages]
        # 在这里执行聊天逻辑，返回聊天结果  
        speak = history
        response,_ = predict(speak, max_tokens, top_p, temperature)
        return {'choices': [{'message':{'role':'','content':response}}]}
    except Exception as e:
        return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": str(e),
                "type": "invalid_request_error",
                "param": "messages",
                "code": "error"
            }
        }
    )


@app.post("/chat")
def create_item(item:Item):
    msg = predict(input=item.msg)
    return msg

def main(port, model_name, debug,corsOrigins):
    # 在这里编写你的代码  
    global modelName
    modelName = model_name
    if not debug:
        load_model()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=corsOrigins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    uvicorn.run(app, host="127.0.0.1", port=port)
    print('server stop')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()  
    parser.add_argument("-p", "--port", type=int, default=8080, help="port number")
    parser.add_argument("-m", "--model_name", type=str, default=modelName, help="model name or model path")
    parser.add_argument("-d", "--debug", action="store_true", help="enable debug mode")
    parser.add_argument("-cors", "--cors", type=str, help="cors domains")
    args = parser.parse_args()  
    print(args)
    origins = ["*"]
    if args.cors:
        origins = args.cors.split(',')
    main(args.port, args.model_name, args.debug,origins)  

