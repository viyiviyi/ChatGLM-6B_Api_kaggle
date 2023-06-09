import json
import time
import asyncio
from fastapi import FastAPI
from sse_starlette.sse import EventSourceResponse
from fastapi.responses import JSONResponse
from typing import List, Optional
import uvicorn
import argparse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
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
    stream: Optional[bool] = False

class ChatCompletion(BaseModel):
    message: Message    

class ChatResponse(BaseModel):
    choices: List[ChatCompletion]

def load_model():
    print('load_model')
    global tokenizer 
    global model
    tokenizer = AutoTokenizer.from_pretrained(modelName, trust_remote_code=True)   
    model = AutoModelForSeq2SeqLM.from_pretrained(modelName, trust_remote_code=True,device_map='auto').half() 
    model = model.eval()

MAX_TURNS = 20
MAX_BOXES = MAX_TURNS * 2

async def predict(input, max_length=None, top_p=None, temperature=None, history=None, stream=False):
    if not model:
        if stream:
            for i in range(10):
                yield f'测试：这是测试内容 {i+1}/10。\n', []
                await asyncio.sleep(0.2)
        else:
            yield '测试：这是测试内容',[]
        return
    if history is None:
        history = []
    if stream:
        # 以流的形式响应数据
        old_response_len = 0
        next_text = ''
        for response, history in model.stream_chat(tokenizer, input, history, max_length=max_length, top_p=top_p, temperature=temperature):
            if len(response) == old_response_len:
                continue
            next_text = response[old_response_len:]
            old_response_len = len(response)
            yield next_text, history
            await asyncio.sleep(0.2)
    else:
        # 一次性响应所有数据
        response, history = model.chat(tokenizer, input, history, max_length=max_length, top_p=top_p, temperature=temperature)
        yield response, history


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

async def event_stream(speak, max_tokens, top_p, temperature, history):
    async for response, _ in predict(speak, max_tokens, top_p, temperature, history, stream=True):
        yield {
            "data": json.dumps({'choices': [{'delta': {'role': 'assistant', 'content': response}}],'created':int(time.time()),'object':'chat.completion.chunk'})
        }
    yield {
            "data": json.dumps({'choices': [{'delta': {},"finish_reason":"stop"}],'created':int(time.time()),'object':'chat.completion.chunk'})
        }
    yield {
            "data": "[DONE]"
        }

@app.post('/v1/chat/completions')  
async def chat_component(data:ChatData):
    try:
        messages = data.messages
        max_tokens = data.max_tokens
        top_p = data.top_p
        temperature = data.temperature
        user = data.user
        n = data.n
        stream = data.stream
        history = convert_to_tuples(messages)
        # 在这里执行聊天逻辑，返回聊天结果  
        speak = ''
        if len(messages) > 0 and (messages[-1].role == 'user' or messages[-1].role == 'system'):
            speak = messages[-1].content
        if stream:
            # 以 SSE 协议响应数据
            generate = event_stream(speak, max_tokens, top_p, temperature, history)
            return EventSourceResponse(generate, media_type="text/event-stream")
        else:
            # 一次性响应所有数据
            async for response, _ in predict(speak, max_tokens, top_p, temperature, history):
                return JSONResponse(status_code=200, content={'choices': [{'message':{'role':'','content':response}}]})
        
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
async def create_item(item:Item):
    async for msg, _ in predict(input=item.msg):
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

