# ChatGLM-6B_Api_kaggle
在kaggle部署chatglm API，和ChatGPT api使用相同的调用方式  
支持[ChatGLM2](https://github.com/THUDM/ChatGLM2-6B)，只需要启动时传入模型名称 (THUDM/chatglm2-6b) 即可

- 接口的调用方式保持和ChatGPT一致，只是有些参数是无效的，便于ChatGPT项目可以直接替换地址使用。
- 同时也复制了这个项目的接口部分代码，应该有人在用了，这里我希望可以兼容[https://forum.koishi.xyz/t/topic/1075](https://forum.koishi.xyz/t/topic/1075)
- 在kaggle部署要使用P100这个单卡gpu，否则不能用，我并不知道怎么让chatglm支持多卡推理
- [一个简单的部署图文流程](./部署流程图文.md)
- 可以copy这个笔记本体验 [ChatGLM-6B_Api_kaggle](https://www.kaggle.com/code/viyiviyi/chatglm-api)

- 启动
```shell
git clone https://github.com/viyiviyi/ChatGLM-6B_Api_kaggle.git

cd ChatGLM-6B_Api_kaggle

pip install -r requirements.txt

python3 api_launch.py -p 8080
# ChatGLM2
python3 api_launch.py -p 8080 -m THUDM/chatglm2-6b
```

- api_launch_controllable.py 这个文件使用了一个不对输入chatglm的内容做预处理的方法，暂时用于测试如何让chatglm输出用户需要的内容。
