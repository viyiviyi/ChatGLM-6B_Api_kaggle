# ChatGLM-6B_Api_kaggle
在kaggle部署chatglm API，和ChatGPT api使用相同的调用方式

- 接口的调用方式保持和ChatGPT一致，只是有些参数无需了，便于ChatGPT项目可以直接替换地址使用。
- 同时也复制了这个项目的接口部分代码，应该有人在用了，这里我希望可以兼容[https://forum.koishi.xyz/t/topic/1075](https://forum.koishi.xyz/t/topic/1075)
- 在kaggle部署要使用P100这个单卡gpu，否则不能用，我并不知道怎么让chatglm支持多卡推理

- 可以copy这个笔记本体验 [ChatGLM-6B_Api_kaggle](https://www.kaggle.com/code/viyiviyi/chatglm-api)

- api_launch_controllable.py 这个文件使用了一个不对输入chatglm的内容做预处理的方法，暂时用于测试如何让chatglm输出用户需要的内容。
