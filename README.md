南京邮电大学 智能图像处理 多模态手势识别项目
====
这是一个做的很简单的项目，为了应付学校的一个软件设计周做的一个小项目\
用到了whisper模型来处理语音转化文字这个模块\
关于计算两个词语之间的相似度，则利用了GoogleNews-vectors-negative300.bin这个模型\
这个模型太大传不上github。\
在google上一搜就能搜到。\
还需要将whisper的tiny模型下载到本地\
这个模型也传不上来，链接在此：https://huggingface.co/openai/whisper-tiny.en/tree/main \
需要将新建一个文件夹tiny 将上述链接中的文件下到该文件夹中。\
该项目实现的流程大概是，利用mediapipe实现一个简单的手势识别，然后计算各个手指之间的角度\
初略实现识别 one two three five six thumb_up gun love fist 等手势\
然后调用录音机模块，进行录音，将录音转成英文。\
再利用词向量相似度模型，进行匹配词库中的单词。\
如果符合输出 YES。\
这是很简陋的一个项目设计，笔者很多想法也没有实现。希望这个项目能够给到你一点启发。
