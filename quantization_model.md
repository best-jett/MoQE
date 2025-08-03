In this project, large language models that employ AWQ, Smoothquant, GPTQ, k-quants, and imatrix quantization methods were selected as experts, and Qwen0.6B, Qwen1.7B, llama3B, and Qwen4B were chosen as the experimental subjects. The datasets in the paper are also based on open-source datasets (for details, see dataset.md). Below are the parameter links of some of the quantization experts selected for this experiment for your reference.
| model          | GitHub                            |
| -------------- | --------------------------------- |
| **Qwen3-0.6B** | <https://github.com/QwenLM/Qwen3> |
| **Qwen3-1.7B** | <https://github.com/QwenLM/Qwen3> |
| **Qwen3-4B**   | <https://github.com/QwenLM/Qwen3> |
| quantization method           | project address                              |
| ----------------------------- | -------------------------------------------- |
| **GPTQ**                      | <https://github.com/AutoGPTQ/AutoGPTQ>       |
| **SmoothQuant**               | <https://github.com/mit-han-lab/smoothquant> |
| **K-quants / GGUF / Imatrix** | <https://github.com/ggerganov/llama.cpp>     |


| quantization model  | GitHub                                      | 
| ------------------- | ------------------------------------------- |
| **Qwen3-0.6B-GGUF** | <https://github.com/QwenLM/Qwen3-0.6B-GGUF> | 
| **Qwen3-1.7B-GGUF** | <https://github.com/QwenLM/Qwen3-1.7B-GGUF> | 
| **Qwen3-4B-GGUF**   | <https://github.com/QwenLM/Qwen3-4B-GGUF>   | 

Some model parameters are not open-sourced and cannot be downloaded from the internet. Therefore, one needs to quantize the corresponding models to replicate the experiments in this paper. Due to the excessive amount of data required from quantization experts and the large size of the quantization expert's parameter data, the non-open-sourced model parameters will not be provided in this project.






本项目中的选择了使用AWQ,Smoothquant,GPTQ，k-quants，imatrix量化方法的大语言模型作为专家，选择Qwen0.6B,Qwen1.7B,llama3B以及Qwen4B作为实验对象。
论文中的数据集也是基于开源的数据集（详细见dataset.md）
下面附上本次实验选择的部分量化专家的参数链接以供参考。
| 模型规模        | GitHub 地址                       |
| -------------- | --------------------------------- |
| **Qwen3-0.6B** | <https://github.com/QwenLM/Qwen3> |
| **Qwen3-1.7B** | <https://github.com/QwenLM/Qwen3> |
| **Qwen3-4B**   | <https://github.com/QwenLM/Qwen3> |


| 方法                            | 项目地址                                    |
| ----------------------------- | -------------------------------------------- |
| **GPTQ**                      | <https://github.com/AutoGPTQ/AutoGPTQ>       |
| **SmoothQuant**               | <https://github.com/mit-han-lab/smoothquant> |
| **K-quants / GGUF / Imatrix** | <https://github.com/ggerganov/llama.cpp>     |

| 规模                  | GitHub 仓库（可一键 clone / 下载）                   | 备注                                     |
| ------------------- | ------------------------------------------- | -------------------------------------- |
| **Qwen3-0.6B-GGUF** | <https://github.com/QwenLM/Qwen3-0.6B-GGUF> | 已内置 Q4\_K\_M / Q5\_K\_M 等 Imatrix 优化量化 |
| **Qwen3-1.7B-GGUF** | <https://github.com/QwenLM/Qwen3-1.7B-GGUF> | 同上，支持 Imatrix                          |
| **Qwen3-4B-GGUF**   | <https://github.com/QwenLM/Qwen3-4B-GGUF>   | 同上，支持 Imatrix                          |


部分模型的参数并没有开源无法从网络上下载，需要自己去量化相应的模型去实现本论文中的实验。
本论文因为需要的量化专家的数据过多且量化专家的参数数据过大所以在本项目中不会提供未开源的模型参数。
