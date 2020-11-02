# Pandora
Explainability in a Question Answering System.

Highly accurate and complex Question Answering models such as those presented on the huggingface transformers library are seen as uncomprehensible black boxes. Pandora deals with these black boxes and increases their interpretability by use of explainx.ai tools.

(*Possible future implementation*: visualization of reasoning paths and graph derivation)


## Available Datasets:
- [HotpotQA](https://hotpotqa.github.io/): QA Dataset with supporting facts to enable explainability. *(So far the most helpful)* For more details on HotpotQA, refer to their [EMNLP 2018 paper](https://arxiv.org/pdf/1809.09600.pdf).
- [NewsQA Dataset](https://www.microsoft.com/en-us/research/project/newsqa-dataset/): Crowd-sourced machine reading comprehension dataset of 120K Q&A pairs. This is more challenging than other datasets since its purpose is to help build algorithms that require human-level reasoning. It can still be used for testing for comparison purposes.
- [THE-QA](https://www.ijcai.org/Proceedings/2019/0916.pdf): (Only found the paper and not the actual dataset :cold_sweat:)
- [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/): presented by Stanford.
- [Natural Questions (NQ)](https://ai.googleblog.com/2019/01/natural-questions-new-corpus-and.html): presented by Google. 16 000 examples have answers by 5 different annotators which can be useful for evaluation and possibly for xAI.
- [Question Answering in Context (QuAC)](https://quac.ai/): information-seeking QA dialogs.
- [Other datasets for QA](https://analyticsindiamag.com/10-question-answering-datasets-to-build-robust-chatbot-systems/) 


## Other explainable Question Answering (xAI-QA) models:
- [Abujabal et al. 2017](https://www.aclweb.org/anthology/D17-2011.pdf) presents **QUINT**, a xAI-QA system that visualizes the complete derivation of the answer. Unsatisfactory answers return a proposal for reformulating the question.
- [Thayaparan et al. Oct 2020](https://arxiv.org/pdf/2010.13128.pdf): This approach answers multiple choice questions and computes a weighted graph of relevant facts for each candidate answer. These facts offer plausible explanations from which the best one is outputed as the final answer.
- [Asai et al 2020](https://arxiv.org/pdf/1911.10470.pdf) "*Learning to Retrieve Reasoning Paths over Wikipedia Graph for Question Answering*"

## Useful Tools:
- [explainx.ai](https://www.explainx.ai/): model agnostic library with visualization tools. 
- [huggingface](https://huggingface.co/): offers open-source libraries (Most notable: Transformers Library)

## To-dos:
- [] Literature Review of state-of-the-art tools, models and xAI-QA approaches.
- [] Implement the HotpotQA dataset together with explainx.ai tools and the huggingface transformer library.
- [] Implement xAI API (the output of explanations in a structured manner)

## Tests:
- Evaluate the model statistics on different transformer models.
- Evaluate the model statistics on different datasets when possible.
- (If explainx.ai does not achieve expected results) implement approach from literature review.




