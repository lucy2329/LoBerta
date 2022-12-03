# LoBerta
Tweaking BERT to work better for logical and propositional reasoning tasks

## Experiments conducted with the Dataset
1. Evaluating BERT with the dataset to derive a baseline performance.
2. Evaluating BERT with changes to the attention mask.
3. Evaluating RoBERTa + ULMFiT over the dataset.
4. Evaluating GPT-3 with appropriate prompts over the dataset.

## Running Experiments
### Evaluating models 1 through 3
Appropriate notebooks have been provided for running the experiments 1 through 3.

### Evaluating GPT-3
In order to evaluate GPT-3, the first step is to fine-tune a model by submitting a request using OpenAI's CLI. A sample JSONL file (for both downsampled and the entire dataset) can be found in the assets folder within the repository. The steps for the fine-tuning is made available [here](https://beta.openai.com/docs/guides/fine-tuning).