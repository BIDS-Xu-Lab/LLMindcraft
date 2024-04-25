
# LLMindCraft 

[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC_BY--NC--ND_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)


Shaping Language Models with Cognitive Insights

LLMindCraft is licensed under [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/).

## Docker environment
```bash
docker pull tothemoon/llm
```
This image packages all environments of LLMindCraft. 

## Fine-tuning in Docker environment
For **single node**:
```bash
docker run --gpus all \
    -d --rm \
    --name llm \
    [-v host_path:container_path] \
    [-w workdir] \
    --entrypoint "/bin/bash -c" \
    tothemoon/llm \
    --cmd "sleep infinity"
```
while for **multiple nodes**:
```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    --privileged \
    --network host \
    [--env env_variable=value] \
    -d --rm \
    --name llm \
    [-v host_path:container_path] \
    [-v ssh_pub_key:/root/.ssh/authorized_keys] \
    [-w workdir] \
    tothemoon/llm \
    --sshd_port [any_port] --cmd "sleep infinity"
```

You can also enter the container by
```bash
docker exec -it llm /bin/bash
```

## Create New Dataset

Create a new data class in `preprocess.py`, like:

Your dataset should be created in the following format:

```python
class MedMCQA(InstructionDataset):
    dataset = "MedMCQA"
    task_type = "classification"
    choices = ["A", "B", "C", "D"]
    prompt = """Given a medical context and a multiple choice question related to it, select the correct answer from the four options.
Question: {text}
Options: {options}.
Please answer with A, B, C, or D only.
Answer:
"""

    def fetch_data(self, datum):
        return {
            "text": datum["question"], "options": ', '.join([op+': '+datum[k] for k, op in zip(['opa', 'opb', 'opc', 'opd'], self.choices)]),
            "answer": self.choices[datum["cop"]-1],
        }
```

In this format:

- `dataset`: The dataset name
- `task_type`: Your task type, should be `classification` or `abstractivesummarization` (TODO: More task types)
- `prompt`: The prompt of the task, which should be later used to be filled with the real data

For **Classification** tasks, additional keys should be defined:

- `choices`: Set of labels

> `fetch_data` is the interface for fetching the required features from raw data

And you should also append your class in the dictionary:

```python
DATASETS = {
    "MedMCQA": MedMCQA,
}

```

Finally, you can build and upload the dataset by:
```bash
bash preprocess.sh
```
Note that the parameters in the `preprocess.sh` should be changed accordingly. For evaluation datasets, `-for_eval` should be used, while for instruction tuning datasets, it should be omitted.
