### 配置数据集
可以参考这个配置来运行。
- wikitext
```
"dataset_config": {
    "source": "huggingface",
    "name": "wikitext",
    "config": "wikitext-2-raw-v1",
    "split": "test"
}
```

- GSM8K
```
"dataset_config": {
    "source": "huggingface",
    "name": "gsm8k",
    "config": "main",
    "split": "test",
    "text_field": "question",
    "prompt_template": "Math problem: {text}\nSolution:"
}
```

- HellaSwag
```
"dataset_config": {
    "source": "huggingface",
    "name": "hellaswag",
    "split": "validation",
    "text_field": "ctx",
    "prompt_template": "Context: {text}\nMost plausible continuation:"
}
```

- ARC-Challenge
```
"dataset_config": {
    "source": "huggingface",
    "name": "ai2_arc",
    "config": "ARC-Challenge",
    "split": "test",
    "text_field": "question",
    "prompt_template": "Science question: {text}\nChoices: {choices}\nAnswer:"
}
```

- HumanEval
```
"dataset_config": {
    "source": "huggingface",
    "name": "openai_humaneval",
    "split": "test",
    "text_field": "prompt",
    "prompt_template": "Complete the Python code:\n{text}"
}
```

- 本地文件
```
"dataset_config": {
    "source": "local",
    "local_path": "/path/to/your/data.txt"
}
```