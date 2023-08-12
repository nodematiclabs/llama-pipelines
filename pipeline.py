import kfp.dsl as dsl
from kfp import compiler

@dsl.component(
    base_image='huggingface/transformers-pytorch-gpu:4.29.2',
    packages_to_install=[
        'git+https://github.com/huggingface/accelerate.git',
        'huggingface-hub',
        'sentencepiece',
        'tokenizers>=0.13.3',
        'torch',
        'git+https://github.com/huggingface/transformers.git'
    ]
)
def ask_a_llama(prompt: str) -> str:
    import torch
    import transformers

    from huggingface_hub import login
    from transformers import AutoTokenizer

    login(token="YOUR TOKEN HERE")

    model = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model)

    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.float32,
        device_map="auto"
    )

    sequences = pipeline(
        prompt,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=256
    )

    return sequences[0]['generated_text']


@dsl.pipeline(
    name="llama-example"
)
def llama_example(prompt: str):
    ask_a_llama_task = ask_a_llama(
        prompt=prompt
    )
    ask_a_llama_task.set_cpu_request("8")
    ask_a_llama_task.set_cpu_limit("8")
    ask_a_llama_task.set_memory_request("16Gi")
    ask_a_llama_task.set_memory_limit("16Gi")
    ask_a_llama_task.set_accelerator_limit("1")
    ask_a_llama_task.set_accelerator_type("NVIDIA_TESLA_T4")  # A100s,...

compiler.Compiler().compile(llama_example, 'pipeline.yaml')