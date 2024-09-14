#!/usr/bin/env python3
import signal
import sys
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import pyshoz

TOPIC = "slm"
LLOCATOR = "udp/224.0.0.123:7447#iface=lo"
POOL_SIZE = 2 * 1024 * 1024  # 2 MiB

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <random seed>")
        exit(1)

    signal.signal(signal.SIGINT, lambda sig, frame: print('Stopped'))

    seed = int(sys.argv[1])
    torch.random.manual_seed(seed)

    device = "cuda"
    model_name = "microsoft/Phi-3.5-mini-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device, trust_remote_code=True)

    template_prefix = """<|system|>
你是精通多國語言的翻譯官，能夠以最接近原意的方式翻譯使用者提供的文句。
現在你的任務是將英文翻譯成繁體中文，例如：Computer 翻成電腦，Artificial Intelligence 翻成人工智慧。
千萬不要產生補充說明，只要輸出翻譯結果就好。舉例：
English: John Doe went to the restaurant last night.
繁體中文: John Doe 昨晚去了那間餐廳。
<|end|>
<|user|>
English: """
    template_suffix = """<|end|>
<|assistant|>
繁體中文: """
    prefix_tokens = tokenizer(template_prefix, return_tensors="pt")["input_ids"].to(device)
    suffix_tokens = tokenizer(template_suffix, return_tensors="pt")["input_ids"].to(device)

    subscriber = pyshoz.Subscriber(TOPIC, LLOCATOR, POOL_SIZE)
    timepoints = []
    translation = []

    def handler(tokens):
        input_tokens = torch.cat((prefix_tokens, tokens, suffix_tokens), dim=1)
        prompt_token_num = input_tokens.shape[1]
        outputs = model.generate(
            input_ids=input_tokens,
            max_new_tokens=128,
            do_sample=True,
        )
        decoded_output = tokenizer.decode(token_ids=outputs[0][prompt_token_num:], skip_special_tokens=True)
        translation.append(decoded_output)
        timepoints.append(time.time())
        print(len(translation))
    subscriber.sub(handler)

    print("Subscriber is ready")
    print("Ctrl+C to leave")
    signal.pause()

    print('The translation:')
    for sent in translation:
        print(sent)

    # dump timepoints
    filename = f"sub-{seed}.txt"
    with open(filename, "w") as f:
        f.write("\n".join([str(point) for point in timepoints]))

if __name__ == "__main__":
    main()