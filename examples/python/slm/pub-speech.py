#!/usr/bin/env python3
import sys
import time

from transformers import AutoTokenizer

import pyshoz

TOPIC = "slm"
LLOCATOR = "udp/224.0.0.123:7447#iface=lo"
POOL_SIZE = 2 * 1024 * 1024  # 2 MiB

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <speech file>")
        exit(1)

    model_name = "microsoft/Phi-3.5-mini-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    publisher = pyshoz.Publisher(TOPIC, LLOCATOR, POOL_SIZE)
    sents = read_sentences(sys.argv[1])

    print("Publisher is ready")
    print("Start sending...")

    beg_time = time.time()
    for i in range(0, len(sents), 2):
        if i + 1 < len(sents):
            content = f"{sents[i]} {sents[i + 1]}"
        else:
            content = sents[i]

        input_tokens = tokenizer(content, return_tensors="pt")["input_ids"]
        speaking_time = input_tokens.shape[1] * 0.5
        buffer_time = 1
        time.sleep(speaking_time + buffer_time)

        msg_tokens = publisher.malloc(2, input_tokens.shape, (0, 64, 1))
        publisher.copy_tensor(msg_tokens, input_tokens)
        publisher.put(msg_tokens)

    print("Finished! Note that the subscribers are probably still translating.")
    print(f"Begin time: {beg_time}")

def read_sentences(path):
    with open(path) as f:
        sentences = [s.replace("\n", "") for s in f]
    return sentences

if __name__ == "__main__":
    main()