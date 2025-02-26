#!/usr/bin/env python3
import sys
import time

from transformers import AutoTokenizer

import pygaps

TOPIC = "slm"
LLOCATOR = "udp/224.0.0.123:7447#iface=lo"
POOL_SIZE = 2 << 20  # 2 MiB
MSG_QUEUE_CAP_EXP = 7

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <speech file>")
        exit(1)

    model_name = "microsoft/Phi-3.5-mini-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    session = pygaps.ZenohSession(LLOCATOR)
    publisher = pygaps.Publisher(session, TOPIC, POOL_SIZE, MSG_QUEUE_CAP_EXP)
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

        msg_tokens = publisher.empty(input_tokens.shape, pygaps.int64)
        msg_tokens.copy_(input_tokens)
        publisher.put(msg_tokens)

    print("Finished! Note that the subscribers are probably still translating.")
    print(f"Begin time: {beg_time}")

def read_sentences(path):
    with open(path) as f:
        sentences = [s.replace("\n", "") for s in f]
    return sentences

if __name__ == "__main__":
    main()