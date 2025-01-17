import pathlib
import signal

import torch
from torch import nn

import pyshoz

TOPIC = "cnn"
LLOCATOR = "udp/224.0.0.123:7447#iface=lo"
POOL_SIZE = 8 << 20;  # 8 MiB
MSG_QUEUE_CAP_EXP = 7

def main():
    signal.signal(signal.SIGINT, lambda _sig, _frame: print("Stopped"))

    weights_path = pathlib.Path(__file__).parent.resolve().joinpath("cifar_net.pth")

    model = Net(3, 10)
    model.load_state_dict(torch.load(weights_path, weights_only=True))
    model.cuda()
    model.eval()

    def handler(input):
        labels = [
            "airplanes",
            "cars",
            "birds",
            "cats",
            "deer",
            "dogs",
            "frogs",
            "horses",
            "ships",
            "trucks",
        ]
        output = model(torch.unsqueeze(input, 0))
        pred = torch.argmax(output).item()
        print(f"Prediction: {labels[pred]}")

    session = pyshoz.ZenohSession(LLOCATOR)
    _subscriber = pyshoz.Subscriber(session, TOPIC, POOL_SIZE, MSG_QUEUE_CAP_EXP, handler)

    print("Ctrl+C to leave")
    signal.pause()

class Conv2dBlock(nn.Module):
    def __init__(self, inc, outc, ksize, **kwargs):
        super(Conv2dBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(inc, outc, ksize, **kwargs),
            nn.BatchNorm2d(outc),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.block(x)
    
class Net(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.out_dim = out_dim
        self.in_dim = in_dim

        self.conv_block1 = Conv2dBlock(self.in_dim, 32, 3)
        self.conv_block2 = Conv2dBlock(32, 64, 3)

        self.fc = nn.Sequential(
            nn.Linear(28*28*64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, out_dim), 
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = torch.flatten(x, start_dim=1)  # flatten all execpt batch dim
        x = self.fc(x)
        return x

if __name__ == "__main__":
    main()