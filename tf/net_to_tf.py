
#!/usr/bin/env python3
import argparse
import os
import yaml
import tfprocess
import numpy as tf
from leela_board import LeelaBoard
import numpy as np
def net_to_tf(net, output_path, cfg=None):
    cfg = yaml.safe_load(open(cfg,"r").read())
    print(yaml.dump(cfg, default_flow_style=False))
    tfp = tfprocess.TFProcess(cfg)
    tfp.init_net()

    tfp.model(tf.zeros((2, 112, 8, 8)))    
    tfp.replace_weights(net, ignore_errors=True)
    fens=[
        "rn1qkbnr/ppp2ppp/3p4/4p3/4P1b1/2N5/PPPPBPPP/R1BQK1NR w KQkq - 2 4", # white to play and win a piece
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2", #e4 e5
        "rn1qkbnr/ppp3pp/3p1p2/4p3/4P1B1/2N5/PPPP1PPP/R1BQK1NR w KQkq - 0 5", #white has already won a piece
        "rn1qkbnr/ppp3pp/3p4/4pp2/4P1B1/2N5/PPPP1PPP/R1BQK1NR w KQkq f6 0 5", #white has already wo a piece"        
    ]

    input=np.stack([LeelaBoard(fen=fen).lcz_features() for fen in fens],axis=0)
        
    res=tfp.model(input)
    tfp.model.save(output_path)
    print(res[2])
    print("done")

if __name__ == '__main__':
    net_to_tf(
        net="/Users/shuntaromaeda/Downloads/768x15x24h-t82-swa-7664000.pb.gz",
        output_path="/tmp/tf_model",
        cfg="/Users/shuntaromaeda/src/github.com/LeelaChessZero/lczero-training/tf/configs/t80.yaml",
    )
