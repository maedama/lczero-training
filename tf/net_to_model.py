#!/usr/bin/env python3
import argparse
import os
import yaml
import tfprocess
import tensorflow as tf

argparser = argparse.ArgumentParser(description='Convert net to model.')
argparser.add_argument('net',
                       type=str,
                       help='Net file to be converted to a model checkpoint.')
argparser.add_argument('--start',
                       type=int,
                       default=0,
                       help='Offset to set global_step to.')
argparser.add_argument('--cfg',
                       type=argparse.FileType('r'),
                       help='yaml configuration with training parameters')
argparser.add_argument('-e',
                       '--ignore-errors',
                       action='store_true',
                       help='Ignore missing and wrong sized values.')

argparser.add_argument('-o','--output',
                       type=str,
                       help='Ignore missing and wrong sized values.')

args = argparser.parse_args()
cfg = yaml.safe_load(args.cfg.read())
print(yaml.dump(cfg, default_flow_style=False))
START_FROM = args.start

tfp = tfprocess.TFProcess(cfg)
tfp.init_net()
tfp.model(tf.zeros((2, 112, 8, 8)))
tfp.replace_weights(args.net, ignore_errors=True)
tfp.global_step.assign(START_FROM)

tfp.model.save(args.output)
