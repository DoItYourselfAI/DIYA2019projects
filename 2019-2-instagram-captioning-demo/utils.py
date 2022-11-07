import torch
import torch.nn as nn
import argparse
from config import Config
import torchvision.utils as tvutils
import matplotlib.pyplot as plt
import numpy as np
import os
import subprocess
import jpype
import json
from konlpy import jvm
from collections import Counter


class OktDetokenizer:
    def __init__(self, jvm_path=None, max_heap_size=1024):
        if not jpype.isJVMStarted():
            jvm.init_jvm(jvm_path, max_heap_size)
            
        oktPackage = jpype.JPackage('org.openkoreantext.processor') 
        self.processor = oktPackage.OpenKoreanTextProcessorJava()
        
    def detokenize(self, tokens):
        jTokens = jpype.java.util.ArrayList()
        
        for token in tokens:
            if type(token) == tuple:
                jTokens.add(token[0])
            elif type(token) == str:
                jTokens.add(token.split('/')[0])
                
        return self.processor.detokenize(jTokens)
    

def load_model(encoder, decoder, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=Config.device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])

