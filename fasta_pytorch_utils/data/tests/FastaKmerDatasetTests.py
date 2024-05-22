import queue
import os
import fasta_utils
from fasta_utils.tokenizers.kmertokenizer import KmerTokenizer
from fasta_utils.vocab import Vocab
from fasta_utils import FastaFileReader
from fasta_pytorch_utils.data import FastaFileQueueDataset, PreprocessEmbedding, StreamingEmbedding
from queue import Queue
import numpy as np
import unittest

import time

script_directory = os.path.dirname(os.path.realpath(__file__))
# root_directory = os.path.abspath(os.path.join(script_directory, "../../../"))

class test(unittest.TestCase):
    def test_example(self):
        kmer_size = 7
        stride = 3
        test_queue = queue.Queue()
        tokenizer = KmerTokenizer(kmer_size=kmer_size, stride=stride)
        vocabulary = Vocab.load(os.path.join(script_directory, "data/7mer-s3-202405182143.pickle"))
        streaming = StreamingEmbedding(vocabulary, tokenizer, window_size=7)
        preprocessing = PreprocessEmbedding(vocabulary, tokenizer, window_size=7)
        dataset = FastaFileQueueDataset(test_queue, embedding_strategy=streaming)
        sequence_file = os.path.join(script_directory, "data/Zm-B97-REFERENCE-NAM-1.0.fa.gz")
        test_queue.put((sequence_file, 0))

        for context, target in dataset:
            print(context, target)
            break


        # with FastaFileReader(sequence_file) as file_reader:
        #     for header, sequence in file_reader.read_at_index(0):
        #         break


        # def sampel_sequence():
        #     counter = 0
        #     for sample in dataset:
        #         counter += 1
        #         if counter == 100000:
        #             break
        # subsequence = sequence[:1000000]
        # start = time.perf_counter()
        # tokens = tokenizer.tokenize_list(subsequence)
        # end = time.perf_counter()
        # print(f"done {end - start:0.44f} seconds")
        #
        # print(f"number of tokens {len(tokens)}")
        # # tokens = tokens[:100000]



        # start = time.perf_counter()
        # # embedded = [vocabulary[token] for token in tokens]
        # windows = list(dataset)
        # end = time.perf_counter()
        # print(f"done {end - start:0.44f} seconds")
        # print(f"number of windows {len(windows)}")
        #
        # start = time.perf_counter()
        # # embedded = [vocabulary[token] for token in tokens]
        # windows = list(dataset.get_windows_alt(subsequence))
        # end = time.perf_counter()
        # print(f"done {end - start:0.44f} seconds")
        # print(f"number of windows {len(windows)}")
        #
        # embedded = [vocabulary[token] for token in tokenizer.tokenize(subsequence)]
        # start = time.perf_counter()
        # # embedded = [vocabulary[token] for token in tokens]
        # windows = list(dataset.get_windows_embedded(embedded))
        # end = time.perf_counter()
        # print(f"done {end - start:0.44f} seconds")
        # print(f"number of windows {len(windows)}")

        # with FastaFileReader(sequence_file) as fasta_reader:
        #     for header, sequence in fasta_reader.read_at_index(0):
        #         print(header)
        #         break
            # for item in tokenizer.tokenize(sequence):
            #     print(item)

if __name__ == "__main__":
    unittest.main()