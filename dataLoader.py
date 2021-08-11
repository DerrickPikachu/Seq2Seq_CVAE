import random

import torch


SOS_token = 0
EOS_token = 1


def readData(path: str, filename: str):
    print('Read lines...')

    lines = open(f'{path}/{filename}.txt', encoding='utf-8').\
        read().strip().split('\n')

    print(f'Get {len(lines)} lines')
    return lines


def letter2number(letter: str):
    # SOS: 0, EOS: 1
    return ord(letter[0]) - ord('a') + 2


def str2seq(string: str):
    seq = []
    for c in string:
        seq.append(letter2number(c))
    return torch.tensor(seq)


def seqToOneHot(seq: torch.tensor):
    one_hot_seq = torch.zeros(len(seq), 28)
    for i in range(len(seq)):
        index = seq[i]
        one_hot_seq[i][index] = 1
    return one_hot_seq


class TenseSet:
    def __init__(self, lines):
        self.tense_map = {
            'sp': 0,
            'tp': 1,
            'pg': 2,
            'p': 3
        }
        self.word_set = []

        for line in lines:
            self.word_set.append(line.split(' '))

    def __getitem__(self, pos):
        string = self.word_set[pos[0]][self.tense_map[pos[1]]]
        return str2seq(string)

    def get_pairs(self):
        pairs = []

        for i in range(len(self.word_set)):
            for j in range(len(self.word_set[i])):
                sequence = str2seq(self.word_set[i][j])
                types = torch.zeros(4).type(dtype=torch.float)
                types[j] = 1.
                pairs.append((
                    sequence,
                    torch.cat([sequence, torch.tensor([EOS_token])]).view(-1, 1),
                    types,
                ))

        return pairs


if __name__ == "__main__":
    lines = readData('data', 'train')
    mySet = TenseSet(lines)
    pairs = mySet.get_pairs()
    print(pairs[0])
