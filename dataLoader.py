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


class TestSet(TenseSet):
    def __init__(self, lines):
        super(TestSet, self).__init__(lines)

        # self.input_tense = [0, 0, 0, 0, 3, 0, 3, 2, 2, 2]
        # self.target_tense = [3, 2, 1, 1, 1, 2, 0, 0, 3, 1]
        self.tense_transpose = [
            [0, 3], [0, 2], [0, 1], [0, 1], [3, 1],
            [0, 2], [3, 0], [2, 0], [2, 3], [2, 1],
        ]

    def get_pairs(self):
        pairs = []

        for i in range(len(self.word_set)):
            input_sequence = str2seq(self.word_set[i][0])
            input_types = torch.zeros(4).type(dtype=torch.float)
            input_types[self.tense_transpose[i][0]] = 1.

            target_sequence = str2seq(self.word_set[i][1])
            target_types = torch.zeros(4).type(dtype=torch.float)
            target_types[self.tense_transpose[i][1]] = 1.
            pairs.append((
                input_sequence,
                input_types,
                torch.cat([target_sequence, torch.tensor([EOS_token])]).view(-1, 1),
                target_types,
            ))

        return pairs


if __name__ == "__main__":
    lines = readData('data', 'test')
    # mySet = TenseSet(lines)
    mySet = TestSet(lines)
    pairs = mySet.get_pairs()
    print(pairs)
