from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from dataLoader import TestSet, readData, SOS_token, EOS_token, number2letter
from model import *


# compute BLEU-4 score
# TODO: Write down the detail of bleu-4
def compute_bleu(output, reference):
    cc = SmoothingFunction()
    if len(reference) == 3:
        weights = (0.33, 0.33, 0.33)
    else:
        weights = (0.25, 0.25, 0.25, 0.25)
    return sentence_bleu([reference], output, weights=weights, smoothing_function=cc.method1)


################################
# Example inputs of compute_bleu
################################
# The target word
# reference = 'accessed'
# The word generated by your model
# output = 'access'


"""============================================================================
example input of Gaussian_score

words = [['consult', 'consults', 'consulting', 'consulted'],
['plead', 'pleads', 'pleading', 'pleaded'],
['explain', 'explains', 'explaining', 'explained'],
['amuse', 'amuses', 'amusing', 'amused'], ....]

the order should be : simple present, third person, present progressive, past
============================================================================"""
def gaussian_score(words):
    words_list = []
    score = 0
    yourpath = 'data/train.txt'  # should be your directory of train.txt
    with open(yourpath, 'r') as fp:
        for line in fp:
            word = line.split(' ')
            word[3] = word[3].strip('\n')
            words_list.extend([word])
        for t in words:
            for i in words_list:
                if t == i:
                    score += 1
    return score / len(words)


def evaluate(encoder: EncoderRNN, decoder: DecoderRNN, dataset: TestSet):
    encoder.eval(), decoder.eval()
    pairs = dataset.get_pairs()

    candidate = []
    reference = []
    for pair in pairs:
        input_seq, input_type, output_seq, output_type = pair
        input_seq = input_seq.to(device)
        input_type = input_type.to(device)
        output_type = output_type.to(device)

        # encode
        encoder_hidden = encoder.initHidden(input_type)
        encoder_cell = encoder.initHidden(input_type)
        hidden_dis, cell_dis = encoder(input_seq, encoder_hidden, encoder_cell)

        # decode
        decoder_input = torch.tensor([[SOS_token]], device=device)
        decoder_hidden = decoder.initHidden(reparameter(*hidden_dis), output_type)
        decoder_cell = decoder.initHidden(reparameter(*cell_dis), output_type)
        word = ''
        while decoder_input.item() != EOS_token:
            decoder_output, decoder_hidden, decoder_cell = decoder(
                decoder_input, decoder_hidden, decoder_cell)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()
            # Collect the output letters
            word += number2letter(decoder_input.item())

        candidate.append(word[:len(word) - 1])
        reference.append(output_seq)

    # Print the bleu score
    bleu_score = 0
    for i in range(len(candidate)):
        bleu_score += compute_bleu(candidate[i], reference[i])
    # print(candidate)
    # print(f'Average BLEU-4 score : {bleu_score / len(candidate)}')

    return candidate, bleu_score / len(candidate)


def sample_gaussian(dim):
    return torch.randn_like(torch.zeros(dim))


def evaluate_gaussian(decoder: DecoderRNN):
    decoder.eval()
    words = []

    for i in range(100):
        hidden_esp = sample_gaussian(decoder.hidden_size)
        cell_esp = sample_gaussian(decoder.hidden_size)
        tense_list = []

        for type in range(4):
            hidden_type = torch.zeros(4).type(dtype=torch.float)
            cell_type = torch.zeros(4).type(dtype=torch.float)
            hidden_type[type] = 1.
            cell_type[type] = 1.

            hidden = torch.cat([hidden_esp, hidden_type])
            cell = torch.cat([cell_esp, cell_type])
            hidden = hidden.to(device).view(1, 1, -1)
            cell = cell.to(device).view(1, 1, -1)
            decoder_input = torch.tensor([[SOS_token]], device=device)

            word = ''
            while decoder_input.item() != EOS_token:
                decoder_output, hidden, cell = decoder(decoder_input, hidden, cell)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()
                word += number2letter(decoder_input.item())

            tense_list.append(word[:len(word) - 1])

        words.append(tense_list)

    return words, gaussian_score(words)


if __name__ == "__main__":
    dataset = TestSet(readData('data', 'test'))
    encoder = torch.load('encoder.pth').to(device)
    decoder = torch.load('decoder.pth').to(device)

    generated_word, gau_score = evaluate_gaussian(decoder)
    # print(generated_word)
    for words in generated_word:
        print(words)
    print(f'Gaussian score: {gau_score}')

    best_bleu = 0
    best_candidate = None

    for i in range(100):
        candidate, bleu = evaluate(encoder, decoder, dataset)
        # best_bleu += bleu
        if bleu > best_bleu:
            best_bleu = bleu
            best_candidate = candidate

    # print(best_candidate)
    for i in range(len(best_candidate)):
        print(f'Input: {dataset.word_set[i][0]:<20}\t'
              f'Target: {dataset.word_set[i][1]:<20}\t'
              f'Prediction: {best_candidate[i]:<20}')
    print(f'BLEU-4 score: {best_bleu}')
