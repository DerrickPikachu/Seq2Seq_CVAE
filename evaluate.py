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
# TODO: Survey the detail of Gaussian score
def gaussian_score(words):
    words_list = []
    score = 0
    yourpath = ''  # should be your directory of train.txt
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


# TODO: Need test
def evaluate(encoder: EncoderRNN, decoder: DecoderRNN, dataset: TestSet):
    encoder.eval(), decoder.eval()
    # dataset = TestSet(readData('data', 'test'))
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
        mean, logvar = encoder(input_seq, encoder_hidden)

        # decode
        decoder_input = torch.tensor([[SOS_token]], device=device)
        decoder_hidden = reparameter(mean, logvar)
        decoder_hidden = decoder.initHidden(decoder_hidden, output_type)
        word = ''
        while decoder_input.item() != EOS_token:
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
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
    print(candidate)
    print(f'Average BLEU-4 score : {bleu_score / len(candidate)}')


if __name__ == "__main__":
    print(compute_bleu(['access', 'play'], ['access', 'play']))
