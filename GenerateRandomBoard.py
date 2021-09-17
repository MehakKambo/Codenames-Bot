import numpy as np

class GameboardGenerator:
    def __init__(self, word_to_index, nouns):
        self.word_to_index = word_to_index
        self.nouns = nouns
        self.noun_index_to_word_index = []
        for noun in nouns:
            try:
                word_index = self.word_to_index[noun]
            except:
                continue
            self.noun_index_to_word_index.append(word_index)

    def generateRandomBoard(self):
        sample = np.random.choice(len(self.noun_index_to_word_index), size=25, replace=False)
        for i in range(sample.shape[0]):
            sample[i] = self.noun_index_to_word_index[sample[i]]
        ourwords = np.random.randint(0, high=2, size=25)
        return sample, ourwords





