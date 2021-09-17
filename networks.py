import torch
import torch.nn

num_nouns = 6801
vector_size = 50
board_size = 25
max_guesses = 5


class SpymasterNetwork(torch.nn.Module):
    def __init__(self):
        super(SpymasterNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(vector_size*board_size + board_size, 200)  # input: board & vector of if the word is ours or not
        self.fc2 = torch.nn.Linear(200, 200)
        self.fc3 = torch.nn.Linear(200, 300)
        self.fc4 = torch.nn.Linear(300, num_nouns)  # output: clue

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = torch.nn.functional.softmax(self.fc4(x))
        return x


class GuessingNetwork(torch.nn.Module):
    def __init__(self):
        super(GuessingNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(vector_size*board_size + num_nouns, 200)  # input: board & clue
        self.fc2 = torch.nn.Linear(200, 200)
        self.fc3 = torch.nn.Linear(200, 300)
        self.fc4 = torch.nn.Linear(300, board_size)  # output: vector of guesses

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = torch.nn.functional.sigmoid(self.fc4(x))
        return x
