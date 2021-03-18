"""

TODO:
    * log_softmax(input, dim=1)
"""

from torch import nn

F = nn.functional

class DNN(nn.Module):


    def __init__(self, n_input=784, n_output=10)-> None:
        """

        Args:

        Notes:
            fc: fully connected layer
        """
        super(DNN, self).__init__()
        self.n_input = n_input
        self.n_output = n_output

        self.fc1 = nn.Linear(self.n_input, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, self.n_output)


    def forward(self, x):
        """

        Args:

        Notes:
            
        """
        x = x.float()
        x = x.view(-1, self.n_input)
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        h4 = F.relu(self.fc4(h3))
        h5 = F.relu(self.fc5(h4))
        h6 = self.fc6(h5)
        output = F.log_softmax(h6, dim=1)
        
        return output


def get_model(model_str: str):

    model = None

    if model_str == 'DNN':
        model = DNN

    return model