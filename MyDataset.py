from torch.utils.data import Dataset
import scipy.io as scio
import torch

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class MyDataset(Dataset):
    def __init__(self, filename):
        super(MyDataset, self).__init__()
        self.data = scio.loadmat(filename)['xx'].astype('float32')

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index]).to(device)

    def __len__(self):
        return len(self.data)

    def get_numpy_data(self):
        return self.data
