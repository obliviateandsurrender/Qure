import pandas as pd
import numpy as np

import torch
import torch.nn as nn

import pennylane as qml
import time


from sentence_transformers import SentenceTransformer

import copy

import seaborn as sns

import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, random_split

from tqdm import tqdm
tqdm.pandas()


class CustomTextDataset(Dataset):
    def __init__(self, text, labels):
        self.labels = labels
        self.text = text
    def __len__(self):
            return len(self.labels)
    def __getitem__(self, idx):
            label = self.labels[idx]
            text = self.text[idx]
            return text, label

##########################################
#Load your dataset in DataFrame in the following format:
# ----------------------------------
#|   text         |      label     |
#|   example 1    |        0       |
#|   example 2    |        1       |
#|     ...        |       ...      |
#-----------------------------------
##########################################


#df = pd.read_pickle("your_text_dataset.pkl")
import numpy as np

# Load the matrix from a .npy file
time_series = np.load('../dataset.npy')

# Normalize
time_series_norm = time_series / 10 * 1
        
    
#create the CustomTextDataSet object and split in training and validation
#using the chosen number of samples
TextDataset = CustomTextDataset(time_series_norm[:,:6,:-1].reshape(6000, 6*9), time_series_norm[:,6, -1].reshape(6000, 1))
train_set, validation_set = random_split(TextDataset, [5000, 1000])


train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
DL_Validation = DataLoader(validation_set, batch_size=32, shuffle=True)


dataloaders = {
    "train": train_loader,
    "validation": DL_Validation
}
dataset_sizes = {
    "train": len(train_set),
    "validation": len(validation_set)
}


def train_model(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 10000.0  # Large arbitrary number
    best_acc_train = 0.0
    best_loss_train = 10000.0  # Large arbitrary number
    
    history_loss = {
        "train": [],
        "validation": []
    }
    print("Training started:")

    for epoch in range(num_epochs):

        # Each epoch has a training and validation phase
        for phase in ["train", "validation"]:
            if phase == "train":
                # Set model to training mode
                model.train()
            else:
                # Set model to evaluate mode
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data.
            n_batches = dataset_sizes[phase] // batch_size
            it = 0
            
            #receives a dict of dataloaders
            count = 0
            for inputs, labels in dataloaders[phase]:
                count += 1
                if phase == "train" and count == 151: break
                if phase == "validation" and count == 30: break
                since_batch = time.time()
                batch_size_ = len(inputs)
#                 inputs = inputs.to(device)
#                 labels = labels.to(device)
                optimizer.zero_grad()
                
                # Track/compute gradient and make an optimization step only when training
                with torch.set_grad_enabled(phase == "train"):
                    #print(inputs.shape)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs.float(), labels.float())
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # Print iteration results
                running_loss += loss.item() * batch_size_
                #print("preds shape: ", preds.shape)
                #print("preds: ", preds)
                #print("labels: ", labels)
                #print("labels shape: ", labels.shape)
                #print("outputs: ", outputs)
                #print("outputs shape: ", outputs.shape)
                batch_corrects = torch.sum((outputs-labels)**2).item()
                running_corrects += batch_corrects
                print(
                    "Phase: {} Epoch: {}/{} Iter: {}/{} Batch time: {:.4f}".format(
                        phase,
                        epoch + 1,
                        num_epochs,
                        it + 1,
                        n_batches + 1,
                        time.time() - since_batch,
                    ),
                    end="\r",
                    flush=True,
                )
                it += 1

            # Print epoch results
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            print(
                "Phase: {} Epoch: {}/{} Loss: {:.4f} Acc: {:.4f}        ".format(
                    "train" if phase == "train" else "validation  ",
                    epoch + 1,
                    num_epochs,
                    epoch_loss,
                    epoch_acc,
                )
            )
            
            history_loss[phase].append(epoch_loss)

            # Check if this is the best model wrt previous epochs
            if phase == "validation" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == "validation" and epoch_loss < best_loss:
                best_loss = epoch_loss
            if phase == "train" and epoch_acc > best_acc_train:
                best_acc_train = epoch_acc
            if phase == "train" and epoch_loss < best_loss_train:
                best_loss_train = epoch_loss

            # Update learning rate
            if phase == "train":
                scheduler.step()

    # Print final results
    model.load_state_dict(best_model_wts)
    
    time_elapsed = time.time() - since
    print(
        "Training completed in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60)
    )
    print("Best test loss: {:.4f} | Best test accuracy: {:.4f}".format(best_loss, best_acc))
    
    sns.lineplot(data=history_loss)
    
    print(best_model_wts)
    
    return model


###################### CONFIG ############################

n_qubits = 9                # Number of qubits
step = 0.001               # Learning rate
batch_size = 32              # Number of samples for each training step
num_epochs = 30              # Number of training epochs
q_depth = 6                 # Depth of the quantum circuit (number of variational layers)
#gamma_lr_scheduler = 0.1    # Learning rate reduction applied every 10 epochs.
q_delta = 0.01              # Initial spread of random quantum weights
start_time = time.time()    # Start of the computation timer

#########################################################

dev = qml.device("default.qubit", wires=n_qubits)

def H_layer(nqubits):
    """Layer of single-qubit Hadamard gates.
    """
    for idx in range(nqubits):
        qml.Hadamard(wires=idx)


def RY_layer(w):
    """Layer of parametrized qubit rotations around the y axis.
    """
    for idx, element in enumerate(w):
        qml.RY(element, wires=idx)


def entangling_layer(nqubits):
    """Layer of CNOTs followed by another shifted layer of CNOT.
    """
    # In other words it should apply something like :
    # CNOT  CNOT  CNOT  CNOT...  CNOT
    #   CNOT  CNOT  CNOT...  CNOT
    for i in range(0, nqubits - 1, 2):  # Loop over even indices: i=0,2,...N-2
        qml.CNOT(wires=[i, i + 1])
    for i in range(1, nqubits - 1, 2):  # Loop over odd indices:  i=1,3,...N-3
        qml.CNOT(wires=[i, i + 1])
        
        
@qml.qnode(dev, interface="torch")
def quantum_net(q_input_features, q_weights_flat):
    """
    The variational quantum circuit.
    """

    # Reshape weights
    q_weights = q_weights_flat.reshape(q_depth, n_qubits)

    # Start from state |+> , unbiased w.r.t. |0> and |1>
    H_layer(n_qubits)

    # Embed features in the quantum node
    RY_layer(q_input_features)

    # Sequence of trainable variational layers
    for k in range(q_depth):
        entangling_layer(n_qubits)
        RY_layer(q_weights[k])

    # Expectation values in the Z basis
    exp_vals = [qml.expval(qml.PauliZ(position)) for position in range(n_qubits)]
    return tuple(exp_vals)



class QuantumSentenceTransformer(nn.Module):
    """
    Torch module implementing the *dressed* quantum net.
    """

    def __init__(self, 
                 freeze_bert_layers = "all",       
                 pretrained_model = "distiluse-base-multilingual-cased-v1", 
                 device='cpu'):
        
        """
        Definition of the *dressed* layout.
        """

        super().__init__()
        
        # init Sentence Transformer and freeze all its layers by default. Only Quantum Layers are going
        # to be trainable
        self.device = device
        self.sentence_transformer  =  SentenceTransformer(pretrained_model, device=device)
        for param in self.sentence_transformer.parameters():
            if freeze_bert_layers == "all" or param in freeze_bert_layers:
                param.requires_grad = False
            
        self.pre_net = nn.Linear(6*9, n_qubits)
        self.q_params = nn.Parameter(q_delta * torch.randn(q_depth * n_qubits))
        self.post_net = nn.Linear(n_qubits, 9)
        
        weight_matrix = torch.randn(n_qubits, 6*9).float()  # weight matrix of shape (n_qubits, 20*9), converted to float
        print(type(weight_matrix))

        self.pre_net.weight = torch.nn.Parameter(weight_matrix)

    def forward(self, input_features):
        """
        Defining how tensors are supposed to move through the *dressed* quantum
        net.
        """
        
        #generate embeddings from input text
        #input_features = self.sentence_transformer.encode(input_text, convert_to_tensor=True)
        # obtain the input features from the text embeddings for the quantum circuit
        # by reducing the feature dimension from 512 to 4
        #KIKIU
        
        input_features = input_features.reshape(32, 6*9).float()
        #print("input feature type: ", type(input_features))
        #print("input shape: ", input_features.shape)
        pre_out = self.pre_net(input_features)
        #print("pre_out shape: ", pre_out)
        q_in = torch.tanh(pre_out) * np.pi / 2.0 #TEST ReLU

        # Apply the quantum circuit to each element of the batch and append to q_out
        q_out = torch.Tensor(0, n_qubits)
        q_out = q_out.to(self.device)
        for elem in q_in:
            q_out_elem = quantum_net(elem, self.q_params).float().unsqueeze(0)
            q_out = torch.cat((q_out, q_out_elem))

        # return the two-dimensional prediction from the postprocessing layer
        return self.post_net(q_out)

model = QuantumSentenceTransformer(device='cpu')
criterion = torch.nn.MSELoss()

#optimizer = torch.optim.SGD(model.parameters(),
#                            lr=step,
#                            momentum=momentum,
#                            weight_decay=weight_decay)

optimizer = optim.Adam(model.parameters(), lr=step, )

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                       T_max=len(train_loader),
                                                       eta_min=0)

train_model(model, criterion, optimizer, scheduler, num_epochs)

