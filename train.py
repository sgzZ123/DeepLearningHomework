import time
from network import *
import copy
from dataset import *
import torch.optim as optim
from torch.optim import lr_scheduler


def train(L_model, M_model, G_model, C_model, criterion, optimizer, schedular, dataset, num_epochs=100):
    best_model_wts = {'L':copy.deepcopy(L_model.state_dict()),
                      'M':copy.deepcopy(M_model.state_dict()),
                      'G':copy.deepcopy(G_model.state_dict()),
                      'C':copy.deepcopy(C_model.state_dict())
                      }
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        start = time.time()
        for phase in ['train', 'valid']:
            if phase == 'train':
                schedular.step()
                L_model.train()
                M_model.train()
                G_model.train()
                C_model.train()
                dataset.training_number = 0
            else:
                L_model.eval()
                M_model.eval()
                G_model.eval()
                C_model.eval()
                dataset.validation_number = 0

            running_loss = 0.0
            running_corrects = 0
            pic_num=0

            while True:
                try:
                    if phase == 'train':
                        img, labels = dataset.GetTrainingData()
                    else:
                        img, labels = dataset.GetValidationData()
                    img, labels = img.float(), labels.float()
                    optimizer.zero_grad()
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        output_L = L_model(img)
                        output_M = M_model(output_L)
                        output_G1, output_G2 = G_model(output_L)
                        t = output_G2
                        input_C = t
                        for i in range(28 // 7 - 1):
                            input_C = torch.cat((input_C, t), 2)
                        t = input_C
                        for i in range(28 // 7 - 1):
                            input_C = torch.cat((input_C, t), 3)
                        input_C = torch.cat((input_C, output_M), 1)
                        output = C_model(input_C)
                        _, preds = torch.max(output, 1)
                        loss = criterion(output, labels)
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    # statistics
                    print('loss:',loss.item())
                    print('img.size(0):',img.size(0))
                    running_loss += loss.item() * img.size(0)
                    running_corrects += torch.sum(preds == labels.data.long())
                    print('running corrects:', running_corrects)
                    pic_num += 1
                    print(pic_num)
                except FileNotFoundError:
                    break
            if phase == 'train':
                print(running_loss)
                print(dataset.training_number)
                epoch_loss = running_loss / dataset.training_number
                epoch_acc = running_corrects / dataset.training_number
            else:
                epoch_loss = running_loss / dataset.validation_number
                epoch_acc = running_corrects / dataset.validation_number
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            
            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts['L'] = copy.deepcopy(L_model.state_dict())
                best_model_wts['M'] = copy.deepcopy(M_model.state_dict())
                best_model_wts['G'] = copy.deepcopy(G_model.state_dict())
                best_model_wts['C'] = copy.deepcopy(C_model.state_dict())
        end = time.time()
        print('time:', (end - start))

    L_model.load_state_dict(best_model_wts['L'])
    M_model.load_state_dict(best_model_wts['M'])
    G_model.load_state_dict(best_model_wts['G'])
    C_model.load_state_dict(best_model_wts['C'])

    return L_model, M_model, G_model, C_model

if __name__ == '__main__':
    print('cuda' if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    L_model = LowLevelNetwork()
    L_model = L_model.to(device)
    M_model = MidLevelNetwork()
    M_model = M_model.to(device)
    G_model = GlobalNetwork()
    G_model = G_model.to(device)
    C_model = ColorizationNetwork()
    C_model = C_model.to(device)
    model = {'L': L_model, 'M': M_model, 'G': G_model, 'C': C_model}
    optimizer = optim.SGD([{'params': model[x].parameters()}
                           for x in ['L', 'M', 'G', 'C']], lr=0.001)
    criterion = nn.MSELoss()
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    dataset = Data()
    L_model, M_model, G_model, C_model = train(L_model, M_model, G_model, C_model, criterion,optimizer,exp_lr_scheduler, dataset)
