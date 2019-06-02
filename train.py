from net1 import *
from Data import *
import sys


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def train_once(train_loader, model, criterion, optimizer,epoch):
    print('start train epoch {}'.format(epoch))
    model.train()

    losses = Avg()
    data_time = Avg()


    for i , (input_gray, input_ab, target) in enumerate(train_loader):
        start = time.time()
        input_gray, input_ab, target = input_gray.to(device), input_ab.to(device), target.to(device)
        output_ab = model(input_gray)
        loss = criterion(output_ab, input_ab)
        losses.update(loss.item(), input_gray.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        data_time.update(time.time() - start)

    print('average time for each pic:{}'.format(data_time.avg))
    print('average loss:{}'.format(losses.avg))
    print('finish training epoch-{}'.format(epoch))


def valid_once(val_loader, model, criterion, epoch):
    print('start validation')
    model.eval()

    data_time = Avg()
    losses = Avg()

    for i, (input_gray, input_ab, target) in enumerate(val_loader):
        start = time.time()
        input_gray, input_ab, target = input_gray.to(device), input_ab.to(device),target.to(device)

        output_ab = model(input_gray)
        loss = criterion(output_ab, input_ab)
        losses.update(loss.item(), input_gray.size(0))

        data_time.update(time.time() - start)

    print('average time for each pic:{}'.format(data_time.avg))
    print('average loss:{:.8f}'.format(losses.avg))
    print('finish validation epoch-{}'.format(epoch))

    return losses.avg


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'true':
        model = ColorNet(load=True).to(device)
    else:
        model = ColorNet(load=False).to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-2,weight_decay=0.0)
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs = 50
    best_losses = 1e10
    for epoch in range(epochs):
        train_once(train_loader, model, criterion, optimizer, epoch)
        with torch.no_grad():
            losses = valid_once(val_loader, model, criterion, epoch)
        if losses < best_losses:
            best_losses = losses
            best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(),'/output_dir/model-epoch-{}-losses-{:.6f}.pth'.format(epochs,best_losses))

