from Data import *
from net import *


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def train(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            pic = 1
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for input_gray, input_ab, target in dataloader[phase]:
                input_gray = input_gray.to(device)
                input_ab = input_ab.to(device)
                target = target.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    output = model(input_gray)
                    loss = criterion(output, input_ab)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * input_gray.size(0)
                print('pic:',pic,'    ','single loss:',loss.item())
                pic += 1

            epoch_loss = running_loss / dataset_sizes[phase]
            print('{} Loss: {:.5f}'.format(phase, epoch_loss))
            for input_gray, input_ab, target in dataloader['valid']:
                save_path = {'grayscale':'D:\\outputs\\gray\\','colorized':'D:\\outputs\\color\\'}
                save_name = '%d.jpg' % (epoch,)
                to_rgb(input_gray.cpu(), model(input_gray).detach().cpu(),save_path=save_path,save_name=save_name)
                break

            # deep copy the model
            if epoch == 0:
                best_loss = epoch_loss
            elif phase == 'valid' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
            print('spend time:',time.time()-since)
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    # 这里要加路径！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
    torch.save(model.state_dict(), '/model-epoch-{}-losses-{:.3f}.pth'.format(num_epochs+1,best_loss))
    return model

if __name__ == '__main__':
    model = Network()
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(),lr=0.001,momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    model = train(model,criterion,optimizer,exp_lr_scheduler,num_epochs=3)

