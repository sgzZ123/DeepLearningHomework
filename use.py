from net1 import *
from Data import *

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pretrained = torch.load('D:\\model-epoch-100-losses-0.004293.pth',
                            map_location=lambda storage,loc:storage)
    model = ColorNet()
    model.to(device)
    #checkpoint = torch.load('D:\\checkpoint-trained.pth.tar', map_location=lambda storage, loc: storage)
    #model.load_state_dict(checkpoint['state_dict'])
    model.load_state_dict(pretrained)
    pic = 1
    for input_gray, input_ab, target in dataloader['train']:
        input_gray = input_gray.to(device)
        input_ab = input_ab.to(device)

        target = target.to(device)

        output = model(input_gray)
        print(output.shape)
        save_path = {'grayscale': 'D:\\output\\gray\\',
                     'colorized': 'D:\\output\\color\\'}
        save_name = 'pic-{}.jpg'.format(pic)

        to_rgb(input_gray.cpu(), output.detach().cpu(), save_path=save_path,
               save_name=save_name)

        print(pic)
        pic += 1