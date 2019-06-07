from net1 import *
from Data import *

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pretrained = torch.load('D:\\furniture-4.pth',
                            map_location=lambda storage,loc:storage)
    model = ColorNet()
    model.to(device)
    #checkpoint = torch.load('D:\\checkpoint-trained.pth.tar', map_location=lambda storage, loc: storage)
    #model.load_state_dict(checkpoint['state_dict'])
    model.load_state_dict(pretrained)
    i=Image.open('D:/git/dpgui/2.jpg')
    img_l = np.asarray(i)
    img_l = img_as_float(img_l)
    img_l = torch.from_numpy(img_l).unsqueeze(0).unsqueeze(0).float()
    output = model(img_l)
    save_path = {'grayscale': 'D:\\output\\gray\\',
                 'colorized': 'D:\\output\\color\\'}
    save_name = 'pic-{}.jpg'.format(1)

    to_rgb(img_l.cpu(), output.detach().cpu(), save_path=save_path,
           save_name=save_name)
    '''
    pic = 1
    for input_gray, input_ab, target in dataloader['train']:
        input_gray = input_gray.to(device)
        print('a',input_gray.shape)
        print('b',input_gray)
        input_ab = input_ab.to(device)
        print(input_gray)

        target = target.to(device)

        output = model(input_gray)
        print(output)
        save_path = {'grayscale': 'D:\\output\\gray\\',
                     'colorized': 'D:\\output\\color\\'}
        save_name = 'pic-{}.jpg'.format(pic)

        to_rgb(input_gray.cpu(), output.detach().cpu(), save_path=save_path,
               save_name=save_name)

        print(pic)
        pic += 1
    '''