from demo.model import read_csv, read_img, format_data, train, test,CNN
import  torch
if __name__ =='__main__':
    lables = read_csv()
    x, y = read_img(lables)
    train_x, train_y, test_x, test_y = format_data(x, y)
    for i in range(10):
        train(train_x, train_y)
        test(test_x, lables)
