import logging
import os

def create_txt(root, txt_name):
    """
    Create txt file for customer Dataset class

    the images and labels will be saved as:
    path, label
    .../Images/000000/1.jpg, 0

    e.g. .../Images/000000/1.jpg the dir name 000000 is label number
    """
    labels = os.listdir(root)
    txt_path = os.path.join(root, (txt_name + '.txt'))
    with open(txt_path, 'w') as f:
        for l in labels:
            if os.path.isdir(os.path.join(root, l)):
                image_paths = os.listdir(os.path.join(root,l))
                for i in image_paths:
                    image_path = os.path.join(root, l, i)
                    f.writelines(image_path + ',' + str(int(l)) + '\n')
    # print(labels)
    print('Finish writing txt')

count = 0
def count_all_images(root):
    '''
    Count the total number of jpg files under a dir
    '''
    global count
    dirs = os.listdir(root)
    for dir in dirs:
        path = os.path.join(root, dir)
        if os.path.isdir(path):
            count_all_images(path)
        if path.endswith('.jpg'):
            count += 1
    print(count)
    return


if __name__ == '__main__':
    create_txt('D:\Data\GTSRB\Final_Training\Images_jpg', 'train')
    # count_all_images('D:\Data\GTSRB\Final_Training')