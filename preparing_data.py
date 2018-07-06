from PIL import Image
import os
import random
dir = './my_pic'
list_ = os.listdir(dir)
list = [os.path.join(dir,_) for _ in list_ if _.split('.')[1] == 'jpg']

i = 0
for name in list:
    img = Image.open(name)
    shape = img.size
    new_size = min(shape[0],shape[1])
    if (shape[0]>=new_size and shape[1]>=new_size):
        for _ in range(10):
            x0 = random.randint(0,shape[0]-new_size)
            y0 = random.randint(0,shape[1]-new_size)
            out = img.crop((x0,y0,x0+new_size,y0+new_size))
            out = out.resize((512,512))
            out.save(os.path.join(dir,'image_%d.jpg'%i))
            i = i+1