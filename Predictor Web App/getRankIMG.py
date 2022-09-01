import sys
import os
from PIL import Image

def rank_img(rank_num):

   img_dir = os.getcwd() + '\RankImages\\'

   print(img_dir)

   if(rank_num == 1):
      image = Image.open(img_dir + r'd.png')
   elif (rank_num == 2):
      image = Image.open(img_dir + r'c.png')
   elif (rank_num == 3):
      image = Image.open(img_dir + r'b.png')
   elif (rank_num == 4):
      image = Image.open(img_dir + r'a.png')
   elif (rank_num == 5):
      image = Image.open(img_dir + r's.png')
   elif (rank_num == 6):
      image = Image.open(img_dir + r'ss.png')
   elif (rank_num == 7):
      image = Image.open(img_dir + r'u.png')
   elif (rank_num == 8):
      image = Image.open(img_dir + r'x.png')
   
   return image

