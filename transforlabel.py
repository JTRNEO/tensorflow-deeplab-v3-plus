import numpy as np 
import json
import cv2
import os 
path='./SegmentationClass/'

f=open('./instances_train.json',encoding='utf-8')
anns=json.load(f)
categorylist=anns['categories']

Sports=[]
Bridges=[]
Storage=[]
Shipping=[]
Port=[]
Planes=[]
Ships=[]
Intersection=[]
Airports=[]
Vehicles=[]
Parking=[]
Buildings=[]


Classes={'Sports Stadium/Field':Sports,
'Bridges':Bridges,
'Storage Tank':Storage,
'Shipping Containers':Shipping,
'Port':Port,
'Planes':Planes,
'Ships':Ships,
'Intersection/Crossroads':Intersection,
'Airports':Airports,
'Vehicles':Vehicles,
'Parking Lots':Parking,
'Buildings':Buildings,
}


for i in categorylist:
    if i['supercategory'] in Classes.keys():
       Classes[i['supercategory']].append(i['id'])

a=os.listdir(path)
a.sort(key=lambda x:int(x[:-4]))

for n in a:
    img=cv2.imread(path+n,cv2.IMREAD_GRAYSCALE)
    for index,x in enumerate(Classes.values()):
        
        for k in x:
            img=np.where(img==k,-(index+1),img)

    img=np.where(img>0,0,img)
    img=-(img)
    cv2.imwrite(path+n,img)        
    