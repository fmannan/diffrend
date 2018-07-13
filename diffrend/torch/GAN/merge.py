import os
import shutil
list=[r'/mnt/home/dvazquez/Sai/real_data/bamboohouse/vis_input',r'/mnt/home/dvazquez/Sai/real_data/bamboohouse2/vis_input',r'/mnt/home/dvazquez/Sai/real_data/bamboohouse3/vis_input',
r'/mnt/home/dvazquez/Sai/real_data/bamboohouse4/vis_input',r'/mnt/home/dvazquez/Sai/real_data/bamboohouse5/vis_input']
# list=[r'/mnt/home/dvazquez/Sai/real_data/bedroom_prerender/vis_input',r'/mnt/home/dvazquez/Sai/real_data/bedroom_prerender4/vis_input',r'/mnt/home/dvazquez/Sai/real_data/bedroom_prerender5/vis_input',
# r'/mnt/home/dvazquez/Sai/real_data/bedroom_prerender6/vis_input',r'/mnt/home/dvazquez/Sai/real_data/bedroom_prerender7/vis_input',r'/mnt/home/dvazquez/Sai/real_data/bedroom_prerender8/vis_input',
# r'/mnt/home/dvazquez/Sai/real_data/bedroom_prerender9/vis_input',r'/mnt/home/dvazquez/Sai/real_data/bedroom_prerender10/vis_input',r'/mnt/home/dvazquez/Sai/real_data/bedroom_prerender2/vis_input',
# r'/mnt/home/dvazquez/Sai/real_data/bedroom_prerender12/vis_input',r'/mnt/home/dvazquez/Sai/real_data/bedroom_prerender13/vis_input',r'/mnt/home/dvazquez/Sai/real_data/bedroom_prerender3/vis_input']

# #list=[r'/home/dvazquez/Sai/temp1',r'/home/dvazquez/Sai/temp2',r'/home/dvazquez/Sai/temp3']


TargetFolder = '/mnt/home/dvazquez/Sai/real_data/house_data/data'

i=1
for dirs in list:

    for files in os.listdir(dirs):
        #import ipdb; ipdb.set_trace()


        if files.endswith('.png'):
            print("Found")
            print(files)
            cam=files[:-4]+".npy"
            light=files[:-9]+"light"+files[-9:-4]+".npy"
            # print(cam)
            # print(light)
            SourceFolder = os.path.join(dirs,files)
            SourceFolder_cam = os.path.join(dirs,cam)
            SourceFolder_light = os.path.join(dirs,light)
            TargetFolder2 = os.path.join(TargetFolder,str(i)+".png")
            TargetFolder2_cam = os.path.join(TargetFolder,str(i)+".npy")
            TargetFolder2_light = os.path.join(TargetFolder,str(i)+"_light.npy")
            shutil.copy2(SourceFolder, TargetFolder2)
            shutil.copy2(SourceFolder_cam, TargetFolder2_cam)
            shutil.copy2(SourceFolder_light, TargetFolder2_light)
            #shutil.copy2(SourceFolder_light, TargetFolder2_light2)
            i+=1
