import glob
import os
import numpy as np
import matplotlib.pyplot as plt

def merge_Annotations(annotations,distMin,merge_type):
    #Calc Centroid and replace by list of points    
    nodules_list = []
    for radioID in range(len(annotations)):
        for nodules in range(len(annotations[radioID])):
            nodules_list.append([*np.round((np.mean(annotations[radioID][nodules][1],axis=0))).astype(int),annotations[radioID][nodules][0]]) 
   
    # Remove Close centroids (Euclidian Norm between nodules > 'distMin')
    filtered_Nodules = []
    while(len(nodules_list) > 0):
        nodule_A = nodules_list[0]

        distances = np.array([int(np.linalg.norm(np.array(nodule_A[0:3])-np.array(nodule_B[0:3]))) for nodule_B in nodules_list])        
        annotations2Merg = np.array([nodule_B for nodule_B, dist in zip(nodules_list,distances) if dist <= distMin],dtype=object)
        idxs = np.array([idx for idx, dist in enumerate(distances) if dist <= distMin])

        centroid = list(map(int,np.mean(annotations2Merg[:,0:3],axis=0)))
        if(merge_type is 'Median'):
            malignancy = np.int(np.median(annotations2Merg[:,3]))  
        else:
            malignancy = np.int(np.ceil(np.mean(annotations2Merg[:,3])))         
        
        filtered_Nodules.append([*centroid,malignancy])
        nodules_list = np.delete(nodules_list,idxs,axis=0)
    
    return filtered_Nodules

def compute_ROI(x,y,z,r):
    return np.array([int(z-r),int(z+r+1),
                    int(y-r),int(y+r+1),
                    int(x-r),int(x+r+1)])

def export_Nodules(exam,annotations,roi_Size,output_Path,exam_Name):
    for nodule_info in annotations:
        centroid = nodule_info[0:3]
        malignancy = nodule_info[3]
        if (malignancy == 0):
            continue
        roi = compute_ROI(*centroid,int(roi_Size/2))
        nodule_Img = np.array(exam[roi[0]:roi[1]-1,roi[2]:roi[3]-1,roi[4]:roi[5]-1],dtype=np.int16)        
        
        if(nodule_Img.shape[0] == roi_Size and
           nodule_Img.shape[1] == roi_Size and
           nodule_Img.shape[2] == roi_Size):
            path = output_Path+'/'+str(malignancy-1)+'/'
            os.makedirs(path, exist_ok=True)
            np.save(path+exam_Name+'_'+str(centroid)+'_'+str(malignancy-1)+'.npy',nodule_Img)
        else:
            print(nodule_Img.shape, '<',roi_Size)
    return True
    
def process_Annotations(src_Path,output_Path,nodule_size_list,distMin,merge_type):
    #Log Variables 
    num_of_nodules_total = 0
    num_of_valid_exams = 0
    num_of_invalid_exams = 0
#    malignancy = []

    # Runs through all folders in the 'PathName' path.
    for idx, path_Folder in enumerate(glob.glob(src_Path+"/*")): 
        exam_Name = path_Folder.split('/')[-1]
        print(idx, exam_Name)
        
        annotations = np.load(path_Folder+'/'+exam_Name+'_annotations.npy')
        if (len(annotations) < 1 ):
            print(exam_Name+' has '+ str(len(annotations))+' annotations')
            num_of_invalid_exams+=1
        else:
            ## Load Exam Volume 
            exam = np.load(path_Folder+'/'+exam_Name+'.npy')
            ## Merge Annotations by distance of centroids 
            merg_annt = merge_Annotations(annotations,distMin,merge_type)    
            ## Extracts nodules from exam and exports them as a npy file 
            for nodule_size in nodule_size_list:
                out_path = output_Path+str(nodule_size)+'x'+str(nodule_size)+'x'+str(nodule_size)+'/'         
                export_Nodules(exam,merg_annt,nodule_size,out_path,exam_Name)
                
            ##Log Variables 
            num_of_nodules_total+=len(merg_annt)
            num_of_valid_exams+=1
#            malignancy.extend(np.array(merg_annt)[:,-1])

#    fig = plt.figure()
#    ax = fig.gca()
#    plt.hist(malignancy, bins='auto')
#    ax.set_yticks(np.arange(0, 1500, 150))
#    plt.grid()
#    plt.show()
    
    print("\nProcessed Exams: ",num_of_invalid_exams+num_of_valid_exams)
    print("Invalid Exams: ",num_of_invalid_exams)
    print("Valid Exams: ",num_of_valid_exams)
    print("Number Of Nodules: ",num_of_nodules_total)    

    return True

def main():
    src_Path = "/home/raul/PROJECTS/LIDC-IDRI/Src_Folder_NPY"
    output_Path = '/home/raul/PROJECTS/LIDC-IDRI/Out_Folder_NODULES_NPY/Median/'

    distMin = 5
    nodule_size_list = [8,16,24,32,48,64]
    #TODO: #   merge_type_list = ['Median','Mean']
    merge_type = 'Median'
    process_Annotations(src_Path,output_Path,nodule_size_list,distMin,merge_type)
    return True

main()