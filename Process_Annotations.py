import glob
import os
import numpy as np

def merge_Annotations(annotations,distMin):
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
        malignancy = np.int(np.median(annotations2Merg[:,3]))          
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
        roi = compute_ROI(*centroid,roi_Size)
        nodule_Img = np.array(exam[roi[0]:roi[1]-1,roi[2]:roi[3]-1,roi[4]:roi[5]-1],dtype=np.int16)        
        path = output_Path+'/'+str(malignancy)+'/'
        os.makedirs(path, exist_ok=True)
        np.save(path+exam_Name+'_'+str(centroid)+'_'+str(malignancy)+'.npy',nodule_Img)
    return True
    
def process_Annotations(src_Path,output_Path,roi_Shape,distMin):
    #Log Variables 
    num_of_nodules_total = 0
    num_of_valid_exams = 0
    num_of_invalid_exams = 0
    
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
            merg_annt = merge_Annotations(annotations,distMin)
            ## Extracts nodules from exam and exports them as a npy file 
            export_Nodules(exam,merg_annt,roi_Shape,output_Path,exam_Name)
            ##Log Variables 
            num_of_nodules_total+=len(merg_annt)
            num_of_valid_exams+=1

    print("\nProcessed Exams: ",num_of_invalid_exams+num_of_valid_exams)
    print("Invalid Exams: ",num_of_invalid_exams)
    print("Valid Exams: ",num_of_invalid_exams)
    print("Number Of Nodules: ",num_of_nodules_total)    

    return True

def main():
    src_Path = "/home/raul/PROJECTS/LIDC-IDRI/Src_Folder_PYTHON"
    output_Path = "/home/raul/PROJECTS/LIDC-IDRI/Out_Folder_PYTHON"         
    nodule_imgs_radius = 8
    distMin = 5
    process_Annotations(src_Path,output_Path,nodule_imgs_radius,distMin)
    return True
main()