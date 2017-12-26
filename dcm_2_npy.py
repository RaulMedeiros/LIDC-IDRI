import dicom
import os
import glob
import numpy as np
import xml.etree.ElementTree as ET
np.set_printoptions(threshold=np.nan)

#Validades the Path Folder identifying the folder with more images. 
def validate_Folder_Path(path_Folder):
    # Calculates the number of files in each sub folder
    len_Files =[]
    for scan_Path_Folder in glob.glob(path_Folder+"/*/*"):
        len_Files.append([len(glob.glob(scan_Path_Folder+"/*")),scan_Path_Folder]) 
    # Choses the folder with more elements and return its path            
    return max(len_Files)[1]

def dicom_List_2_Array(lstFilesDCM):
    # Get ref file    
    RefDs = dicom.read_file(lstFilesDCM[0])
    RescaleIntercept = int(RefDs.data_element('RescaleIntercept').value)    
    
    exam = []
    # loop through all the DICOM files
    for idx, filenameDCM in enumerate(lstFilesDCM):
        # read the file and rescale to hounsfield units
        pixel_array = np.array(dicom.read_file(filenameDCM).pixel_array,dtype=np.int16)+RescaleIntercept
        # store the raw image data
        exam.append(pixel_array)
    
    return np.array(exam)

def exam_2_Mat(exam_Path):
    # Runs through all files in the 'ExamPath+"/*"' folder
    dicom_Paths = []
    for path_Files in glob.glob(exam_Path+'/*'):
        if (".dcm" in path_Files): 
            ds = dicom.read_file(path_Files)
            #Stores the z index and the respective path
            dicom_Paths.append([float(ds.ImagePositionPatient[2]),path_Files])

    #Sorts the List of Dicoms by Z index 
    dicom_Paths=sorted(dicom_Paths,key=lambda x: x[0])

    #creates a List of the slices pixels
    exam_Mat = dicom_List_2_Array([i[1] for i in dicom_Paths])
    list_idx_Z = [item[0] for item in dicom_Paths]
    return exam_Mat, list_idx_Z

def load_Nodule_Voxels(unblindedReadNodule,listOfIdxZ):
    nodule_points = []
    for roi in unblindedReadNodule.iter('{http://www.nih.gov}roi'): 
        
        if(roi[2].text == "TRUE"):
            # add segmented pixels in a slice to an array
            xList = []; yList = []; zList = [];
            for edgeMap in roi.iter('{http://www.nih.gov}edgeMap'):       
                xList.append(int(edgeMap[0].text))     
                yList.append(int(edgeMap[1].text))   
                zList.append(listOfIdxZ.index(float(roi[0].text)))                  
            nodule_points.extend(np.stack((np.array(xList), np.array(yList), np.array(zList)), axis=-1))
    return np.array(nodule_points)

# Extract List of Rois
def XML_Parser(ExamPath,listOfIdxZ):
   
    exam_Annotations = []
    # Finds a xml file into the ExamPath folder   
    xml_File = glob.glob(ExamPath+"/*.xml")
    
    if (len(xml_File) > 1):
        print('more than one xml file in the folder')
    else:
        tree = ET.parse(xml_File[0])
        root = tree.getroot()
        
        # Looks for All Radiologists  
        for readingSession in root.iter('{http://www.nih.gov}readingSession'):    
        
            radiologist_Annotations = []
            # Looks for All Annotations(Nodules) of a given radiologists 
            for unblindedReadNodule in readingSession.iter('{http://www.nih.gov}unblindedReadNodule'):  
            
                # Checks if there are a Malignancy into this annotations(Object/Nodules)
                if (unblindedReadNodule[1].text != None and unblindedReadNodule[1].tag == '{http://www.nih.gov}characteristics' ):
                    
                    nodule_points = load_Nodule_Voxels(unblindedReadNodule,listOfIdxZ)
                    malignancy = int(unblindedReadNodule[1][8].text) 
                    radiologist_Annotations.append(np.array([malignancy,nodule_points],dtype=object)) #
            
            if(radiologist_Annotations):
                exam_Annotations.append(radiologist_Annotations)
                
            print('num of nodules',len(radiologist_Annotations))      
        print('num of radiologists',len(exam_Annotations))   
    print('')
    return exam_Annotations

def extration_Folder(src_Path,output_Path,compress_Exam,output_Annotation_Format):
    folders_List = glob.glob(src_Path+"/*")
    num_of_Folders = len(folders_List)
    
    ## Runs through all folders in the 'folders_List'.
    for idx, path_Folder in enumerate(folders_List):
        
        ## Validades the Path Folder identifying the folder with more images. 
        exam_Path = validate_Folder_Path(path_Folder)
        exam_Name = path_Folder.split('/')[-1]

        print(idx,'of',num_of_Folders,'|',exam_Name)

        ## Transforms the Exam into an Array ('Mat') 
        exam_Arr,list_Of_Idx_Z = exam_2_Mat(exam_Path)
        
        ## Check Shape of Exam
        if(exam_Arr.shape[2] != 512 or exam_Arr.shape[1] != 512 or exam_Arr.shape[0] < 10):
            print(exam_Arr.shape)
            continue
        
        ## Read the XML Annotations and add all Nodules Centroids into a List.
        annotations = XML_Parser(exam_Path,list_Of_Idx_Z)
        
        if (len(annotations) > 0): 
            
            ## Create Directory to export exam and annotation
            save_Folder_Path = output_Path+exam_Name
            os.makedirs(save_Folder_Path, exist_ok=True)
            
            ## Export Exam
            if(compress_Exam == True):
                np.savez_compressed(save_Folder_Path+'/'+exam_Name+'.npz',exam_Arr)
            else:
                np.save(save_Folder_Path+'/'+exam_Name+'.npy',exam_Arr)

            ## Export Annotations
            if(output_Annotation_Format is 'txt'):
                np.savetxt(save_Folder_Path+'/'+exam_Name+'_annotations.txt',annotations,fmt='%s')
            else: 
                np.save(save_Folder_Path+'/'+exam_Name+'_annotations.npy',annotations)
                
    return True

def main():
    src_Path = "/home/raul/PROJECTS/LIDC-IDRI/Src_Folder"
    output_Path = "/home/raul/PROJECTS/LIDC-IDRI/Src_Folder_PYTHON/"
    compress_Exam = False
    output_Annotation_Format = 'npy'
    extration_Folder(src_Path,output_Path,compress_Exam,output_Annotation_Format)
    return True

main()