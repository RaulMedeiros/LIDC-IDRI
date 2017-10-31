import dicom
import os
import glob
import numpy as np
import xml.etree.ElementTree as ET
np.set_printoptions(threshold=np.nan)

def isNotCloseToOthers(nodules_list,distMin):         
    filtered_Nodules = []
    for nodule in nodules_list:
        isTooClose =[0]
        pointA = np.array(nodule[0][-3:])
        
        for fnode in filtered_Nodules:
            pointB = np.array(fnode[0][-3:])
            
            if (np.linalg.norm(pointA-pointB) < distMin):
                isTooClose.append(1)
                    
        if (np.sum(isTooClose) == 0):
            filtered_Nodules.append(nodule)
            
    return filtered_Nodules

def mergeNodules(nodules_list,distMin, min_anotations = 0):      
    
#    print('nodules_list',np.array(nodules_list))
    filtered_Nodules = []
    for idxA, nodule_A in enumerate(nodules_list):
        
        distances = np.array([int(np.linalg.norm(np.array(nodule_A[0])-np.array(nodule_B[0]))) for nodule_B in nodules_list])        
        annotations2Merg = np.array([nodules_list[idxB] for idxB, dist in enumerate(distances) if dist <= distMin])         
#        print(distances)
        
        if(len(annotations2Merg) < min_anotations ):
            print('nope')
            continue
        else:            
            #TODO: Del merged elements from nodules_list
            centroid = np.mean(annotations2Merg[:,0])
            malignancy = round(np.mean(annotations2Merg[:,1]), 3)
            filtered_Nodules.append([*centroid,malignancy,annotations2Merg[0,2]])
            
#    print('before',np.array(filtered_Nodules))
    filtered_Nodules = set(map(tuple, filtered_Nodules))    
#    print('\nfiltered_Nodules',np.array(filtered_Nodules))

    return filtered_Nodules

#Validades the Path Folder identifying the folder with more images. 
def validateFolderPath(pathFolder):
    # Calculates the number of files in each sub folder
    lenFiles =[]
    for scanpathFolder in glob.glob(pathFolder+"/*/*"):
        lenFiles.append([len(glob.glob(scanpathFolder+"/*")),scanpathFolder]) 
    # Choses the folder with more elements and return its path            
    return max(lenFiles)[1]

def dicom_List_2_Array(lstFilesDCM):
    
    # Get ref file    
    RefDs = dicom.read_file(lstFilesDCM[0])
    RescaleIntercept = int(RefDs.data_element('RescaleIntercept').value)    
    
    exam = []
#     # Load dimensions based on the number of rows, columns, and slices (along the Z axis)
#     ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))

#     # Load spacing values (in mm)
#     ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))

#     # The array is sized based on 'ConstPixelDims'
#     ArrayDicom = np.zeros(ConstPixelDims, dtype=np.int16)

    # loop through all the DICOM files
    for idx, filenameDCM in enumerate(lstFilesDCM):
        # read the file and rescale to hounsfield units
        pixel_array = np.array(dicom.read_file(filenameDCM).pixel_array,dtype=np.int16)+RescaleIntercept
        # store the raw image data
#         ArrayDicom[:, :, idx] = pixel_array
        exam.append(pixel_array)
    
#     return np.transpose(ArrayDicom)
    return np.array(exam)

def exam2Mat(ExamPath):
    # Runs through all files in the 'ExamPath+"/*"' folder
    dicomPaths = []
    for pathFiles in glob.glob(ExamPath+'/*'):
        if (".dcm" in pathFiles): 
            ds = dicom.read_file(pathFiles)
            #Stores the z index and the respective path
            dicomPaths.append([float(ds.ImagePositionPatient[2]),pathFiles])

    #Sorts the List of Dicoms by Z index 
    dicomPaths=sorted(dicomPaths,key=lambda x: x[0])

    #creates a List of the slices pixels
    examMat = dicom_List_2_Array([i[1] for i in dicomPaths])
    listZidx=[item[0] for item in dicomPaths]

    return examMat, listZidx

# Extract List of Rois
def XMLReaderCentroidsExtractor(ExamPath,listOfIdxZ,distMin = 10):
    #Variables
    listOfRoi = []
   
    # Finds a xml file into the ExamPath folder       
    for xmlPath in glob.glob(ExamPath+"/*.xml"):   
        tree = ET.parse(xmlPath)
        root = tree.getroot()
        
        listOfRoi_Unchecked =[]
        # Looks for All Radiologists  
        for readingSession in root.iter('{http://www.nih.gov}readingSession'):    
        
            # Looks for All Annotations(Nodules) of a given radiologists 
            for unblindedReadNodule in readingSession.iter('{http://www.nih.gov}unblindedReadNodule'):  
            
                # Checks if there are a Malignancy into this annotations(Object/Nodules)
                if (unblindedReadNodule[1].text != None and unblindedReadNodule[1].tag == '{http://www.nih.gov}characteristics' ):

                    malignancy=int(unblindedReadNodule[1][8].text) #Malignancy
                    
                    roi_slices_list = []
                    for roi in unblindedReadNodule.iter('{http://www.nih.gov}roi'): 
                        if(roi[2].text == "TRUE"):
                            try:
                                # Load The annotation poits
                                xList = []; yList = []
                                for edgeMap in roi.iter('{http://www.nih.gov}edgeMap'):       
                                    xList.append(int(edgeMap[0].text))     
                                    yList.append(int(edgeMap[1].text))   

                                # Calculates the Center and Area of the a given nodule
                                xC = ((max(xList)-min(xList))/2 )+min(xList)
                                yC = ((max(yList)-min(yList))/2 )+min(yList)                       
                                zC = listOfIdxZ.index(float(roi[0].text))            
                                centroid = np.array([np.int(xC),np.int(yC),np.int(zC)])
                                            
                                roi_slices_list.append([centroid,malignancy,ExamPath[len(srcPath)+1:len(srcPath)+15]])                                                                                
                            except:
                                print('error',ExamPath)
                                continue                  
                    
                    # Selects only the central slice of the Nodule
                    listOfRoi_Unchecked.append(roi_slices_list[len(roi_slices_list)//2])
#                    listOfRoi_Unchecked.extend(roi_slices_list)

            
        # CHECK IF SOME NODULE TO CLOSE DO THIS 
#        listOfRoi.extend(isNotCloseToOthers(listOfRoi_Unchecked,distMin))
        listOfRoi.extend(mergeNodules(listOfRoi_Unchecked,distMin))
        
    return listOfRoi


def computeROI(x,y,z,r):
    return np.array([int(z-r[0]),int(z+r[0]+1),
                    int(y-r[1]),int(y+r[1]+1),
                    int(x-r[2]),int(x+r[2]+1)])


# Extract List of Rois
def extractNodules(examArr,NoduleCenter,roiRadius):
    roiShape = computeROI(*NoduleCenter,roiRadius)
    img = np.array(examArr[roiShape[0]:roiShape[1]-1,roiShape[2]:roiShape[3]-1,roiShape[4]:roiShape[5]-1],dtype=np.int16)    
    return img

def exportNodulesImgs(img,roi,roiRadius,outputPath,outputExtention,planes2print):    
    centerPoint = str(roi[0:3])
    classID = str(roi[-2])
    ExamName = roi[-1]
    malignancy = str(roi[3])
#    print(centerPoint,classID,ExamName)
        
    imgName = ExamName+'_'+centerPoint+'_'+malignancy
    
    for plane in planes2print:
        OutImg = []
        if(plane is 'Axial'):
            OutImg = img[roiRadius[0]-1,:,:]
        if(plane is 'Coronal'):
            OutImg = img[:,roiRadius[1]-1,:]
        if(plane is 'Sagittal'):
            OutImg = img[:,:,roiRadius[2]-1]
        if(plane is 'Full'):
            OutImg = img
        if(plane is '3Axis'):
            OutImg = np.array([img[roiRadius[0]-1,:,:],
                               img[:,roiRadius[1]-1,:],
                               img[:,:,roiRadius[2]-1]]).T

            
        for outExt in outputExtention:
            path = outputPath +'/'+outExt+'/'+plane+'/'
            os.makedirs(path, exist_ok=True)
                     
            if(outExt is 'npy'):
                np.save(path+imgName+'.npy',OutImg)
            if(outExt is 'txt'):
                np.savetxt(path+imgName+'.txt',OutImg,fmt='%s')

    return True

def extration_Folder(srcPath,outputPath,outputExtention,planes2print,roiSize):
    #Log Variables 
    cont=0

    # Runs through all folders in the 'PathName' path.
    for pathFolder in glob.glob(srcPath+"/*"): 

        cont+=1
        print(cont)
        
        #Validades the Path Folder identifying the folder with more images. 
        ExamPath=validateFolderPath(pathFolder)
        
        #Transforms the Exam into an Array ('Mat') 
        examArr,listOfIdxZ = exam2Mat(ExamPath)
        
        if(examArr.shape[2] != 512 or examArr.shape[1] != 512 or examArr.shape[0] <=1):
            print(examArr.shape)
            continue
       
        #Read the XML Annotations and add all Nodules Centroids into a List.
        for roi in XMLReaderCentroidsExtractor(ExamPath,listOfIdxZ):
        
            roi = list(roi)
            xyzCenterPoint = roi[0:3]
            try:    
                # Returs an array with the requested ROI/VOI pixels/Voxels
                img = extractNodules(examArr,xyzCenterPoint,roiRadius) 

            except:
                print('Extraction Fail: \n Exam = ',ExamPath,'| ROI =',roi)
                continue
            
            #Assert
            if(img.shape != (roiSize[0]*2, roiSize[1]*2, roiSize[2]*2)):
                print("Error img shape in the wrong format,img.shape = ",img.shape)
                continue
            
            #Export Nodule Images (txt or npt) 
            exportNodulesImgs(img,roi,roiSize,outputPath,outputExtention,planes2print)

    return True

# In[3]:

srcPath = "D:/MESTRADO/Bases"
outputPath = "D:/MESTRADO/Out_Folder/32/"

outputExtention = ['npy'] # ['npy','txt']

planes2print = ['Axial','Coronal','Sagittal','3Axis']
# planes2print = ['Axial']

roiRadius = [16,16,16]
                      
extration_Folder(srcPath,outputPath,outputExtention,planes2print,roiRadius)
