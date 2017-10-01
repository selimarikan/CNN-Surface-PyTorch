import xml.etree.ElementTree as ET
from lxml import etree
from PIL import Image, ImageDraw
import numpy as np
import os

def CreateWhiteImage(width, height):
    npWhiteImg = np.zeros((height, width, 3))
    npWhiteImg += 255
    return Image.fromarray(npWhiteImg.astype(np.uint8), mode='RGB')

def LoadXMLFile(path):
    tree = etree.parse(path)
    root = tree.getroot()
    namespace = root.tag[0:(root.tag.index('}')+1)]
    return root, namespace

if __name__ == '__main__':
    baseDirectory = 'g:/Selim/Thesis/Code/'
    datasetName = 'Aviles MSoS'
    phase = 'test'

    # Configuration parameters
    fillColor = (255,255,0,255)
    imageWidth = 0
    imageHeight = 0
    
    datasetDirectory = os.path.join(baseDirectory, datasetName)
    xmlPath = os.path.join(datasetDirectory, phase + '/Category.ImportantDefects.xml')
    
    tree = etree.parse(xmlPath)
    root = tree.getroot()
    namespace = root.tag[0:(root.tag.index('}')+1)]
    
    defectsElement = root.findall('Defects', root.nsmap)[0]
    defectElements = defectsElement.getchildren()
    imageROIMap = []
    
    for defectElement in defectElements:
        imageName = defectElement.find('ImageFile', root.nsmap).text.replace('.tif', '.png')
        imagePath = os.path.join(datasetDirectory, phase + '/defect/' + imageName)
        im = Image.open(imagePath)
        imageWidth, imageHeight = im.size
        print(imagePath)
        
        # Create image
        npImg = np.zeros((imageHeight, imageWidth, 3))

        # Add green defect-free background
        npImg[:,:,1] += 255

        # Convert np image to PIL
        pilImg = Image.fromarray(npImg.astype(np.uint8), mode='RGB')
        draw = ImageDraw.Draw(pilImg)

        pilWhiteImg = CreateWhiteImage(imageWidth * 2, imageHeight)

        defectZoneElement = defectElement.find('DefectZone', root.nsmap).getchildren()
        defectCoordinates = []
        for defectZoneEdge in defectZoneElement:
            print(defectZoneEdge.attrib)
            defectCoordinates.append([defectZoneEdge.attrib['X'], defectZoneEdge.attrib['Y']])
            
        print(defectCoordinates)
        rect = draw.rectangle((int(defectCoordinates[0][0]), 
                               min(int(defectCoordinates[0][1]), int(defectCoordinates[2][1])),
                               int(defectCoordinates[2][0]),
                               max(int(defectCoordinates[0][1]), int(defectCoordinates[2][1]))), fill=fillColor)

        pilWhiteImg.paste(im, (0, 0))
        pilWhiteImg.paste(pilImg, (imageWidth, 0))

        saveFolder = os.path.join(os.path.join(datasetDirectory, phase), 'Generated')
        if not os.path.exists(saveFolder):
            os.makedirs(saveFolder)
        savePath = saveFolder + '/genImage_' + imageName + '.png'
        # Convert and save
        pilWhiteImg.save(savePath)
        #im = im.crop((int(defectCoordinates[0][0]), 
        #              min(int(defectCoordinates[0][1]), int(defectCoordinates[2][1])),
        #              int(defectCoordinates[2][0]),
        #              max(int(defectCoordinates[0][1]), int(defectCoordinates[2][1]))))
        #im.save(os.path.join(datasetDirectory, phase + '/extractedDefect/' + imageName.replace('.tif', '.png')))