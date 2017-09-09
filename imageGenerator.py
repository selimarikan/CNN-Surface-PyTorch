from PIL import Image, ImageDraw
import numpy as np

def CreateWhiteImage(width, height):
    npWhiteImg = np.zeros((height, width, 3))
    npWhiteImg += 255
    return Image.fromarray(npWhiteImg.astype(np.uint8), mode='RGB')

if __name__ == '__main__':

    # Configuration parameters
    fillColor = (255,255,0,255)
    imageWidth = 720
    imageHeight = 128

    # Generation parameters
    generateCount = 120
    lineWidthRange = [2, 6]
    lineCountRange = [0, 2]
    ellipseRadiusRange = [3, 18]
    ellipseCountRange = [0, 2]
    arcRadiusRange = [2, 6]
    arcAngleRange = [0, 360]
    arcCountRange = [1, 3]

    pilWhiteImg = CreateWhiteImage(imageWidth * 2, imageHeight)

    for i in range(generateCount):
        # Create image
        npImg = np.zeros((imageHeight, imageWidth, 3))

        # Add green defect-free background
        npImg[:,:,1] += 255
    
        # Convert np image to PIL
        pilImg = Image.fromarray(npImg.astype(np.uint8), mode='RGB')

        # Add defects on image 
        # !! PIL uses TL corner as origin !!
        draw = ImageDraw.Draw(pilImg) 

        # Draw all lines
        for j in range(np.amax(lineCountRange)):
            startEndX = np.random.randint(imageWidth, size=2)
            startEndY = np.random.randint(imageHeight, size=2)
            width = np.random.randint(low=np.amin(lineWidthRange), high=np.amax(lineWidthRange))

            draw.line(( np.amin(startEndX),
                        np.amin(startEndY),
                        np.amax(startEndX),
                        np.amax(startEndY)), fill=fillColor, width=width)

        # Draw all ellipses
        for j in range(np.amax(ellipseCountRange)):
            startEndX = np.random.randint(imageWidth, size=2)
            startEndX[1] = startEndX[0] + np.random.randint(low=np.amin(ellipseRadiusRange), high=np.amax(ellipseRadiusRange))
            startEndY = np.random.randint(imageHeight, size=2)
            startEndY[1] = startEndY[0] + np.random.randint(low=np.amin(ellipseRadiusRange), high=np.amax(ellipseRadiusRange))

            draw.ellipse((  np.amin(startEndX),
                            np.amin(startEndY),
                            np.amax(startEndX),
                            np.amax(startEndY)), fill=fillColor)

        # Draw all arcs
        for j in range(np.amax(arcCountRange)):
            startEndX = np.random.randint(imageWidth, size=2)
            startEndX[1] = startEndX[0] + np.random.randint(low=np.amin(arcRadiusRange), high=np.amax(arcRadiusRange))
            startEndY = np.random.randint(imageHeight, size=2)
            startEndY[1] = startEndY[0] + np.random.randint(low=np.amin(arcRadiusRange), high=np.amax(arcRadiusRange))
            startAngle = np.random.randint(low=np.amin(arcAngleRange), high=np.amax(arcAngleRange))
            endAngle = np.random.randint(low=np.amin(arcAngleRange), high=np.amax(arcAngleRange))

            draw.chord((  np.amin(startEndX),
                          np.amin(startEndY),
                          np.amax(startEndX),
                          np.amax(startEndY)), startAngle, endAngle, fill=fillColor)

        # Paste generated image into white image
        pilWhiteImg.paste(pilImg, (imageWidth, 0))

        # Convert and save
        pilWhiteImg.save('generated/genImage_' + str(i + 1000) + '.png')
