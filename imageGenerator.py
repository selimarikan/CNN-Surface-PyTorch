from PIL import Image, ImageDraw
import numpy as np

# Configuration parameters
fillColor = (255,255,0,255)
imageWidth = 720
imageHeight = 128

# Generation parameters
generateCount = 120
lineWidthRange = [2, 6]
lineCountRange = [1, 4]

# White image
npWhiteImg = np.zeros((imageHeight, imageWidth * 2, 3))
npWhiteImg += 255
pilWhiteImg = Image.fromarray(npWhiteImg.astype(np.uint8), mode='RGB') 

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

    # Paste generated image into white image
    pilWhiteImg.paste(pilImg, (imageWidth, 0))

    # Convert and save
    pilWhiteImg.save('generated/genImage_' + str(i + 1000) + '.png')
