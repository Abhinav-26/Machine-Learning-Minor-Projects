IMAGE TO PENCIL SKETCH WITH PYTHON :-

We need to read the image in RGB format and then convert it into a grayscale image.
This will turn an image into a classic black and white photo.
Then the next thing to do is invert the grayscale image also called negative image,this will be our inverted grayscale image.
Inversion can be used to enhance the details.
Then we can finally create the pencilsketch by mixing the grayscale image with inverted blurry image. 
This can be done by dividing the grayscale image by the inverted blurry image.
We will need the OpenCv library of python to do all this stuff.