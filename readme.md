STEP 1 : Creation of TFR Dataset for the images
-----------------------------------------------
STEP a: Put images in this directory : path/to/directory/images
STEP b: Put annotations in this directory : path/to/directory/annotations
STEP c: Before generating the TFR dataset change some paths in the 'config.ini' file, run the following command:
        1. nano path/to/config/config.ini
        If you want to generate multiple files from one image file then configure the follwing paths:
        IMAGE (name of the image without extension)
        IMAGE_TYPE (type of the image i.e jpg)
STEP d: Run the following python script to generate the TFR dataset:
          python3 path/to/directory/DriverScript.py config
        
        (a)If you want to generate multiple files from one image file then run the following python script:
          python3 path/to/directory/OneToManyImg.py config
        
          -Provide the input that the program requires,ex: enter 1 to rotate the image in different angles
        
        (b)If you want to rename a list of files in a specific format (img_<count of file>) then run the following script:
          python3 path/to/directory/RenamingImageFiles.py config
        
        (c)If you want to get the maximum and minimum height and width of list of images then run the follwing script:
          python3 path/to/directory/GettingMaxMinHeightWidth.py config

