### **Face Detection using Haar Cascade Classifiers**<br>

<br>

1. Install Python
2. Run `pip install virtualenv`
3. Run `make-env.bat` or `make-env-3.9.bat`
4. Input Path &nbsp;--> `input`
5. Output Path --> `output`
6. Draws the **face** box with a `green` border and **eye** box with `blue` border

**Arguments**

1. `--mode | -m` &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - *image* or *video* or *realtime*
2. `--model | -mo` &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - *face* or *eye* (Eye makes use of face first to extract ROI)
3. `--filename | -f` &nbsp;&nbsp;&nbsp; - Name of the image file (with extension)
4. `--downscale | -ds` - Downscale video by a factor before inference 
5. `--save | -s` &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - Save the processed file (`NotImplemented`)