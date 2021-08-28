Face and Eye Detection using Haar Cascade Classifiers

### **CLI Arguments:**
<pre>
1. --image     : Flag that controls entry to perform detection on an image
2. --video     : Flag that controls entry to perform detection on a video file
3. --realtime  : Flag that controls entry to perform realtime detection
4. --mode      : Mode in which to operate (Face or Eye)
5. --name      : Name of the file (Used when --image or --video is set)
6. --downscale : Used to downscale the video file (Useful for display purposes)
</pre>

Needs --image, --video or --realtime

&nbsp;

**Notes**:

Add pyinstaller command to build a .exe