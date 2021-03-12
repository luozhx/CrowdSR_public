# CrowdSR

This is part of the source code of paper CrowdSR: Enabling High-Quality Video Ingest in Crowdsourced Livecast via Super-Resolution.

It just uses for demonstrating the procedure of the system.


## Run
0. Requirements:
    * pytorch
    * aiortc 
    * aiohttp
    * requests
    * opencv-contrib-python
    * scikit-image
    

1. Prepare data 
   
   First, preparing a high resolution video(1920*1080) and a low resolution video(960x540) in work directory.
   
   Second, put the high resolution frames into a dedicated directory for every other video and put them into a directory.
   For example, the directory should like below.
   ```
   candidates
   |--broadcaster1
      |--0001.png
      |--0002.png
      ....
   |--broadcaster2
   ....
   ```

2. Start service
   
   Run the python scripts and add the option on your need.
   
   `python server.py`
   
   `python client.py`


## Citing

   TBD
