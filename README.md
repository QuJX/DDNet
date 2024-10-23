# Double Domain Guided Real-Time Low-Light Image Enhancement for Ultra-High-Definition Transportation Surveillance


## 1. Requirement ##
* __Python__ == 3.7
* __Torch__ == 1.12.0

## 2. Test platform
* The experimental computational device is a PC with an AMD EPYC 7543 32-Core Processor CPU accelerated by an Nvidia A40 GPU, which is also widely used in industrial-grade servers (e.g., Advantech SKY-6000 series and Thinkmate GPX servers).

## 3. Test
* Put the test images into the input floder
* Run test.py
* The results will be saved into the output floder.
* For the time testing, the inference time is tested by the test code ending time - test code starting time.
* To calculate the exact cuda ending time, you should add the 'torch.cuda.synchronize()' before the ending time recorded. (Thanks for the reminder from @CuddleSabe)

## 4. Downloads
* The checkpoint and UHD test data are available at: https://pan.baidu.com/s/1LGc7ox7QyLIdEAahmwYtxg  Pass code：mipc 
* Google Drive: https://drive.google.com/file/d/1X18X50iMKfRrGgrr1PT6tE8_ubqTAnpx/view?usp=drive_link

* PS: Due to our personal reasons (Graduation and changing computers), I have connected with my coorperator and we can't find the training code. We are sorry about that. 
