# Smart Engagement Room

This project proposes the Smart Engagement Room, a novel classroom environment that leverages Ambient Intelligence (AmI) to create a more dynamic and interactive learning experience.

The Smart Engagement Room addresses limitations of traditional classrooms by:
- Automated Attendance: Replacing manual attendance checks with a real-time computer vision system.
- Dynamic Lighting: Adjusting lighting based on student location for energy efficiency and learning environment optimization.
- Posture & Gesture Detection: Utilizing AmI to detect raised hands and student posture, allowing instructors to identify student inquiries and tailor their teaching approach.

By integrating AmI, the Smart Engagement Room fosters a more interactive and engaging learning environment for both students and instructors.

## Setup

### Web Application

The Website can be accessed here: [Smart Engagement Room](https://smart-engagement-room.vercel.app/).

To authenticate in the website you can use the following credentials:

```
Username: grupo6
Password: ami
```

It is composed by 4 Webpages:
- Home | Dashboard - Page with the statistics to be shown to the professor after a lecture.
- Attendance - Page with the attedances or absences of students.
- Classroom Layout - Page with the representation of the classroom layout and with reactive lights
and hands gesture detection.
- Admin - This page is used for development and presentation purposes. In real world it should not be visible.

The Website is hosted online in the [Vercel Platform](https://vercel.com).
It communicates with a database hosted in a [Mongo Atlas Clust](https://www.mongodb.com/atlas),
that stores the statistics to be shown in the website.

#### Run Locally

To run the website in your local machine you should first install [NodeJS](https://nodejs.org/en) and [npmJS](https://www.npmjs.com/).

To run the website:

```bash
npm run start
```
The database connection is already setup with no interventation needed.

### Computer Vision

The computer vision repository handles the grpc connection between the raspberry pi and the computer that handles and calculates all the data from the frames sent by the raspberry pi.

The raspberry pi will be the grpc client (ambint_client.py) that captures images from the camera and sends to the computer (ambint_server.py)

#### Inside the Server

- Reinstall and install packages
    ```s
    pip uninstall requests_toolbelt urllib3
    pip install -r requirements.txt
    ```

- Install the head and hand models with the instructions in `models` folder

- Install and start CompreFace in docker locally (follow the tutorial available in their [github repo](https://github.com/exadel-inc/CompreFace?tab=readme-ov-file#getting-started-with-compreface))

    - When the CompreFace is running, you will need to create an account, create a project, create the subjects and paste the api key inside `ambint_compreface.py`

- Initialize grpc server
    ```s
    python3 ambint_server.py
    ```
- To switch between student identification and head and hand detection, hold the `s` key until the grid appears


#### Inside the raspberry pi:

- Create a virtual python environment to install the necessary libraries
    ```s
    python -m venv ambint-env
    ```

- Source to created environment
    ```s
    source ambint-env/bin/activate
    ```

- Install the libraries necessary via pip (`opencv-python` and `grpcio-tools`)
    ```s
    pip install opencv-python grpcio-tools
    ```

- Initialize grpc client (make sure that you have the webcam connected to the rasberry pi)
    ```s
    python3 ambint_client.py
    ```

### Arduino and Raspberry (Communication using MQTT)

Before you begin:

- Install the required libraries on your Raspberry Pi:

```s
sudo apt install mosquitto mosquitto-clients
```

Test the installation by running the following command:

```s
mosquitto -v
```

Then you should edit the mosquitto configuration file:

```s
sudo nano /etc/mosquitto/mosquitto.conf
```

Add the following lines:

```s
allow_anonymous true
listener 1883
```

Then restart Mosquitto for the changes to take effect:

```s
sudo systemctl restart mosquitto
```

On your arduino, make sure you have the following libraries installed:
- WiFi.h
- PubSubClient.h

Board compatibility:
- Arduino Nano ESP32

The arduino and the other components (RCWL-0516 and wires) should be assemble as the image below:

![Alt text](https://cdn.discordapp.com/attachments/1047456846409117726/1219968945772167308/electric-scheme.png?ex=660d3b9e&is=65fac69e&hm=aee7943c9cc014c79b1c4792c99a840914f8871f669c1e55b489e86f649447be&)

Then before doing changes in the code for the MQTT communication you should check for the ip address of the raspberry. For that, use the following command:

```s
hostname -I
```

Then you should do the following changes in the code:

- Of Arduino (arduino-mqtt.ino):

```s
const char* ssid = "<YOUR_WIFI_SSIS>";
const char* passwor = "<YOUR_WIFI_PASSWORD>";

const char* mqtt_server = "<RASPBERRYPI_IP>";
```

- Of RaspberryPi (subscriber.py):

```s
broker = '<RASPBERRYPI_IP'
```

After these steps, you can test the communication:
1. Power on the Arduino and upload the code.
2. Run the Python script on the Raspberry Pi.

Additional Notes:
- Replace placeholder values like <YOUR_WIFI_SSID> and <RASPBERRYPI_IP> with your actual credentials and IP address.
