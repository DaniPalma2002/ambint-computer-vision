from ast import List
import json
from compreface import CompreFace
from compreface.service import RecognitionService

def identify_students():
    DOMAIN: str = 'http://localhost'
    PORT: str = '8000'
    RECOGNITION_API_KEY: str = 'f7aec73a-6da8-4835-ac1b-9e299dc7fc88'


    compre_face: CompreFace = CompreFace(DOMAIN, PORT, {
        "limit": 0,
        "det_prob_threshold": 0.8,
        "prediction_count": 1,
        "status": "true"
    })

    recognition: RecognitionService = compre_face.init_face_recognition(
        RECOGNITION_API_KEY)

    image_path: str = 'images/frame.jpg'

    result = recognition.recognize(image_path)

    json_res = json.dumps(result, indent=4, sort_keys=True)
    print(json_res)

    list_of_students: List = []
    if 'code' in result:
        return list_of_students
    for i in range(len(result['result'])):
        #print(result['result'][i]['subjects'][0]['subject'])
        list_of_students.append(result['result'][i]['subjects'][0]['subject'])

    return list_of_students

    
if __name__ == "__main__":
    identify_students()
