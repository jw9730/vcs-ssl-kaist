import json
import requests
from urllib.parse import urljoin
import base64
import cv2
import numpy as np

# I/O functions
def imageToString(img):
    return base64.b64encode(cv2.imencode('.jpg', img)[1]).decode()

def stringToImage(imagestring):
    data = base64.b64decode(imagestring)
    jpg_arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(jpg_arr, cv2.IMREAD_COLOR)

def generateDummySceneGraph():
    return {                
        "image_id":1,
        "relationships":[
            {"synsets":["along.r.01"],"predicate":"ON","relationship_id":15927,"object_id":5046,"subject_id":5045},
            {"synsets":["wear.v.01"],"predicate":"wears","relationship_id":15928,"object_id":5048,"subject_id":1058529},
            {"synsets":["have.v.01"],"predicate":"has","relationship_id":15929,"object_id":5050,"subject_id":5049},
            {"synsets":["along.r.01"],"predicate":"ON","relationship_id":15930,"object_id":1058508,"subject_id":1058507},
            {"synsets":["along.r.01"],"predicate":"ON","relationship_id":15931,"object_id":1058534,"subject_id":5055},
            {"synsets":["have.v.01"],"predicate":"has","relationship_id":15932,"object_id":1058511,"subject_id":1058529},
            {"synsets":["next.r.01"],"predicate":"next to","relationship_id":15933,"object_id":1058539,"subject_id":1058534},
            {"synsets":["have.v.01"],"predicate":"has","relationship_id":15934,"object_id":5060,"subject_id":1058515},
            {"synsets":["have.v.01"],"predicate":"has","relationship_id":15935,"object_id":1058518,"subject_id":1058529},
            {"synsets":["along.r.01"],"predicate":"ON","relationship_id":15936,"object_id":1058534,"subject_id":1058519},
            {"synsets":["wear.v.01"],"predicate":"wears","relationship_id":15937,"object_id":5048,"subject_id":1058529},
            {"synsets":["have.v.01"],"predicate":"has","relationship_id":15938,"object_id":1058525,"subject_id":1058532},
            {"synsets":["have.v.01"],"predicate":"has","relationship_id":15939,"object_id":1058511,"subject_id":1058529},
            {"synsets":["wear.v.01"],"predicate":"wears","relationship_id":15940,"object_id":1058528,"subject_id":1058529},
            {"synsets":["have.v.01"],"predicate":"has","relationship_id":15941,"object_id":1058530,"subject_id":1058532},
            {"synsets":["have.v.01"],"predicate":"has","relationship_id":15942,"object_id":1058531,"subject_id":1058532},
            {"synsets":["along.r.01"],"predicate":"parked on","relationship_id":15943,"object_id":1058534,"subject_id":5051},
            {"synsets":["along.r.01"],"predicate":"parked on","relationship_id":15944,"object_id":1058534,"subject_id":1058535},
            {"synsets":["along.r.01"],"predicate":"parked on","relationship_id":15945,"object_id":1058539,"subject_id":1058536},
            {"synsets":["along.r.01"],"predicate":"parked on","relationship_id":15946,"object_id":1058539,"subject_id":1058515},
            {"synsets":["along.r.01"],"predicate":"ON","relationship_id":4265923,"object_id":3798575,"subject_id":1058535},
            {"synsets":["behind.r.01"],"predicate":"behind","relationship_id":3186256,"object_id":1058532,"subject_id":1058519},
            {"synsets":["have.v.01"],"predicate":"holding","relationship_id":3186257,"object_id":1058541,"subject_id":1058540},
            {"synsets":["wear.v.01"],"predicate":"WEARING","relationship_id":3186258,"object_id":1058511,"subject_id":1058529},
            {"synsets":["have.v.01"],"predicate":"holding","relationship_id":3186259,"object_id":1058541,"subject_id":1058532},
            {"synsets":["about.r.07"],"predicate":"near","relationship_id":3186260,"object_id":1058545,"subject_id":1058544},
            {"synsets":["wear.v.01"],"predicate":"WEARING","relationship_id":3186261,"object_id":1058525,"subject_id":1058532},
            {"synsets":["about.r.07"],"predicate":"near","relationship_id":3186262,"object_id":1058545,"subject_id":1058544},
            {"synsets":["along.r.01"],"predicate":"ON","relationship_id":3186263,"object_id":1058529,"subject_id":1058511},
            {"synsets":["have.v.01"],"predicate":"holding","relationship_id":4265924,"object_id":1058541,"subject_id":3798576},
            {"synsets":["wear.v.01"],"predicate":"WEARING","relationship_id":4265925,"object_id":1058518,"subject_id":3798577},
            {"synsets":["along.r.01"],"predicate":"along","relationship_id":4265926,"object_id":3798578,"subject_id":1058548},
            {"synsets":["in.r.01"],"predicate":"IN","relationship_id":3186264,"object_id":1058511,"subject_id":1058529},
            {"synsets":["wear.v.01"],"predicate":"WEARING","relationship_id":3186265,"object_id":1058528,"subject_id":1058529},
            {"synsets":["along.r.01"],"predicate":"on top of","relationship_id":3186266,"object_id":1058539,"subject_id":1058519},
            {"synsets":["next.r.01"],"predicate":"next to","relationship_id":3186267,"object_id":1058539,"subject_id":1058545},
            {"synsets":["wear.v.01"],"predicate":"WEARING","relationship_id":3186268,"object_id":1058518,"subject_id":1058529},
            {"synsets":["behind.r.01"],"predicate":"behind","relationship_id":3186269,"object_id":1058529,"subject_id":1058544},
            {"synsets":["by.r.01"],"predicate":"by","relationship_id":3186270,"object_id":1058534,"subject_id":1058549},
            {"synsets":["wear.v.01"],"predicate":"WEARING","relationship_id":3186271,"object_id":1058530,"subject_id":1058532},
            {"synsets":[],"predicate":"with","relationship_id":4265927,"object_id":3798579,"subject_id":1058508}],
        "objects":[
            {"synsets":["clock.n.01"],"h":339,"object_id":1058498,"names":["clock"],"w":79,"attributes":["green","tall"],"y":91,"x":421},
            {"synsets":["street.n.01"],"h":262,"object_id":5046,"names":["street"],"w":714,"attributes":["sidewalk"],"y":328,"x":77},
            {"synsets":["shade.n.01"],"h":192,"object_id":5045,"names":["shade"],"w":274,"y":338,"x":119},
            {"synsets":["man.n.01"],"h":262,"object_id":1058529,"names":["man"],"w":60,"y":249,"x":238},
            {"synsets":["gym_shoe.n.01"],"h":26,"object_id":5048,"names":["sneakers"],"w":52,"attributes":["grey"],"y":489,"x":243},
            {"synsets":["headlight.n.01"],"h":15,"object_id":5050,"names":["headlight"],"w":23,"attributes":["off"],"y":366,"x":514},
            {"synsets":["car.n.01"],"h":98,"object_id":5049,"names":["car"],"w":74,"y":315,"x":479},
            {"synsets":["bicycle.n.01"],"h":34,"object_id":5051,"names":["bike"],"w":28,"attributes":["parked","far away"],"y":319,"x":318},
            {"synsets":["bicycle.n.01"],"h":35,"object_id":1058535,"names":["bike"],"w":29,"attributes":["parked","far away","chained"],"y":319,"x":334},
            {"synsets":["sign.n.02"],"h":182,"object_id":1058507,"names":["sign"],"w":88,"attributes":["black"],"y":13,"x":118},
            {"synsets":["building.n.01"],"h":536,"object_id":1058508,"names":["building"],"w":218,"attributes":["tall","brick","made of bricks"],"y":2,"x":1},
            {"synsets":["trunk.n.01"],"h":327,"object_id":5055,"names":["tree trunk"],"w":87,"y":234,"x":622},
            {"synsets":["sidewalk.n.01"],"h":266,"object_id":1058534,"names":["sidewalk"],"w":722,"attributes":["brick"],"y":331,"x":77},
            {"synsets":["shirt.n.01"],"h":101,"object_id":1058511,"names":["shirt"],"w":59,"attributes":["red","orange"],"y":289,"x":241},
            {"synsets":["street.n.01"],"h":233,"object_id":1058539,"names":["street"],"w":440,"attributes":["clean"],"y":283,"x":358},
            {"synsets":["car.n.01"],"h":174,"object_id":1058515,"names":["car"],"w":91,"attributes":["white","parked"],"y":342,"x":708},
            {"synsets":["back.n.01"],"h":170,"object_id":5060,"names":["back"],"w":67,"y":339,"x":721},
            {"synsets":["spectacles.n.01"],"h":12,"object_id":1058518,"names":["glasses"],"w":20,"y":268,"x":271},
            {"synsets":["parking_meter.n.01"],"h":143,"object_id":1058519,"names":["parking meter"],"w":32,"attributes":["orange"],"y":327,"x":574},
            {"synsets":["shoe.n.01"],"h":34,"object_id":1058525,"names":["shoes"],"w":46,"attributes":["brown"],"y":481,"x":391},
            {"synsets":["man.n.01"],"h":251,"object_id":1058532,"names":["man"],"w":75,"y":264,"x":372},
            {"synsets":["trouser.n.01"],"h":118,"object_id":1058528,"names":["pants"],"w":38,"attributes":["black"],"y":384,"x":245},
            {"synsets":["jacket.n.01"],"h":97,"object_id":1058530,"names":["jacket"],"w":89,"attributes":["gray","grey"],"y":296,"x":356},
            {"synsets":["trouser.n.01"],"h":128,"object_id":1058531,"names":["pants"],"w":54,"attributes":["gray","grey"],"y":369,"x":382},
            {"synsets":[],"h":185,"object_id":1058536,"names":["work truck"],"w":265,"attributes":["white"],"y":271,"x":521},
            {"synsets":["sidewalk.n.01"],"h":189,"object_id":3798575,"names":["sidewalk"],"w":50,"y":318,"x":343},
            {"synsets":["chin.n.01"],"h":9,"object_id":1058541,"names":["chin"],"w":11,"attributes":["raised"],"y":288,"x":399},
            {"synsets":["guy.n.01"],"h":250,"object_id":1058540,"names":["guy"],"w":82,"y":264,"x":369},
            {"synsets":["van.n.05"],"h":134,"object_id":1058542,"names":["van"],"w":233,"attributes":["parked","white"],"y":298,"x":529},
            {"synsets":["wall.n.01"],"h":533,"object_id":1058543,"names":["wall"],"w":134,"attributes":["grey"],"y":1,"x":0},
            {"synsets":["tree.n.01"],"h":360,"object_id":1058545,"names":["tree"],"w":176,"y":0,"x":249},
            {"synsets":["bicycle.n.01"],"h":35,"object_id":1058544,"names":["bikes"],"w":40,"y":319,"x":321},
            {"synsets":["arm.n.01"],"h":43,"object_id":1058546,"names":["arm"],"w":32,"attributes":["raised"],"y":283,"x":368},
            {"synsets":["shirt.n.01"],"h":66,"object_id":1058547,"names":["shirt"],"w":37,"attributes":["grey"],"y":306,"x":384},
            {"synsets":["man.n.01"],"h":248,"object_id":3798576,"names":["man"],"w":97,"y":264,"x":362},
            {"synsets":["man.n.01"],"h":264,"object_id":3798577,"names":["man"],"w":72,"y":251,"x":230},
            {"synsets":["road.n.01"],"h":218,"object_id":3798578,"names":["road"],"w":340,"y":295,"x":435},
            {"synsets":[],"h":430,"object_id":1058548,"names":["lamp post"],"w":41,"y":63,"x":537},
            {"synsets":["tree.n.01"],"h":557,"object_id":1058549,"names":["trees"],"w":606,"attributes":["sparse"],"y":0,"x":190},
            {"synsets":["window.n.01"],"h":148,"object_id":3798579,"names":["windows"],"w":173,"y":4,"x":602}]
    }

# container url
URL = 'http://127.0.0.1:10032/'


# data load
img = cv2.imread('./images/1.jpg')
masked_img = cv2.imread('./images/1.png')
with open('./temp_results/part2_scenegraph_birmingham.json', 'r', encoding='utf-8') as jsonfile:
    json_data = json.load(jsonfile)
    assert 'masked_object' in json_data.keys(), "masked_object key is not in json_data!"

data = json.dumps(
    {
        'img': imageToString(img),
        'masked_img': imageToString(masked_img),        
        'json_data': json_data
    }
)

# run application
response = requests.post(urljoin(URL, '/api/run'), data=data)

print(response.status_code)
# print(response.request)
print(response.json())

with open('./temp_results/part3_scenegraph_kaist.json','w') as outfile:
    json_contents = response.json()
    json.dump(json_contents, outfile)
