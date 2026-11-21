* **CV1 – fața și ochii:**
  * `perception/face.py`: MediaPipe pentru față
  * `face/eyes.py`: EAR, PERCLOS și rata de clipire
  * `face/head_pose.py`: poziția capului și `gaze_on_road`
  * `face/unconscious.py`: cap căzut și ochi închiși prelungit
  * `face/calibration.py`: calibrarea din primele 30 s
  * `dashboard/`: dashboard-ul
* **CV2 – corp, mâini și obiecte:**
  * `camera.py`, `pipeline.py`, `publisher.py`: camera, bucla principală și trimiterea pe MQTT
  * `perception/body.py`, `perception/objects.py`: Pose, Hands și YOLO
  * `body/chest.py`: mâna pe piept
  * `body/breathing.py`: respirația grea
  * `body/phone.py`: telefonul
  * `events/tracker.py`: evenimentele cu durată și încredere
* **Comune:** `schemas.py`, `context.py`, `config.py` / `config.yaml` și `fake/fake_vision.py`.