# SpectraVision
This repository is a POC for using object detection with cv2 and YOLOv11

### For References:

[Ultralytics Docs](https://docs.ultralytics.com/)

## How to use
### Setup 
```bash
python -m venv .venv
```
Mac:

```bash
python3 -m venv .venv
```

> [!NOTE]
> 
>Make sure you add it as you interpreter 

### Activate your environment

```bash
.\.venv/Scripts/activate
```
Mac:

```bash
source .venv/Scripts/activate
```

### Install requirements
```bash
pip install -r requirements.txt
```

### Start 
```bash
python main.py
```


### Docker Compose 

1. You need to have Docker Desktop
2. Then run 
    ```bash
    docker compose up -d 
    ```
3. Now Start the Project

> [!NOTE]
>
>You can open the grafana Dashboard on http://localhost:3000/d/EaSfNcTHk/object-count?orgId=1&from=now-5m&to=now Username: admin PW: admin