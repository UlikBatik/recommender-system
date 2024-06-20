# Recommender System API Documentation 🧑‍💻

## API URL 🔗

[UlikBatik API]

```http
  http://localhost:5000/
```

## How to run this API on your local machine 💻

If you want to run this API Server on your local machine, you need to do this steps:

- First, clone this repository. `git clone https://github.com/UlikBatik/recommender-system.git`
- Second, open terminal and go to this project's root directory.
- Third, type `pip install -r requirements.txt` in your terminal and hit enter button.
- Fourth, start xampp.
- fifth, create `.env` file for your configuration.
- sixth, type `python app.py` in your terminal and hit enter button.
- Finally, the server link will be display on yout termina;

# API Endpoints

## Recommender System Endpoints

### Recommend

Get POSTID data for user recommendation based on user likes

```http
  POST /recommend
```

Register account for user


| Parameter | Type     | Description   |
| :---------- | :--------- | :-------------- |
| `userid`  | `string` | **Required**. |

#### Response

```http
  }
  "status": "true",
  "message": "This is your post recommendations",
   "data": {
        "POSTID": ,
    },
   }
```
