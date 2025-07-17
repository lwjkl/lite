import requests

# api_url = "http://localhost:1234/index-images"

# payload = {
#     "image_directory": "/home/wjlee@merimenkl.com/Documents/temp",  # change to your image folder
#     "wildcard": "*.JPG",
#     "batch_size": 64
# }

# try:
#     response = requests.post(api_url, json=payload)
#     print("Status Code:", response.status_code)
#     print("Response JSON:", response.json())
# except Exception as e:
#     print("Error:", e)

api_url = "http://localhost:1234/search-similar"

# Path to your query image
query_image_path = "/home/wjlee@merimenkl.com/Documents/temp/5522035_243132101.JPG"  # ✅ change this

# Parameters
num_results = 5
download_ids = False  # Set to True if you want text file of IDs

# Prepare the multipart form data
with open(query_image_path, "rb") as f:
    files = {"file": f}
    data = {
        "num_results": num_results,
        "download_ids": download_ids
    }

    try:
        response = requests.post(api_url, files=files, data=data)
        print("Status Code:", response.status_code)
        if download_ids:
            # Save the text file with image IDs
            with open("similar_image_ids.txt", "wb") as out_file:
                out_file.write(response.content)
            print("Downloaded IDs saved to similar_image_ids.txt")
        else:
            # Print JSON response
            print("Response JSON:", response.json())
    except Exception as e:
        print("Error:", e)

