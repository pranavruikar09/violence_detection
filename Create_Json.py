import json
import datetime

# Current_time = datetime.datetime.now()
Current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
def Create_JSON(Current_Location):
    print(Current_time)
    # create a dictionary with the data
    data = {
        "Location": Current_Location,
        "Time and Date": str(Current_time),
        "Priority": "Low",
        "Crime": "Assault"
    }

    # write the data to a JSON file
    with open("data.json", "w") as json_file:
        json.dump(data, json_file)
