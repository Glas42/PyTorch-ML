import src.variables as variables
import json


def EnsureFile(file:str):
    try:
        with open(file, "r") as f:
            try:
                json.load(f)
            except:
                with open(file, "w") as ff:
                    ff.write("{}")
    except:
        with open(file, "w") as f:
            f.write("{}")


def Get(name:str, value:any=None):
    try:
        EnsureFile(f"{variables.PATH}settings.json")
        with open(f"{variables.PATH}settings.json", "r") as f:
            settings = json.load(f)

        if settings[name] == None:
            return value
        
        return settings[name]
    except:
        if value != None:
            Set(name, value)
            return value
        else:
            pass


def Set(name:str, data:any):
    try:
        EnsureFile(f"{variables.PATH}settings.json")
        with open(f"{variables.PATH}settings.json", "r") as f:
            settings = json.load(f)

            settings[name] = data

        with open(f"{variables.PATH}settings.json", "w") as f:
            f.truncate(0)
            json.dump(settings, f, indent=6)
    except:
        pass