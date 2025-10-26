# this is the type dictionary for the project
type_dict = {
    "string": str,
    "integer": int,
    "float": float,
    "boolean": bool,
    "list": list,
    "dictionary": dict,
    "tuple": tuple,
    "set": set,
    "none": None,
    "any": None,
}

# this is the function to get the type of the variable
def get_type(variable):
    if variable in type_dict:   
        return type_dict[variable]
    else:
        raise ValueError(f"Invalid type: {variable}")

# this is the function to set the type of the variable
def set_type(variable, type_name):
    if type in type_dict:
        type_dict[variable] = type
    else:
        raise ValueError(f"Invalid type: {type}")