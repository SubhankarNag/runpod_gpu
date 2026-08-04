"""
Configuration Parser Module
Safely loads .cfg files and converts string representations of lists, booleans, 
and numbers into native Python objects.
"""
import configparser
import ast

def load_config(config_path="config.cfg"):
    """
    Reads the configuration file and returns a nested dictionary.
    """
    config = configparser.ConfigParser()
    config.read(config_path)
    
    parsed_config = {}
    for section in config.sections():
        parsed_config[section] = {}
        for key, val in config.items(section):
            try:
                # Attempt to parse as a Python datatype (int, float, list, bool)
                parsed_val = ast.literal_eval(val)
            except (ValueError, SyntaxError):
                # If parsing fails, keep it as a standard string
                parsed_val = val
            parsed_config[section][key] = parsed_val
            
    return parsed_config