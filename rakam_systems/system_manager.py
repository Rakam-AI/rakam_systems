import requests
import yaml

class SystemManager:
    def __init__(self, system_config_path: str):
        # Load system configuration from YAML file
        self.system_config_path = system_config_path
        self.config_data = self._load_system_config()

    def _load_system_config(self):
        """Load and parse the system configuration from YAML."""
        try:
            with open(self.system_config_path, 'r') as config_file:
                return yaml.safe_load(config_file)
        except FileNotFoundError:
            raise Exception(f"System configuration file '{self.system_config_path}' not found.")
        except yaml.YAMLError as e:
            raise Exception(f"Error parsing YAML configuration: {e}")

    def execute_component_function(self, component_name: str, function_name: str, input_data: dict):
        """Execute a function for a given component with provided input data."""
        url = self._get_component_url(component_name, function_name)
        try:
            response = requests.post(url, json=input_data)
            response.raise_for_status()  # Raise an error for bad HTTP status
            return response.json()  # Return the response in JSON format
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error executing {function_name} on {component_name}: {e}")

    def _get_component_url(self, component_name: str, function_name: str):
        """Generate the URL for the specified component and function."""
        # Ensure base_url is retrieved from config or defaulted
        base_url = self.config_data.get('base_url', 'http://localhost')
        
        # Verify component and function existence
        if component_name not in self.config_data.get('components', {}):
            raise ValueError(f"Component '{component_name}' not found in configuration.")
        
        if function_name not in self.config_data['components'][component_name].get('functions', []):
            raise ValueError(f"Function '{function_name}' not found for component '{component_name}' in configuration.")
        
        # Construct full URL
        endpoint_path = self.config_data['components'][component_name]['functions'][function_name]
        url = f"{base_url}/{endpoint_path}"
        return url
