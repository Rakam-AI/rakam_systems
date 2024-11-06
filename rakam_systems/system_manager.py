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
            # Capture server response text if available for debugging
            error_message = f"Error executing {function_name} on {component_name}: {e} | URL: {url} | response: {response}"
            if response is not None and response.text:
                error_message += f" | Server response: {response.text}"
            raise Exception(error_message)

    def _get_component_url(self, component_name: str, function_name: str):
        """Generate the URL for the specified component and function within server groups."""
        base_url = self.config_data.get('base_url', 'http://localhost:8000/api')

        # Locate the server group and component
        for server_group in self.config_data.get('ServerGroups', []):
            for component in server_group.get('components', []):
                # Check if component_name exists within this component dictionary
                if component_name in component:
                    functions = component[component_name]
                    if function_name in functions:
                        endpoint_path = functions[function_name]
                        return f"{base_url}/{endpoint_path}"
        
        # Raise an error if component or function is not found
        raise ValueError(f"Function '{function_name}' for component '{component_name}' not found in configuration.")  

