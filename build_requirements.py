import os
import yaml

# Path to the YAML configuration file
config_file = "application/rakam_systems/components_config.yaml"

# Load the YAML configuration
with open(config_file, "r") as f:
    components = yaml.safe_load(f)

# Create a directory for component requirements if it doesn't exist
output_dir = "application/rakam_systems/component_requirements"
os.makedirs(output_dir, exist_ok=True)

# Write requirements for each component
for component, details in components.items():
    requirements_file = os.path.join(output_dir, f"{component}_requirements.txt")
    with open(requirements_file, "w") as f:
        for library in details["libraries"]:
            f.write(f"{library}\n")
    print(f"Generated {requirements_file} with libraries: {', '.join(details['libraries'])}")

print(f"\nRequirements files for each component have been generated in '{output_dir}' directory.")
