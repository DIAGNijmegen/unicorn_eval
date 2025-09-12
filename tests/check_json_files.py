import json
import os

base_dir = 'tests/vision/input-test-radiology-vision-d934e4b3-de1a-47f5-b494-c1ee5bde080f'
json_files = []

# Find all JSON files
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith('.json'):
            json_files.append(os.path.join(root, file))

print(f'Found {len(json_files)} JSON files')
print()

# Check each file for JSON decode errors
broken_files = []
for json_file in json_files:
    try:
        with open(json_file, 'r') as f:
            json.load(f)
    except json.JSONDecodeError as e:
        print(f'JSON DECODE ERROR in {json_file}:')
        print(f'  {e}')
        print()
        broken_files.append(json_file)
    except Exception as e:
        print(f'OTHER ERROR in {json_file}:')
        print(f'  {e}')
        print()

print(f'Summary: {len(broken_files)} files with JSON decode errors out of {len(json_files)} total files')
if broken_files:
    print('Files with JSON decode errors:')
    for f in broken_files:
        print(f'  {f}')