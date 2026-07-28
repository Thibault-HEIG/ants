import os
import json
from datetime import datetime

SAVES_DIR = "saves"

def convert_old_format(save_data):
    if not isinstance(save_data, list):
        return None
    
    # Old format
    notes = ""
    genomes_by_species = {}
    
    for item in save_data:
        if "notes_to_self" in item:
            notes = item["notes_to_self"]
        elif "species_name" in item and "genome" in item:
            sp_name = item["species_name"]
            if sp_name not in genomes_by_species:
                genomes_by_species[sp_name] = []
            genomes_by_species[sp_name].append(item["genome"])
            
    # New format
    new_data = {
        "notes": notes,
        "timestamp": datetime.now().isoformat(),
        "round_time": 0.0,
        "generation_counts": {},
        "species": {},
        "history": {
            "time": [],
            "fitness": {}
        },
        "constants_snapshot": {},
        "stats": {}
    }
    
    for sp_name, genomes in genomes_by_species.items():
        new_data["species"][sp_name] = {
            "genomes": genomes,
            "config": {}
        }
        
    return new_data

def main():
    if not os.path.exists(SAVES_DIR):
        print(f"Directory {SAVES_DIR} does not exist.")
        return
        
    for filename in os.listdir(SAVES_DIR):
        if not filename.endswith(".json"):
            continue
            
        filepath = os.path.join(SAVES_DIR, filename)
        
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
                
            if isinstance(data, list):
                print(f"Converting {filename} to new format...")
                new_data = convert_old_format(data)
                if new_data is not None:
                    with open(filepath, "w") as f:
                        json.dump(new_data, f, indent=2)
                    print(f"Successfully converted {filename}")
            else:
                print(f"Skipping {filename} - already in new format")
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    main()
