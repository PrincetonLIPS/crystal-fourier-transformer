"""
Script to download crystal structures and properties from the Materials Project API.

Usage:
    1. Get an API key from https://materialsproject.org/
    2. Set your API key below or as environment variable MP_API_KEY
    3. Run: python -m data.get_materials

The script will download CIF files and create an id_prop.csv file in the specified output directory.
"""

import csv
import os
from pymatgen.ext.matproj import MPRester
from emmet.core.summary import HasProps


def download_materials_with_elasticity(api_key, root_dir, max_materials=None):
    """Download materials with elasticity data (bulk and shear modulus).
    
    Args:
        api_key: Materials Project API key
        root_dir: Directory to save CIF files and property CSV
        max_materials: Optional limit on number of materials to download
    """
    with MPRester(api_key) as mpr:
        print("Searching for materials with elasticity data...")
        mats = mpr.materials.summary.search(
            has_props=[HasProps.elasticity], 
            fields=["material_id", "bulk_modulus", "shear_modulus"]
        )
        
        if max_materials:
            mats = mats[:max_materials]
        
        os.makedirs(root_dir, exist_ok=True)
        
        # Create separate property files for bulk and shear modulus
        bulk_path = os.path.join(root_dir, 'id_prop_bulk.csv')
        shear_path = os.path.join(root_dir, 'id_prop_shear.csv')
        
        print(f"Downloading {len(mats)} materials...")
        
        success_count = 0
        for i, material in enumerate(mats):
            try:
                material_id = str(material.material_id)
                bulk_modulus = material.bulk_modulus.vrh if material.bulk_modulus else None
                shear_modulus = material.shear_modulus.vrh if material.shear_modulus else None
                
                if bulk_modulus is None and shear_modulus is None:
                    continue
                
                # Download CIF file
                structure = mpr.get_structure_by_material_id(material_id, final=True)
                cif_path = os.path.join(root_dir, f'{material_id}.cif')
                structure.to(filename=cif_path)
                
                # Write to property files
                if bulk_modulus is not None:
                    with open(bulk_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([material_id, bulk_modulus])
                
                if shear_modulus is not None:
                    with open(shear_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([material_id, shear_modulus])
                
                success_count += 1
                if (i + 1) % 100 == 0:
                    print(f"  Downloaded {i + 1}/{len(mats)} materials...")
                    
            except Exception as e:
                print(f"Failed to fetch data for {material_id}: {e}")
        
        print(f"Successfully downloaded {success_count} materials to {root_dir}")


def download_materials_with_energy_bandgap(api_key, root_dir, max_materials=None):
    """Download materials with thermodynamic and electronic structure data.
    
    Args:
        api_key: Materials Project API key
        root_dir: Directory to save CIF files and property CSV
        max_materials: Optional limit on number of materials to download
    """
    with MPRester(api_key) as mpr:
        print("Searching for materials with energy and band gap data...")
        mats = mpr.materials.summary.search(
            has_props=[HasProps.thermo, HasProps.electronic_structure], 
            fields=["material_id", "energy_per_atom", "band_gap"]
        )
        
        if max_materials:
            mats = mats[:max_materials]
        
        os.makedirs(root_dir, exist_ok=True)
        
        # Create separate property files
        formation_path = os.path.join(root_dir, 'id_prop_formation.csv')
        total_path = os.path.join(root_dir, 'id_prop_total.csv')
        bandgap_path = os.path.join(root_dir, 'id_prop_bandgap.csv')
        
        print(f"Downloading {len(mats)} materials...")
        
        success_count = 0
        for i, material in enumerate(mats):
            try:
                material_id = str(material.material_id)
                formation_energy = material.formation_energy_per_atom
                total_energy = material.energy_per_atom
                band_gap = material.band_gap
                
                # Download CIF file
                structure = mpr.get_structure_by_material_id(material_id, final=True)
                cif_path = os.path.join(root_dir, f'{material_id}.cif')
                structure.to(filename=cif_path)
                
                # Write to property files
                if formation_energy is not None:
                    with open(formation_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([material_id, formation_energy])
                
                if total_energy is not None:
                    with open(total_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([material_id, total_energy])
                
                if band_gap is not None:
                    with open(bandgap_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([material_id, band_gap])
                
                success_count += 1
                if (i + 1) % 100 == 0:
                    print(f"  Downloaded {i + 1}/{len(mats)} materials...")
                    
            except Exception as e:
                print(f"Failed to fetch data for {material_id}: {e}")
        
        print(f"Successfully downloaded {success_count} materials to {root_dir}")


def download_custom_materials(api_key, material_ids, property_name, root_dir):
    """Download specific materials by ID.
    
    Args:
        api_key: Materials Project API key
        material_ids: List of material IDs (e.g., ['mp-123', 'mp-456'])
        property_name: Name of property to fetch (e.g., 'formation_energy_per_atom')
        root_dir: Directory to save CIF files and property CSV
    """
    with MPRester(api_key) as mpr:
        os.makedirs(root_dir, exist_ok=True)
        id_prop_path = os.path.join(root_dir, 'id_prop.csv')
        
        print(f"Downloading {len(material_ids)} materials...")
        
        for material_id in material_ids:
            try:
                # Get property value
                material_data = mpr.summary.search(
                    material_ids=[material_id], 
                    fields=[property_name]
                )
                
                if not material_data:
                    print(f"No data found for {material_id}")
                    continue
                
                prop_value = getattr(material_data[0], property_name)
                
                # Download CIF file
                structure = mpr.get_structure_by_material_id(material_id, final=True)
                cif_path = os.path.join(root_dir, f'{material_id}.cif')
                structure.to(filename=cif_path)
                
                # Write to property file
                with open(id_prop_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([material_id, prop_value])
                
                print(f"  Downloaded {material_id}: {property_name}={prop_value}")
                
            except Exception as e:
                print(f"Failed to fetch data for {material_id}: {e}")
        
        print(f"Saved materials to {root_dir}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Download materials from Materials Project")
    parser.add_argument('--api_key', type=str, default=os.environ.get('MP_API_KEY', ''),
                        help="Materials Project API key (or set MP_API_KEY env var)")
    parser.add_argument('--output_dir', type=str, default='data/materials',
                        help="Output directory for CIF files and property CSV")
    parser.add_argument('--property', type=str, default='elasticity',
                        choices=['elasticity', 'energy', 'both'],
                        help="Property type to download")
    parser.add_argument('--max_materials', type=int, default=None,
                        help="Maximum number of materials to download")
    
    args = parser.parse_args()
    
    if not args.api_key:
        print("Error: Please provide an API key via --api_key or MP_API_KEY environment variable")
        print("Get your API key from https://materialsproject.org/")
        exit(1)
    
    if args.property in ['elasticity', 'both']:
        download_materials_with_elasticity(
            args.api_key, 
            os.path.join(args.output_dir, 'elastic') if args.property == 'both' else args.output_dir,
            args.max_materials
        )
    
    if args.property in ['energy', 'both']:
        download_materials_with_energy_bandgap(
            args.api_key,
            os.path.join(args.output_dir, 'energy') if args.property == 'both' else args.output_dir,
            args.max_materials
        )
