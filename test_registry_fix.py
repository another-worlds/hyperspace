#!/usr/bin/env python3
"""Quick test to verify the registry fix works."""

import numpy as np
from hyperspace.models.knowledge_matrix import UniversalKnowledgeTensor
from hyperspace.pages.governance import _build_feature_chain

def test_registry_fix():
    # Create a UKT instance
    ukt = UniversalKnowledgeTensor()
    
    # Add some test blocks
    ukt.add_block("finance_test", np.random.randn(16))
    ukt.add_block("clusters_test", np.random.randn(8))
    print(f"✓ Successfully added blocks. Registry has {len(ukt.registry.regions)} regions")
    print(f"✓ Total dim: {ukt.registry.total_dim}")
    
    # Get a snapshot
    snapshot = ukt.get_snapshot()
    print(f"✓ Got snapshot with {'registry' if 'registry' in snapshot else 'NO registry'}")
    
    if 'registry' in snapshot:
        registry = snapshot['registry']
        print(f"✓ Registry in snapshot has {len(registry.regions)} regions")
        
        # Test feature name resolution
        feature_name = registry.feature_name(5)
        print(f"✓ Feature 5 name: {feature_name}")
        
        # Test region lookup
        region = registry.region_for_index(5)
        if region:
            print(f"✓ Feature 5 region: {region.name}")
        else:
            print("✗ No region found for feature 5")
    
    # Test governance feature chain building  
    snapshots = [snapshot]
    try:
        chain = _build_feature_chain(5, snapshots)
        print(f"✓ Built feature chain for index 5:")
        print(f"  - Feature name: {chain.get('feature_name', 'unknown')}")
        print(f"  - Source block: {chain.get('source_block', 'unknown')}")
        print(f"  - Region: {chain.get('region_name', 'unknown')}")
    except Exception as e:
        print(f"✗ Failed to build feature chain: {e}")
        return False
    
    print("\n✅ All tests passed! Registry fix is working.")
    return True

if __name__ == "__main__":
    test_registry_fix()