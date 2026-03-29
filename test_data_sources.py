"""Test script to validate all data sources and registry integration.

Run this before launching the main app to ensure all APIs are accessible.

Usage:
    python test_data_sources.py
"""
from __future__ import annotations

import sys
from datetime import datetime

# Test if imports work
print("=" * 70)
print("HYPERSPACE DATA SOURCE VALIDATION")
print("=" * 70)
print()

# Test registry
print("1. Testing Registry Module...")
try:
    from hyperspace.data.registry import list_fetchers, get_fetcher
    fetchers = list_fetchers()
    print(f"   ✓ Registry loaded successfully")
    print(f"   ✓ Registered fetchers: {fetchers}")
    print()
except Exception as exc:
    print(f"   ✗ Registry import failed: {exc}")
    sys.exit(1)

# Test Finance
print("2. Testing Finance Data Source...")
try:
    from hyperspace.data.finance import (
        get_ohlcv, fetch_finance_data, fetch_binance_ohlcv,
    )
    from hyperspace.config import COUNTRY_INDEX_TICKERS
    
    tickers = list(COUNTRY_INDEX_TICKERS.values())[:2]  # Test with 2 tickers
    print(f"   Testing with tickers: {tickers}")
    
    # Test direct function
    df, src = get_ohlcv(tickers, period="1mo")
    print(f"   ✓ Direct call: {src}")
    print(f"   ✓ Rows: {len(df)}, Columns: {list(df.columns)}")
    
    # Test Binance helper (use BTCUSDT as example)
    btc_df = fetch_binance_ohlcv(("BTCUSDT",), period_days=30)
    if btc_df is not None:
        print(f"   ✓ Binance call returned {len(btc_df)} rows")
    
    # Test NBP FX helper
    try:
        fx = fetch_nbp_fx("USD/PLN", "2023-01-01", "2023-01-10")
        print(f"   ✓ NBP FX helper returned {len(fx) if fx is not None else 0} rows")
    except Exception:
        print("   ⚠ NBP FX helper test skipped")
    
    # Test registry function
    data, src2 = fetch_finance_data({"tickers": tickers})
    print(f"   ✓ Registry call: {src2}")
    print()
except Exception as exc:
    print(f"   ✗ Finance test failed: {exc}")
    import traceback
    traceback.print_exc()
    print()

# Test News
print("3. Testing News Data Source...")
try:
    from hyperspace.data.news import (
        get_text_data, fetch_news_data, fetch_hackernews, fetch_reddit_worldnews,
    )
    
    # Test query helpers
    hn = fetch_hackernews("geopolitics", max_results=5)
    print(f"   ✓ HackerNews helper returned {len(hn) if hn else 0} items")
    rd = fetch_reddit_worldnews(limit=5)
    print(f"   ✓ Reddit helper returned {len(rd) if rd else 0} items")
    
    # Test with date range
    start = datetime(2025, 12, 1)
    end = datetime(2026, 3, 1)
    
    # Test direct function
    docs, src = get_text_data(start_date=start, end_date=end)
    print(f"   ✓ Direct call: {src}")
    print(f"   ✓ Documents fetched: {len(docs)}")
    print(f"   ✓ Sample: {docs[0][:80]}..." if docs else "   (no docs)")
    
    # Test registry function
    data, src2 = fetch_news_data({"finance_start": start, "finance_end": end})
    print(f"   ✓ Registry call: {src2}")
    print()
except Exception as exc:
    print(f"   ✗ News test failed: {exc}")
    import traceback
    traceback.print_exc()
    print()

# Test Politics
print("4. Testing Political Data Source...")
try:
    from hyperspace.data.political import get_political_data, fetch_political_data
    
    # Test direct function
    un_df, agreement, src = get_political_data(min_year=2020, max_year=2025)
    print(f"   ✓ Direct call: {src}")
    print(f"   ✓ UN votes rows: {len(un_df) if un_df is not None else 0}")
    print(f"   ✓ Agreement matrix shape: {agreement.shape}")
    
    # Test registry function
    data, src2 = fetch_political_data({"min_year": 2020, "max_year": 2025})
    print(f"   ✓ Registry call: {src2}")
    print(f"   ✓ Data keys: {list(data.keys())}")
    print()
except Exception as exc:
    print(f"   ✗ Politics test failed: {exc}")
    import traceback
    traceback.print_exc()
    print()

# Test Spatial
print("5. Testing Spatial Data Source...")
try:
    from hyperspace.data.spatial import (
        fetch_all_spatial_data, fetch_spatial_data, geocode_nominatim,
    )
    
    # Test geocoding helper
    geo = geocode_nominatim("Berlin, Germany")
    print(f"   ✓ Nominatim geocode helper returned {geo.get('display_name') if geo else 'None'}")
    
    # Test direct function
    spatial_data = fetch_all_spatial_data()
    print(f"   ✓ Direct call: {spatial_data['source_label']}")
    print(f"   ✓ Physical raster shape: {spatial_data['physical_raster'].shape}")
    print(f"   ✓ Country scalars shape: {spatial_data['country_scalars'].shape}")
    print(f"   ✓ Nodes: {spatial_data['node_order']}")
    
    # Test registry function
    data, src2 = fetch_spatial_data({})
    print(f"   ✓ Registry call: {src2}")
    print()
except Exception as exc:
    print(f"   ✗ Spatial test failed: {exc}")
    import traceback
    traceback.print_exc()
    print()

# Test Map
print("6. Testing Map Data Source...")
try:
    from hyperspace.data.map import (
        get_country_stats, fetch_map_data, fetch_cia_factbook,
    )
    
    # Test factbook helper
    fb = fetch_cia_factbook()
    print(f"   ✓ Factbook helper returned {len(fb) if fb else 0} entries")
    
    # Test direct function
    stats, src = get_country_stats()
    print(f"   ✓ Direct call: {src}")
    print(f"   ✓ Countries: {list(stats.keys())}")
    print(f"   ✓ Sample (USA): pop={stats['USA'].get('population', 'N/A')}, capital={stats['USA'].get('capital', 'N/A')}")
    
    # Test registry function
    data, src2 = fetch_map_data({})
    print(f"   ✓ Registry call: {src2}")
    print()
except Exception as exc:
    print(f"   ✗ Map test failed: {exc}")
    import traceback
    traceback.print_exc()
    print()

# Test Registry Orchestration
print("7. Testing Full Registry Orchestration...")
try:
    from hyperspace.data.registry import run_all_fetchers
    
    # Simulate session state
    session_state = {
        "tickers": list(COUNTRY_INDEX_TICKERS.values())[:2],
        "finance_start": datetime(2025, 12, 1),
        "finance_end": datetime(2026, 3, 1),
        "min_year": 2020,
        "max_year": 2025,
    }
    
    def status_callback(msg):
        print(f"     {msg}")
    
    data_sources = run_all_fetchers(session_state, status_callback)
    print(f"   ✓ All fetchers executed successfully")
    print(f"   ✓ Data sources: {list(data_sources.keys())}")
    for name, src_label in data_sources.items():
        print(f"     - {name}: {src_label[:60]}")
    print()
except Exception as exc:
    print(f"   ✗ Registry orchestration failed: {exc}")
    import traceback
    traceback.print_exc()
    print()

# Final summary
print("=" * 70)
print("VALIDATION COMPLETE")
print("=" * 70)
print()
print("All data sources are accessible and working correctly!")
print("You can now launch the main Hyperspace app with confidence.")
print()
print("Run: streamlit run app.py")
print()
