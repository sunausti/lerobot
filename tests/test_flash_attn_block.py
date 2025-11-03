#!/usr/bin/env python
"""Test script to verify flash_attn is properly blocked."""

import os
import sys

# Set environment variables
os.environ["TRANSFORMERS_NO_FLASH_ATTN"] = "1"
os.environ["DISABLE_FLASH_ATTN"] = "1"

# Block flash_attn import
sys.modules['flash_attn'] = None
sys.modules['flash_attn_interface'] = None
sys.modules['flash_attn.flash_attn_interface'] = None

print("Environment variables set:")
print(f"  TRANSFORMERS_NO_FLASH_ATTN={os.environ.get('TRANSFORMERS_NO_FLASH_ATTN')}")
print(f"  DISABLE_FLASH_ATTN={os.environ.get('DISABLE_FLASH_ATTN')}")

print("\nBlocked modules:")
print(f"  sys.modules['flash_attn']={sys.modules.get('flash_attn')}")
print(f"  sys.modules['flash_attn_interface']={sys.modules.get('flash_attn_interface')}")

# Try to import flash_attn
try:
    import flash_attn
    print(f"\n❌ ERROR: flash_attn was imported! {flash_attn}")
except (ImportError, AttributeError) as e:
    print(f"\n✅ SUCCESS: flash_attn import blocked: {e}")

# Try to import transformers
try:
    from transformers import AutoConfig
    print("✅ transformers imported successfully")
except Exception as e:
    print(f"❌ ERROR importing transformers: {e}")

# Check if transformers detects flash_attn
try:
    from transformers.utils import is_flash_attn_2_available
    available = is_flash_attn_2_available()
    if available:
        print(f"❌ ERROR: transformers thinks flash_attn is available!")
    else:
        print(f"✅ SUCCESS: transformers correctly detects flash_attn as unavailable")
except Exception as e:
    print(f"Note: Could not check flash_attn availability: {e}")

print("\n" + "="*60)
print("Test complete. If all checks passed, flash_attn is blocked.")
print("="*60)
