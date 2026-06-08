#!/usr/bin/env python3
"""
评估脚本功能验证脚本。

验证 eval_direct.py, eval_libero_plus.py, run_eval.py 的功能是否正确。
"""

import subprocess
import sys
import pathlib

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent

def test_help(script_name):
    """测试脚本的 help 功能。"""
    print(f"\n{'='*60}")
    print(f"测试 {script_name} --help")
    print(f"{'='*60}")
    result = subprocess.run(
        [sys.executable, str(SCRIPT_DIR / script_name), "--help"],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        print(f"✅ {script_name} --help 成功")
        return True
    else:
        print(f"❌ {script_name} --help 失败")
        print(result.stderr)
        return False

def check_nfe_options(script_name, expected_nfe):
    """检查脚本是否支持预期的 NFE 选项。"""
    result = subprocess.run(
        [sys.executable, str(SCRIPT_DIR / script_name), "--help"],
        capture_output=True,
        text=True
    )
    help_text = result.stdout + result.stderr

    print(f"\n检查 {script_name} NFE 选项:")
    print(f"  期望: {expected_nfe}")

    for nfe in expected_nfe:
        if str(nfe) in help_text:
            print(f"  ✅ NFE={nfe} 支持")
        else:
            print(f"  ❌ NFE={nfe} 不支持")
            return False
    return True

def check_preset_options(script_name, expected_presets):
    """检查脚本是否支持预期的 Preset 选项。"""
    result = subprocess.run(
        [sys.executable, str(SCRIPT_DIR / script_name), "--help"],
        capture_output=True,
        text=True
    )
    help_text = result.stdout + result.stderr

    print(f"\n检查 {script_name} Preset 选项:")
    print(f"  期望: {expected_presets}")

    for preset in expected_presets:
        if preset in help_text:
            print(f"  ✅ Preset={preset} 支持")
        else:
            print(f"  ❌ Preset={preset} 不支持")
            return False
    return True

def main():
    print("="*60)
    print("评估脚本功能验证")
    print("="*60)

    all_passed = True

    # 测试 eval_direct.py
    print("\n" + "="*60)
    print("测试 eval_direct.py")
    print("="*60)

    all_passed &= test_help("eval_direct.py")
    all_passed &= check_nfe_options("eval_direct.py", [1, 2, 4, 10])
    all_passed &= check_preset_options("eval_direct.py", ["quick", "preset", "fullset"])

    # 测试 eval_libero_plus.py
    print("\n" + "="*60)
    print("测试 eval_libero_plus.py")
    print("="*60)

    all_passed &= test_help("eval_libero_plus.py")
    all_passed &= check_nfe_options("eval_libero_plus.py", [1, 2, 4, 10])
    all_passed &= check_preset_options("eval_libero_plus.py", ["quick", "medium", "full", "full90"])

    # 测试 run_eval.py
    print("\n" + "="*60)
    print("测试 run_eval.py")
    print("="*60)

    all_passed &= test_help("run_eval.py")

    # 测试 run_eval.py 的 dataset 选项
    result = subprocess.run(
        [sys.executable, str(SCRIPT_DIR / "run_eval.py"), "--help"],
        capture_output=True,
        text=True
    )
    help_text = result.stdout + result.stderr

    print("\n检查 run_eval.py dataset 选项:")
    for dataset in ["libero", "libero-plus"]:
        if dataset in help_text:
            print(f"  ✅ Dataset={dataset} 支持")
        else:
            print(f"  ❌ Dataset={dataset} 不支持")
            all_passed = False

    # 最终结果
    print("\n" + "="*60)
    if all_passed:
        print("✅ 所有验证通过！")
        return 0
    else:
        print("❌ 部分验证失败！")
        return 1

if __name__ == "__main__":
    sys.exit(main())
